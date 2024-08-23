import logging
import math

import torch
import torch.nn as nn
from fairseq import checkpoint_utils, utils
from fairseq.data import data_utils as fairseq_data_utils
from fairseq.models.transformer import Embedding
from fairseq.models import (
    FairseqEncoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.sign_to_text import (
    S2TTransformerModel,
    S2TTransformerEncoder,
)
from fairseq.models.sign_to_text.modules import Adapter, CTC, DynamicLinearCombination
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
)

logger = logging.getLogger(__name__)


@register_model("sign2text_sate")
class S2TSATEModel(S2TTransformerModel):
    """Adapted Transformer model (https://arxiv.org/abs/1706.03762) for
    speech-to-text tasks. The Transformer encoder/decoder remains the same.
    A trainable input subsampler is prepended to the Transformer encoder to
    project inputs into the encoder dimension as well as downsample input
    sequence for computational efficiency."""

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        S2TTransformerModel.add_args(parser)
        S2TSATEModel.add_specific_args(parser)

    @staticmethod
    def add_specific_args(parser):
        # SATE setting
        parser.add_argument(
            "--text-encoder-layers",
            default=6,
            type=int,
            help="layers of the text encoder",
        )
        parser.add_argument(
            "--text-attention-type",
            default="selfattn",
            type=str,
            help="attention type of the textual encoder",
        )
        parser.add_argument(
            "--adapter",
            default="league",
            type=str,
            help="adapter type",
        )
        parser.add_argument(
            "--ctc-compress-strategy",
            default="avg",
            type=str,
            help="compress strategy, such as avg, weighted, and softmax",
        )
        parser.add_argument(
            "--share-adapter-and-ctc",
            default=False,
            action="store_true",
            help="share the projection weights of the adapter and ctc",
        )
        parser.add_argument(
            "--share-adapter-and-embed",
            default=False,
            action="store_true",
            help="share the projection weights of the adapter and embed",
        )
        parser.add_argument(
            "--adapter-temperature",
            default=1.0,
            type=float,
            help="temperature of the CTC softmax in adapter",
        )
        parser.add_argument(
            "--adapter-embed-norm",
            default=False,
            action="store_true",
            help="use the layer norm for embed output",
        )
        parser.add_argument(
            "--adapter-out-norm",
            default=False,
            action="store_true",
            help="use the layer norm for final output",
        )
        parser.add_argument(
            "--acoustic-encoder",
            default="transformer",
            type=str,
            help="the architecture of the acoustic encoder",
        )
        parser.add_argument(
            "--load-pretrained-acoustic-encoder-from",
            type=str,
            metavar="STR",
            help="model to take acoustic encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--load-pretrained-text-encoder-from",
            type=str,
            metavar="STR",
            help="model to take text encoder weights from (for initialization)",
        )
        # target CTC
        parser.add_argument(
            "--target-sae-adapter",
            type=str,
            help="adapter type of target sae ",
        )
        parser.add_argument(
            "--target-ctc-layer",
            default=0,
            type=int,
            help="ctc layer for target sentence",
        )
        parser.add_argument(
            "--share-target-ctc-and-embed",
            action="store_true",
            help="share the weight of target ctc and embed",
        )
        parser.add_argument(
            "--target-interleaved-ctc-layers",
            default=None,
            type=str,
            help="interleaved ctc layers for target sentence",
        )
        parser.add_argument(
            "--share-target-sae-and-ctc",
            action="store_true",
            help="share the weight of target sae and ctc",
        )
        # freeze
        parser.add_argument(
            "--freeze-acoustic-encoder",
            action="store_true",
            help="freeze the parameters of the acoustic encoder",
        )
        parser.add_argument(
            "--freeze-textual-encoder",
            action="store_true",
            help="freeze the parameters of the acoustic encoder",
        )
        parser.add_argument(
            "--freeze-decoder",
            action="store_true",
            help="freeze the parameters of the decoder",
        )

    @classmethod
    def build_encoder(cls, args, task=None, decoder_embed_tokens=None):
        encoder = S2TSATEEncoder(args, task, decoder_embed_tokens)

        if getattr(args, "load_pretrained_encoder_from", None):
            logger.info(
                f"loaded pretrained acoustic encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder,
                checkpoint=args.load_pretrained_encoder_from,
                strict=False
            )

        if getattr(args, "load_pretrained_acoustic_encoder_from", None):
            logger.info(
                f"loaded pretrained acoustic encoder from: "
                f"{args.load_pretrained_acoustic_encoder_from}"
            )
            encoder.acoustic_encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder.acoustic_encoder,
                checkpoint=args.load_pretrained_acoustic_encoder_from,
                strict=False
            )

        if getattr(args, "load_pretrained_text_encoder_from", None):
            logger.info(
                f"loaded pretrained text encoder from: "
                f"{args.load_pretrained_text_encoder_from}"
            )
            encoder.textual_encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder.textual_encoder,
                checkpoint=args.load_pretrained_text_encoder_from,
                strict=False
            )
        if args.share_adapter_and_ctc and hasattr(encoder.adapter, "embed_adapter"):
            encoder.adapter.embed_adapter.weight = encoder.acoustic_encoder.ctc.ctc_projection.weight

        return encoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )
        encoder = cls.build_encoder(args, task, decoder_embed_tokens)
        if getattr(args, "encoder_freeze_module", None):
            utils.freeze_parameters(encoder, args.encoder_freeze_module)
            logging.info("freeze the encoder module: {}".format(args.encoder_freeze_module))

        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        if getattr(args, "decoder_freeze_module", None):
            utils.freeze_parameters(decoder, args.decoder_freeze_module)
            logging.info("freeze the decoder module: {}".format(args.decoder_freeze_module))

        if args.share_adapter_and_embed and hasattr(encoder.adapter, "embed_adapter"):
            encoder.adapter.embed_adapter.weight = decoder_embed_tokens.weight

        return cls(encoder, decoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return decoder_out


class TextualEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embed_tokens=None):

        super().__init__(None)

        self.register_buffer("version", torch.Tensor([3]))  # for consistent
        embed_dim = args.encoder_embed_dim
        layer_num = args.text_encoder_layers
        self.layer_num = layer_num
        self.embed_tokens = embed_tokens

        self.embed_scale = math.sqrt(embed_dim)
        if args.encoder_no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = dictionary.pad_index

        self.encoder_embed_norm = getattr(args, "encoder_embed_norm", False)
        if self.encoder_embed_norm:
            self.embed_ln = LayerNorm(embed_dim)

        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )

        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, args.encoder_embed_dim, self.padding_idx
        )

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for _ in range(layer_num)]
        )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

        # CTC
        self.use_ctc = getattr(args, "target_ctc_weight", 0) > 0
        if self.use_ctc:
            self.ctc_layer = getattr(args, "target_ctc_layer", layer_num)
            if self.ctc_layer == 0:
                self.ctc_layer = layer_num
            self.inter_ctc = True if self.ctc_layer != layer_num else False
            if self.inter_ctc:
                logger.info("Target CTC loss in layer %d" % self.ctc_layer)
            self.ctc = CTC(embed_dim,
                           dictionary_size=embed_tokens.num_embeddings if embed_tokens is not None else len(dictionary),
                           dropout=args.dropout,
                           need_layernorm=True if self.inter_ctc else False)

            if embed_tokens is not None and args.share_target_ctc_and_embed and \
                    self.ctc.ctc_projection.weight.size() == embed_tokens.weight.size():
                self.ctc.ctc_projection.weight = embed_tokens.weight

        self.interleaved_ctc_drop_prob = args.interleaved_ctc_drop_prob
        self.interleaved_ctc_layers = []
        self.target_interleaved_ctc_layers = getattr(args, "target_interleaved_ctc_layers", None)
        self.sae_ground_truth_ratio = getattr(args, "sae_ground_truth_ratio", 0)

        if self.target_interleaved_ctc_layers is not None:
            target_interleaved_ctc_layers = self.target_interleaved_ctc_layers.split(",")
            for layer_idx in target_interleaved_ctc_layers:
                layer_idx = int(layer_idx)
                assert layer_idx <= layer_num, (layer_idx, layer_num)

                if layer_idx <= 0:
                    layer_idx += layer_num
                self.interleaved_ctc_layers.append(layer_idx)

                logger.info("Interleaved target CTC loss in layer %d" % layer_idx)

            if not self.use_ctc:
                self.ctc = CTC(embed_dim,
                               dictionary_size=len(dictionary),
                               dropout=args.dropout)
                if embed_tokens is not None and args.share_target_ctc_and_embed and \
                        self.ctc.ctc_projection.weight.size() == embed_tokens.weight.size():
                    self.ctc.ctc_projection.weight = embed_tokens.weight

            strategy = {
                "embed_norm": getattr(args, "sae_embed_norm", False),
                "out_norm": getattr(args, "sae_out_norm", False),
                "ctc_compress_strategy": getattr(args, "ctc_compress_strategy", None),
                "ctc_temperature": getattr(args, "sae_ctc_temperature", 1.0),
                "distribution_cutoff": getattr(args, "sae_distribution_cutoff", None),
                "gumbel": getattr(args, "sae_gumbel", False),
                "distribution_hard": getattr(args, "sae_distribution_hard", None),
                "drop_prob": getattr(args, "sae_drop_prob", 0),
            }

            self.sae = Adapter(embed_dim, args.target_sae_adapter,
                               len(dictionary),
                               strategy=strategy)
            if args.share_target_sae_and_ctc and hasattr(self.sae, "embed_adapter"):
                self.sae.embed_adapter.weight = self.ctc.ctc_projection.weight

            self.interleaved_ctc_drop_prob = args.interleaved_ctc_drop_prob

    def forward(self, x, encoder_padding_mask=None, history=None, **kwargs):

        if self.encoder_embed_norm:
            x = self.embed_ln(x)
        x = self.embed_scale * x
        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        x = positions + x
        x = self.dropout_module(x)

        target_ctc_logit = None
        target_interleaved_ctc_logits = []
        layer_idx = 0
        for layer in self.layers:
            if history is not None:
                x = history.pop()
            x = layer(x, encoder_padding_mask, pos_emb=positions)
            layer_idx += 1

            if self.use_ctc and self.inter_ctc and self.ctc_layer == layer_idx:
                target_ctc_logit = self.ctc(x.clone(), encoder_padding_mask, "Target Layer %d" % layer_idx)

            if layer_idx != self.layer_num and layer_idx in self.interleaved_ctc_layers:
                if self.interleaved_ctc_drop_prob > 0:
                    p = torch.rand(1).uniform_()
                    if p < self.interleaved_ctc_drop_prob:
                        break

                norm_x = self.layer_norm(x)
                logit = self.ctc(norm_x, encoder_padding_mask, "Target Layer %d" % layer_idx)
                target_interleaved_ctc_logits.append(logit)

                # CTC alignment
                oracle = None
                oracle_mask = None
                force_emit = None
                if self.sae_ground_truth_ratio > 0:
                    ctc_alignment_oracle = kwargs.get("ctc_alignment_oracle", None)
                    if ctc_alignment_oracle is not None and ctc_alignment_oracle["target"] is not None:
                        oracle, best_aligns_pad = ctc_alignment_oracle["target"]
                        oracle_mask = (torch.rand(oracle.size(),
                                                  device=oracle.device) < self.sae_ground_truth_ratio).bool()
                        force_emit = best_aligns_pad.masked_fill(~oracle_mask, -1)

                if self.sae.adapter_type != "none":
                    x, encoder_padding_mask = self.sae([norm_x, logit], encoder_padding_mask, oracle, oracle_mask)

            if history is not None:
                history.push(x)

        if history is not None:
            x = history.pop()

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if self.use_ctc and target_ctc_logit is None:
            target_ctc_logit = self.ctc(x, encoder_padding_mask, "Target output", is_top=True)

        return x, target_ctc_logit, target_interleaved_ctc_logits

    def reorder_encoder_out(self, encoder_out, new_order):
        pass


class S2TSATEEncoder(FairseqEncoder):
    """Speech-to-text Conformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args, task=None, decoder_embed_tokens=None):
        super().__init__(None)

        # acoustic encoder
        acoustic_encoder_type = args.acoustic_encoder
        if acoustic_encoder_type == "transformer":
            self.acoustic_encoder = S2TTransformerEncoder(args, task, decoder_embed_tokens)
        else:
            logging.error("Unsupported model arch {}!".format(acoustic_encoder_type))

        # adapter
        self.adapter_temperature = args.adapter_temperature
        strategy = {
            "embed_norm": getattr(args, "adapter_embed_norm", False),
            "out_norm": getattr(args, "adapter_out_norm", False),
            "ctc_compress_strategy": getattr(args, "ctc_compress_strategy", None),
            "distribution_cutoff": getattr(args, "adapter_distribution_cutoff", None),
            "drop_prob": getattr(args, "adapter_drop_prob", 0),
        }

        self.adapter = Adapter(args.encoder_embed_dim,
                               args.adapter,
                               len(task.source_dictionary),
                               strategy=strategy)

        assert not (args.share_adapter_and_ctc and args.share_adapter_and_embed), "Can not be True at the same time"
        if args.share_adapter_and_ctc and hasattr(self.adapter, "embed_adapter"):
            self.adapter.embed_adapter.weight = self.acoustic_encoder.ctc.ctc_projection.weight
        if args.share_adapter_and_embed and hasattr(self.adapter, "embed_adapter"):
            self.adapter.embed_adapter.weight = decoder_embed_tokens.weight

        acoustic_encoder_attention_type = args.encoder_attention_type
        args.encoder_attention_type = args.text_attention_type
        # textual encoder
        self.textual_encoder = TextualEncoder(args, task.source_dictionary, decoder_embed_tokens)

        args.encoder_attention_type = acoustic_encoder_attention_type

        self.freeze_acoustic_encoder = getattr(args, "freeze_acoustic_encoder", False)
        self.freeze_textual_encoder = getattr(args, "freeze_textual_encoder", False)
        self.sae_ground_truth_ratio = getattr(args, "sae_ground_truth_ratio", 0)

        if getattr(args, "use_enc_dlcl", False):
            layer_num = args.encoder_layers + args.text_encoder_layers + 2
            self.history = DynamicLinearCombination(args, is_encoder=True, layer_num=layer_num)
        else:
            self.history = None

    def set_ctc_infer(self, ctc_infer, post_process, src_dict=None, tgt_dict=None, path=None):
        if hasattr(self.acoustic_encoder, "ctc"):
            assert src_dict is not None
            logger.info("Acoustic Encoder CTC Inference")
            self.acoustic_encoder.ctc.set_infer(ctc_infer, post_process, src_dict,
                                                path=path + ".src_ctc" if path is not None else None)
            # path=os.path.join(path, "src_ctc") if path is not None else None)
        if hasattr(self.textual_encoder, "ctc"):
            assert tgt_dict is not None
            logger.info("Textual Encoder CTC Inference")
            self.textual_encoder.ctc.set_infer(ctc_infer, post_process, tgt_dict,
                                               path=path + ".tgt_ctc" if path is not None else None)
            # path=os.path.join(path, "tgt_ctc") if path is not None else None)

    def ctc_valid(self, lprobs, targets, input_lengths, dictionary, lang="source"):
        if lang == "source":
            if hasattr(self.acoustic_encoder, "ctc"):
                return self.acoustic_encoder.ctc.valid(lprobs, targets, input_lengths, dictionary)
            else:
                logger.error("No ctc module in textual encoder")
        else:
            if hasattr(self.textual_encoder, "ctc"):
                return self.textual_encoder.ctc.valid(lprobs, targets, input_lengths, dictionary)
            else:
                logger.error("No ctc module in textual encoder")

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        if self.history is not None:
            self.history.clean()

        if self.freeze_acoustic_encoder:
            with torch.no_grad():
                acoustic_encoder_out = self.acoustic_encoder(src_tokens, src_lengths, **kwargs)
        else:
            acoustic_encoder_out = self.acoustic_encoder(src_tokens, src_lengths, **kwargs)

        encoder_out = acoustic_encoder_out["encoder_out"][0]
        encoder_padding_mask = acoustic_encoder_out["encoder_padding_mask"][0]
        ctc_padding_mask = encoder_padding_mask
        if "mixup" in acoustic_encoder_out:
            mixup = acoustic_encoder_out["mixup"]
        else:
            mixup = None

        if "ctc_logit" in acoustic_encoder_out and len(acoustic_encoder_out["ctc_logit"]) > 0:
            ctc_logit = acoustic_encoder_out["ctc_logit"][0]
            # ctc_prob = F.softmax(ctc_logit / self.adapter_temperature, dim=-1, dtype=torch.float32)
        else:
            ctc_logit = None
            # ctc_prob = None
        x = (encoder_out, ctc_logit)

        x, encoder_padding_mask = self.adapter(x, encoder_padding_mask)

        if self.history is not None:
            acoustic_history = self.acoustic_encoder.history
            layer_num = acoustic_history.layer_num
            idx = torch.arange(layer_num).unsqueeze(0).T.repeat(1, layer_num).to(x.device).unsqueeze(2)
            self.history.weight.scatter(0, idx, acoustic_history.weight)
            self.history.layers.extend(acoustic_history.layers)
            self.history.count = acoustic_history.count

            self.history.push(x)

        if self.freeze_textual_encoder:
            with torch.no_grad():
                x, target_ctc_logit, target_interleaved_ctc_logits = self.textual_encoder(x, encoder_padding_mask,
                                                                                          self.history, **kwargs)
        else:
            x, target_ctc_logit, target_interleaved_ctc_logits = self.textual_encoder(x, encoder_padding_mask,
                                                                                      self.history, **kwargs)

        return {
            "encoder_out": [x],  # T x B x C
            "ctc_logit": [ctc_logit],  # T x B x C
            "interleaved_ctc_logits": acoustic_encoder_out.get("interleaved_ctc_logits", []),  # B x T x C
            "target_ctc_logit": [target_ctc_logit],  # B x T x C
            "target_interleaved_ctc_logits": target_interleaved_ctc_logits,  # B x T x C
            "ctc_padding_mask": [ctc_padding_mask],  # B x T
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "mixup": mixup,
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        new_ctc_logit = (
            [] if len(encoder_out["ctc_logit"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["ctc_logit"]]
        )

        new_encoder_out = (
            [] if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            [] if len(encoder_out["encoder_padding_mask"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_padding_mask"]]
        )

        new_encoder_embedding = (
            [] if len(encoder_out["encoder_embedding"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "ctc_logit": new_ctc_logit,
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }


@register_model_architecture(model_name="sign2text_sate", arch_name="sign2text_sate")
def base_architecture(args):
    # Convolutional subsampler

    args.input_channels = getattr(args, "input_channels", 1)
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 1024)

    args.subsampling_type = getattr(args, "subsampling_type", "conv1d")
    args.subsampling_layers = getattr(args, "subsampling_layers", 2)
    args.subsampling_filter = getattr(args, "subsampling_filter", 1024)
    args.subsampling_kernel = getattr(args, "subsampling_kernel", 5)
    args.subsampling_stride = getattr(args, "subsampling_stride", 2)
    args.subsampling_norm = getattr(args, "subsampling_norm", "none")
    args.subsampling_activation = getattr(args, "subsampling_activation", "glu")

    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_type = getattr(args, "encoder_attention_type", "selfattn")
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_type = getattr(args, "decoder_attention_type", "selfattn")
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)

    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.encoder_no_scale_embedding = getattr(args, "encoder_no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    args.encoder_embed_linear = getattr(args, "encoder_embed_linear", False)
    args.encoder_embed_norm = getattr(args, "encoder_embed_norm", False)

    # CTC
    args.ctc_layer = getattr(args, "ctc_layer", 0)
    args.target_ctc_layer = getattr(args, "target_ctc_layer", 0)
    args.share_ctc_and_embed = getattr(args, "share_ctc_and_embed", False)
    args.share_target_ctc_and_embed = getattr(args, "share_target_ctc_and_embed", False)

    # Conformer
    args.encoder_activation_fn = getattr(args, "encoder_activation_fn", "relu")
    args.macaron_style = getattr(args, "macaron_style", False)
    args.use_cnn_module = getattr(args, "use_cnn_module", False)
    args.cnn_module_kernel = getattr(args, "cnn_module_kernel", 31)
    args.cnn_module_norm = getattr(args, "cnn_module_norm", "batch_norm")

    # settings for DLCL
    args.use_enc_dlcl = getattr(args, "use_enc_dlcl", False)
    args.use_dec_dlcl = getattr(args, "use_dec_dlcl", False)
    args.init_value = getattr(args, 'init_value', 'avg')
    args.weight_type = getattr(args, 'weight_type', 'scalar')
    args.encoder_learnable = getattr(args, 'encoder_learnable', True)
    args.decoder_learnable = getattr(args, 'decoder_learnable', True)
    args.normalize_embed = getattr(args, 'normalize_embed', False)
    args.history_dropout = getattr(args, 'history_dropout', 0.0)
    args.history_window_size = getattr(args, 'history_window_size', -1)

    # Relative position encoding
    args.max_encoder_relative_length = getattr(args, 'max_encoder_relative_length', -1)
    args.max_decoder_relative_length = getattr(args, 'max_decoder_relative_length', -1)
    args.k_only = getattr(args, 'k_only', True)

    # local modeling
    args.hard_mask_window = getattr(args, 'hard_mask_window', 0)
    args.gauss_mask_sigma = getattr(args, 'gauss_mask_sigma', 0)
    args.init_mask_weight = getattr(args, 'init_mask_weight', 0)

    # interleaved CTC
    args.interleaved_ctc_layers = getattr(args, "interleaved_ctc_layers", None)
    args.interleaved_ctc_temperature = getattr(args, "interleaved_ctc_temperature", 1)
    args.interleaved_ctc_drop_prob = getattr(args, "interleaved_ctc_drop_prob", 0)

    # Semantics-augmented Encoding (sae)
    args.sae_adapter = getattr(args, "sae_adapter", "none")
    args.target_sae_adapter = getattr(args, "target_sae_adapter", args.sae_adapter)
    args.share_sae_and_ctc = getattr(args, "share_sae_and_ctc", False)
    args.share_target_sae_and_ctc = getattr(args, "share_target_sae_and_ctc", False)
    args.sae_drop_prob = getattr(args, "sae_drop_prob", 0)
    args.sae_distribution_cutoff = getattr(args, "sae_distribution_cutoff", None)
    args.sae_distribution_hard = getattr(args, "sae_distribution_hard", False)
    args.sae_gumbel = getattr(args, "sae_gumbel", False)

    # mixup
    args.inter_mixup = getattr(args, "inter_mixup", False)
    args.inter_mixup_layer = getattr(args, "inter_mixup_layer", None)
    args.inter_mixup_beta = getattr(args, "inter_mixup_beta", 0.5)
    args.inter_mixup_prob = getattr(args, "inter_mixup_prob", 1)
    args.inter_mixup_ratio = getattr(args, "inter_mixup_ratio", 1)

    # PDS
    args.pds_stages = getattr(args, "pds_stages", None)
    args.pds_layers = getattr(args, "pds_layers", None)
    args.pds_ratios = getattr(args, "pds_ratios", None)

    args.pds_ds_method = getattr(args, "pds_ds_method", "conv")
    args.pds_embed_dims = getattr(args, "pds_embed_dims", None)
    args.pds_embed_norm = getattr(args, "pds_embed_norm", False)
    args.pds_position_embed = getattr(args, "pds_position_embed", None)

    args.pds_attn_heads = getattr(args, "pds_attn_heads", None)
    args.pds_ffn_ratios = getattr(args, "pds_ffn_ratios", None)
    args.pds_cnn_kernel_sizes = getattr(args, "pds_cnn_kernel_sizes", None)

    args.pds_attn_ds_ratios = getattr(args, "pds_attn_ds_ratios", None)
    args.pds_conv_strides = getattr(args, "pds_conv_strides", None)
    args.pds_attn_strides = getattr(args, "pds_attn_strides", None)

    args.ctc_layer = getattr(args, "ctc_layer", 0)
    args.pds_dropout = getattr(args, "pds_dropout", args.dropout)

    args.pds_fusion = getattr(args, "pds_fusion", False)
    args.pds_fusion_method = getattr(args, "pds_fusion_method", "all_conv")

    # SATE
    args.acoustic_encoder = getattr(args, "acoustic_encoder", "transformer")
    args.adapter = getattr(args, "adapter", "league")
    args.ctc_compress_strategy = getattr(args, "ctc_compress_strategy", "avg")
    args.adapter_temperature = getattr(args, "adapter_temperature", 1.0)
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 6)
    args.text_attention_type = getattr(args, "text_attention_type", "selfattn")
    args.share_adapter_and_ctc = getattr(args, "share_adapter_and_ctc", False)
    args.share_adapter_and_embed = getattr(args, "share_adapter_and_embed", False)
    args.adapter_embed_norm = getattr(args, "adapter_embed_norm", False)
    args.adapter_out_norm = getattr(args, "adapter_out_norm", False)


@register_model_architecture("sign2text_sate", "sign2text_sate_s")
def sign2text_sate_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)


@register_model_architecture("sign2text_sate", "sign2text_sate_s_relative")
def sign2text_sate_s_relative(args):
    args.encoder_attention_type = "relative"
    args.decoder_attention_type = "relative"
    args.max_encoder_relative_length = 100
    args.max_decoder_relative_length = 20
    args.k_only = True
    sign2text_sate_s(args)


@register_model_architecture("sign2text_sate", "sign2text_sate_xs")
def sign2text_sate_xs(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 3)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 4)
    args.dropout = getattr(args, "dropout", 0.3)
    sign2text_sate_s(args)


@register_model_architecture("sign2text_sate", "sign2text_sate_m")
def sign2text_sate_m(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    base_architecture(args)


@register_model_architecture("sign2text_sate", "sign2text_sate_l")
def sign2text_sate_l(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.2)
    base_architecture(args)
