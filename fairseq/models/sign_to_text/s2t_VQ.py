import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from random import choice

import torch.nn.functional as F
from fairseq.modules.fairseq_dropout import FairseqDropout

import torch
import torch.nn as nn
from fairseq.data import data_utils as fairseq_data_utils
from fairseq import checkpoint_utils, utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)

from fairseq.models.sign_to_text import (
    S2TTransformerModel,
    S2TTransformerEncoder,
    S2TSATEModel,
    S2TSATEEncoder,
)
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    RelPositionalEncoding,
    TransformerEncoderLayer,
)
from fairseq.models.sign_to_text.modules import (
    Adapter,
    CTC,
    subsampling,
    Shrink,
    DynamicLinearCombination,
    TransformerS2EncoderLayer,
    S2TTransformerEncoderLayer,
    VQMemory,
    TransformerIMMTEncoderLayer,
)
from fairseq.models.transformer import Embedding, TransformerDecoder, TransformerEncoder#, TransformerMixUpEncoder
from torch import Tensor
logger = logging.getLogger(__name__)

@register_model("sign2text_vq")
class S2TvqModel(FairseqEncoderDecoderModel):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)


    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        S2TTransformerModel.add_args(parser)
        S2TvqModel.add_specific_args(parser)


    @staticmethod
    def add_specific_args(parser):
        # multi-encoder
        parser.add_argument(
            "--asr-encoder",
            default="transformer",
            choices=["transformer", "pds", "sate", "wav2vec"],
            type=str,
            help="the architecture of the ASR encoder",
        )
        parser.add_argument(
            "--mt-encoder",
            default="transformer",
            type=str,
            help="the architecture of the MT encoder",
        )
        parser.add_argument(
            "--mt-encoder-dim",
            type=int,
            help="the dimension of the MT encoder",
        )
        parser.add_argument(
            "--mt-encoder-layers",
            default=6,
            type=int,
            help="the layers of the MT encoder",
        )
        # collaboration
        parser.add_argument(
            "--encoder-collaboration-mode",
            default="none",
            type=str,
            help="how to calculate attention during league in encoder",
        )
        parser.add_argument(
            "--decoder-collaboration-mode",
            default="none",
            type=str,
            help="how to calculate attention during league in encoder",
        )

        # league
        parser.add_argument(
            "--encoder-league-s1-ratio",
            default=0.5,
            type=float,
            help="league ratio of the s1 representation",
        )
        parser.add_argument(
            "--encoder-league-s2-ratio",
            default=0.5,
            type=float,
            help="league ratio of the s2 representation",
        )
        parser.add_argument(
            "--encoder-league-drop-net",
            action="store_true",
            help="drop one input during league",
        )
        parser.add_argument(
            "--encoder-league-drop-net-prob",
            default=0.0,
            type=float,
            help="probability of dropping one representations",
        )
        parser.add_argument(
            "--encoder-league-drop-net-mix",
            action="store_true",
            help="mix the two input with any probability",
        )
        parser.add_argument(
            "--decoder-league-s1-ratio",
            default=0.5,
            type=float,
            help="league ratio of the s1 representation",
        )
        parser.add_argument(
            "--decoder-league-s2-ratio",
            default=0.5,
            type=float,
            help="league ratio of the s2 representation",
        )
        parser.add_argument(
            "--decoder-league-drop-net",
            action="store_true",
            help="drop one input during league",
        )
        parser.add_argument(
            "--decoder-league-drop-net-prob",
            default=0.0,
            type=float,
            help="probability of dropping one representations",
        )
        parser.add_argument(
            "--decoder-league-drop-net-mix",
            action="store_true",
            help="mix the two input with any probability",
        )

        parser.add_argument(
            "--load-pretrained-asr-encoder-from",
            type=str,
            metavar="STR",
            help="model to take asr encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--load-pretrained-mt-encoder-from",
            type=str,
            metavar="STR",
            help="model to take mt encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--inter-gloss-mixup-layer",
            default="0",
            type=str,
            help="the gloss encoder layers to apply mixup",
        )
        parser.add_argument(
            "--fuse-layer-num",
            default="0",
            type=int,
            help="",
        )
        parser.add_argument(
            "--fuse-mode",
            default="",
            type=str,
            help="",
        )
        # vq memory
        parser.add_argument(
            "--memory-num",
            type=int,
            help="number of shared memory module"
        )
        parser.add_argument(
            "--vq-vars",
            type=int,
            help="number of vq vars"
        )
        parser.add_argument(
            "--vq-groups",
            type=int,
            help="number of vq groups"
        )
        parser.add_argument(
            "--vq-temp",
            type=Tuple[float, float, float],
            help="can be tuple of 3 values (start, end, decay)"
        )
        parser.add_argument(
            "--weight-proj-depth",
            type=int,
            help="weight proj depth in codebook"
        )
        parser.add_argument(
            "--weight-proj-factor",
            type=int,
            help="weight proj factor in codebook"
        )
        pass

    @classmethod
    def build_encoder(cls, args, task=None, embed_tokens=None):
        encoder = S2TvqEncoder(args, task, embed_tokens)

        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from, strict=False
            )
            logger.info(
                f"loaded pretrained encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
        if getattr(args, "load_pretrained_asr_encoder_from", None):
            encoder.asr_encoder = checkpoint_utils.load_sign_pretrained_component_from_model(
                component=encoder.asr_encoder, checkpoint=args.load_pretrained_asr_encoder_from, strict=True
            )
            logger.info(
                f"loaded pretrained asr encoder from: "
                f"{args.load_pretrained_asr_encoder_from}"
            )
        if getattr(args, "load_pretrained_mt_encoder_from", None):
            encoder.mt_encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder.mt_encoder, checkpoint=args.load_pretrained_mt_encoder_from, strict=False
            )
            logger.info(
                f"loaded pretrained mt encoder from: "
                f"{args.load_pretrained_mt_encoder_from}"
            )
        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):

        decoder = TransformerDecoder(args, task.target_dictionary, embed_tokens)

        if getattr(args, "load_pretrained_decoder_from", None):
            logger.info(
                f"loaded pretrained decoder from: "
                f"{args.load_pretrained_decoder_from}"
            )
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_decoder_from, strict=True
            )

        return decoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim
            )

        encoder = cls.build_encoder(args, task, encoder_embed_tokens)
        if getattr(args, "encoder_freeze_module", None):
            utils.freeze_parameters(encoder, args.encoder_freeze_module)
            logging.info("freeze the encoder module: {}".format(args.encoder_freeze_module))

        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        if getattr(args, "decoder_freeze_module", None):
            utils.freeze_parameters(decoder, args.decoder_freeze_module)
            logging.info("freeze the decoder module: {}".format(args.decoder_freeze_module))
        return cls(encoder, decoder)

    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def forward(self, src_tokens, src_lengths, prev_output_tokens,gloss_src_tokens, gloss_src_lengths, **kwargs):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        sign_encoder_out, gloss_encoder_out = self.encoder(src_tokens, src_lengths,gloss_src_tokens, gloss_src_lengths)

        if "fixed_encoder_out" in sign_encoder_out.keys():
            sign_encoder_out["sign_distribution"] = self.decoder.output_layer(sign_encoder_out["fixed_encoder_out"][0].transpose(0,1))
            gloss_encoder_out["gloss_distribution"] = self.decoder.output_layer(gloss_encoder_out["fixed_encoder_out"][0].transpose(0,1))

        sign_decoder_output = self.decoder(prev_output_tokens=prev_output_tokens, encoder_out=sign_encoder_out)
        # gloss_encoder_out["encoder_out"] = [gloss_encoder_out["memory_info"]["memory_input"]]
        # gloss_encoder_out["encoder_padding_mask"] = [gloss_encoder_out["memory_info"]["memory_padding_mask"]]
        gloss_decoder_output = self.decoder(prev_output_tokens=prev_output_tokens, encoder_out=gloss_encoder_out)

        return sign_encoder_out, gloss_encoder_out, sign_decoder_output,gloss_decoder_output

class S2TvqEncoder(FairseqEncoder):
    """Speech-to-text Transformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args, task=None, embed_tokens=None):
        super().__init__(None)
        self.task = task
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.asr_encoder = S2TTransformerEncoder(args, task)

        self.memory_num = args.memory_num
        if self.memory_num:


            self.vq_memory = VQMemory(
                args.memory_num,
                args.encoder_attention_heads,
                args.encoder_embed_dim,
                args.vq_vars,
                args.vq_temp,
                args.vq_groups,
                args.encoder_embed_dim,
                weight_proj_depth=args.weight_proj_depth,
                weight_proj_factor=args.weight_proj_factor
            )

        #融合模块
        self.fuse_layer_num = args.fuse_layer_num
        self.fuse_mode = args.fuse_mode
        if self.fuse_layer_num:
            setattr(args, "use_s2_attn_norm", True)
            setattr(args, "encoder_collaboration_mode", "parallel")
            if self.fuse_mode == "gloss_sign" or self.fuse_mode == "sign_vq":
                self.fuse_layers = nn.ModuleList(
                    [TransformerS2EncoderLayer(args) for _ in range(self.fuse_layer_num)])
            elif "concat" in self.fuse_mode :
                self.fuse_layers = nn.ModuleList(
                    [TransformerIMMTEncoderLayer(args) for _ in range(self.fuse_layer_num)])
            self.sign_norm = LayerNorm(args.encoder_embed_dim)
            self.textual_norm = LayerNorm(args.encoder_embed_dim)

        self.encoder_collaboration_mode = args.encoder_collaboration_mode
        setattr(args, "use_s2_attn_norm", False)
        asr_encoder_layers = args.encoder_layers
        setattr(args, "encoder_layers", args.mt_encoder_layers)
        attn_type = args.encoder_attention_type
        setattr(args, "encoder_attention_type", "selfattn")
        if args.inter_mixup:
            self.mt_encoder = TransformerMixUpEncoder(args, task.source_dictionary, embed_tokens)
        else:
            self.mt_encoder = TransformerEncoder(args, task.source_dictionary, embed_tokens)
        setattr(args, "encoder_attention_type", attn_type)
        setattr(args, "encoder_layers", asr_encoder_layers)

    def fuse(self, sign_encoder_out,gloss_encoder_out,mode:str):
        if "concat" in mode:
            # <fix>, pseudo gloss features, then do the same operation as src_tokens
            if mode == "concat_gloss":
            # T' x B x C
                gloss_f = gloss_encoder_out["encoder_out"][0]
                gloss_padding_mask = gloss_encoder_out["encoder_padding_mask"][0]

            elif mode == "concat_vq":
                gloss_f = sign_encoder_out["fixed_encoder_out"][0]
                gloss_padding_mask = sign_encoder_out["fixed_encoder_padding_mask"][0]
            # <fix>, concat visual features and text features, T' x B x C
            x_sign = sign_encoder_out["encoder_out"][0]
            x = torch.cat([x_sign, gloss_f], dim=0)
            encoder_padding_mask = sign_encoder_out["encoder_padding_mask"][0]

            # <fix>, modify padding mask
            src_len, bsz, _ = x_sign.size()
            tgt_len, bsz, _ = x.size()
            expand_encoder_padding_mask = encoder_padding_mask.unsqueeze(1).expand(bsz, tgt_len, src_len)
            # gloss_padding_mask = torch.zeros(gloss_f.shape[1], gloss_f.shape[0]).type_as(
            #                  encoder_padding_mask)  # batch, img_len

            new_encoder_padding_mask = torch.cat([encoder_padding_mask, gloss_padding_mask], dim=1)  # batch, src_len+img_len


            for id, layer in enumerate(self.fuse_layers):  # <fix>
                # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
                x= layer(id, x, x_sign, expand_encoder_padding_mask)  # <fix>

            # sign_x = self.sign_norm(sign_x)
            sign_encoder_out["encoder_out"] = [x]
            sign_encoder_out["encoder_padding_mask"] = [new_encoder_padding_mask]
            return sign_encoder_out, gloss_encoder_out
        elif mode == "sign_vq":
            Q = sign_encoder_out["encoder_out"][0]
            Q_padding_mask = sign_encoder_out["encoder_padding_mask"][0]
            KV = sign_encoder_out["fixed_encoder_out"][0]
            KV_padding_mask = sign_encoder_out["fixed_encoder_padding_mask"][0]
        elif mode == "gloss_sign":
            Q = gloss_encoder_out["encoder_out"][0]
            Q_padding_mask = gloss_encoder_out["encoder_padding_mask"][0]
            KV = sign_encoder_out["encoder_out"][0]
            KV_padding_mask = sign_encoder_out["encoder_padding_mask"][0]


        for layer in self.fuse_layers:
            Q = layer(
                Q,
                encoder_padding_mask=Q_padding_mask,
                s2=KV,
                s2_encoder_padding_mask=KV_padding_mask,
            )
        #sign_x = self.sign_norm(sign_x)
        sign_encoder_out["encoder_out"] = [Q]
        sign_encoder_out["encoder_padding_mask"] = [Q_padding_mask]
        return sign_encoder_out,gloss_encoder_out
    def vq(self, sign_encoder_out,gloss_encoder_out):

        sign_x = sign_encoder_out["encoder_out"][0]
        sign_encoder_padding_mask = sign_encoder_out["encoder_padding_mask"][0]
        gloss_x = gloss_encoder_out["encoder_out"][0]
        gloss_encoder_padding_mask = gloss_encoder_out["encoder_padding_mask"][0]
        sign_memory_info = {
            "memory_input": sign_x,
            "memory_padding_mask": sign_encoder_padding_mask
        }
        sign_encoder_out["memory_info"] = sign_memory_info

        gloss_memory_info = {
            "memory_input": gloss_x,
            "memory_padding_mask": gloss_encoder_padding_mask
        }
        gloss_encoder_out["memory_info"] = gloss_memory_info
        x, encoder_padding_mask, vq_info = self.vq_memory(sign_x, sign_encoder_padding_mask)
        sign_encoder_out["fixed_encoder_out"] = [x]
        sign_encoder_out["fixed_encoder_padding_mask"] = [encoder_padding_mask]
        sign_encoder_out["vq_info"] = vq_info

        x, encoder_padding_mask, vq_info = self.vq_memory(gloss_x, gloss_encoder_padding_mask)
        gloss_encoder_out["fixed_encoder_out"] = [x]
        gloss_encoder_out["fixed_encoder_padding_mask"] = [encoder_padding_mask]
        gloss_encoder_out["vq_info"] = vq_info
        return sign_encoder_out,gloss_encoder_out

    def forward(self, src_tokens, src_lengths,gloss_src_tokens, gloss_src_lengths,**kwargs):
        sign_encoder_out = self.asr_encoder(src_tokens, src_lengths)
        gloss_encoder_out = self.mt_encoder(gloss_src_tokens, gloss_src_lengths, sign_encoder_out)
        if self.memory_num:
            sign_encoder_out, gloss_encoder_out = self.vq(sign_encoder_out,gloss_encoder_out)
        if self.fuse_layer_num:
            sign_encoder_out, gloss_encoder_out = self.fuse(sign_encoder_out,gloss_encoder_out,mode=self.fuse_mode)

        return sign_encoder_out,gloss_encoder_out


    def reorder_encoder_out(self, encoder_out, new_order):
        """
         Reorder encoder output according to *new_order*.

         Args:
             encoder_out: output from the ``forward()`` method
             new_order (LongTensor): desired order

         Returns:
             *encoder_out* rearranged according to *new_order*
         """
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]
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
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }

class TransformerDecoderScriptable(TransformerDecoder):
    def extract_features(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        # call scriptable method from parent class
        x, extra = self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )
        return x, extra

    def get_normalized_probs_scriptable(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, "adaptive_softmax") and self.adaptive_softmax is not None:
            if sample is not None:
                assert "target" in sample
                target = sample["target"]
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out

        logits = net_output[0]
        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)



@register_model_architecture(model_name="sign2text_vq", arch_name="sign2text_vq")
def base_architecture(args):
    # Convolutional subsampler
    args.subsampling_type = getattr(args, "subsampling_type", "conv1d")
    args.subsampling_layers = getattr(args, "subsampling_layers", 2)
    args.subsampling_filter = getattr(args, "subsampling_filter", 512)
    args.subsampling_kernel = getattr(args, "subsampling_kernel", 5)
    args.subsampling_stride = getattr(args, "subsampling_stride", 2)
    args.subsampling_norm = getattr(args, "subsampling_norm", "none")
    args.subsampling_activation = getattr(args, "subsampling_activation", "glu")

    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
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
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
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
    args.share_ctc_and_embed = getattr(args, "share_ctc_and_embed", False)

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
    args.normalize_embed = getattr(args, 'normalize_embed', False)
    args.history_dropout = getattr(args, 'history_dropout', 0.0)
    args.history_window_size = getattr(args, 'history_window_size', -1)

    # Relative position encoding
    args.max_encoder_relative_length = getattr(args, 'max_encoder_relative_length', -1)
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

    # VQ Memory
    args.memory_num = getattr(args, "memory_num", 0)
    args.vq_vars = getattr(args, "vq_vars", 0)
    args.vq_groups = getattr(args, "vq_groups", 16)
    args.vq_temp = getattr(args, "vq_temp", (2, 0.5, 0.999995))
    args.weight_proj_depth = getattr(args, "weight_proj_depth", 2)
    args.weight_proj_factor = getattr(args, "weight_proj_factor", 2)
    # mixup
    args.inter_mixup = getattr(args, "inter_mixup", False)
    args.inter_mixup_layer = getattr(args, "inter_mixup_layer", "-1")
    args.inter_gloss_mixup_layer = getattr(args, "inter_gloss_mixup_layer", "0")
    args.inter_mixup_decoder_layer = getattr(args, "inter_mixup_decoder_layer", "0")
    args.inter_mixup_beta = getattr(args, "inter_mixup_beta", 0.5)
    args.inter_mixup_prob = getattr(args, "inter_mixup_prob", 1)
    args.inter_mixup_ratio = getattr(args, "inter_mixup_ratio", 0.3)
    args.inter_mixup_keep_org = getattr(args, "inter_mixup_keep_org", False)
    args.inter_mixup_decoder_emb = getattr(args, "inter_mixup_decoder_emb", False)

    #Fuse
    args.fuse_layer_num = getattr(args, "fuse_layer_num", 0)
    args.fuse_mode = getattr(args, "fuse_mode", "self")

    # dual
    args.encoder_collaboration_mode = getattr(args, "encoder_collaboration_mode", "none")
    args.decoder_collaboration_mode = getattr(args, "decoder_collaboration_mode", "none")

    args.encoder_league_s1_ratio = getattr(args, "encoder_league_s1_ratio", 0.5)
    args.encoder_league_s2_ratio = getattr(args, "encoder_league_s2_ratio", 0.5)
    args.encoder_league_drop_net = getattr(args, "encoder_league_drop_net", False)
    args.encoder_league_drop_net_prob = getattr(args, "encoder_league_drop_net_prob", 0.0)
    args.encoder_league_drop_net_mix = getattr(args, "encoder_league_drop_net_mix", False)

    args.decoder_league_s1_ratio = getattr(args, "decoder_league_s1_ratio", 0.5)
    args.decoder_league_s2_ratio = getattr(args, "decoder_league_s2_ratio", 0.5)
    args.decoder_league_drop_net = getattr(args, "decoder_league_drop_net", False)
    args.decoder_league_drop_net_prob = getattr(args, "decoder_league_drop_net_prob", 0.0)
    args.decoder_league_drop_net_mix = getattr(args, "decoder_league_drop_net_mix", False)


@register_model_architecture("sign2text_vq", "sign2text_vq_s")
def sign2text_vq_s(args):
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.subsampling_filter = getattr(args, "subsampling_filter", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    base_architecture(args)


@register_model_architecture("sign2text_vq", "sign2text_vq_s_relative")
def sign2text_vq_s_relative(args):
    args.max_encoder_relative_length = 100
    args.k_only = True
    sign2text_vq_s(args)


@register_model_architecture("sign2text_vq", "sign2text_vq_xs")
def sign2text_vq_xs(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 4)
    args.dropout = getattr(args, "dropout", 0.3)
    sign2text_vq_s(args)


@register_model_architecture("sign2text_vq", "sign2text_vq_sp")
def sign2text_vq_sp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    sign2text_vq_s(args)


@register_model_architecture("sign2text_vq", "sign2text_vq_m")
def sign2text_vq_m(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    base_architecture(args)


@register_model_architecture("sign2text_vq", "sign2text_vq_mp")
def sign2text_vq_mp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    sign2text_vq_m(args)


@register_model_architecture("sign2text_vq", "sign2text_vq_l")
def sign2text_vq_l(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.2)
    base_architecture(args)


@register_model_architecture("sign2text_vq", "sign2text_vq_lp")
def sign2text_vq_lp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    sign2text_vq_l(args)
