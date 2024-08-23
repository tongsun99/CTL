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
)
from fairseq.models.transformer import Embedding, TransformerDecoder, TransformerEncoder#, TransformerMixUpEncoder
from torch import Tensor
logger = logging.getLogger(__name__)

@register_model("sign2text_dual")
class S2TDualModel(FairseqEncoderDecoderModel):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)


    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        S2TTransformerModel.add_args(parser)
        S2TSATEModel.add_specific_args(parser)
        S2TDualModel.add_specific_args(parser)


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
            "--Q-length",
            default="0",
            type=int,
            help="",
        )
        parser.add_argument(
            "--fuse-layer-num",
            default="0",
            type=int,
            help="",
        )
        pass

    @classmethod
    def build_encoder(cls, args, task=None, embed_tokens=None):
        encoder = S2TDualEncoder(args, task, embed_tokens)

        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from, strict=False
            )
            logger.info(
                f"loaded pretrained encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
        if getattr(args, "load_pretrained_asr_encoder_from", None):
            encoder.asr_encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder.asr_encoder, checkpoint=args.load_pretrained_asr_encoder_from, strict=False
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
        encoder_out, gloss_encoder_out = self.encoder(src_tokens, src_lengths,gloss_src_tokens, gloss_src_lengths)
        decoder_output = self.decoder(prev_output_tokens=prev_output_tokens, encoder_out=encoder_out)
        gloss_decoder_output = self.decoder(prev_output_tokens=prev_output_tokens, encoder_out=gloss_encoder_out)

        sign_distribution = self.decoder.output_layer(encoder_out["encoder_out"][0].transpose(0,1))
        gloss_distribution = self.decoder.output_layer(gloss_encoder_out["encoder_out"][0].transpose(0,1))
        return sign_distribution, gloss_distribution

    def get_distribution(self, src_tokens, src_lengths, gloss_src_tokens, gloss_src_lengths,prev_output_tokens, **kwargs):
        sign_encoder_out,mt_encoder_fake_out,mt_encoder_out = self.encoder.get_kl_encoder_out(src_tokens, src_lengths,gloss_src_tokens, gloss_src_lengths)
        sign_distribution = self.decoder.output_layer(sign_encoder_out["encoder_out"][0].transpose(0,1))
        gloss_distribution = self.decoder.output_layer(mt_encoder_fake_out["encoder_out"][0].transpose(0,1))
        sign_encoder_out["sign_distribution"] = sign_distribution
        mt_encoder_fake_out["gloss_distribution"] = gloss_distribution
        st_decoder_output = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=sign_encoder_out
        )
        mt_decoder_output = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=mt_encoder_fake_out
        )
        return sign_encoder_out, mt_encoder_fake_out, mt_encoder_out, st_decoder_output, mt_decoder_output

class ShrinkTransformerEncoder(FairseqEncoder):
    """Speech-to-text Transformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args, task=None, embed_tokens=None):
        super().__init__(None)

        dim = args.encoder_embed_dim
        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(dim)
        if args.encoder_no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = 1
        #blank改动
        #self.blank_idx = task.source_dictionary.pad()
        self.task = task
        self.blank_idx = task.source_dictionary.index("<ctc_blank>")
        self.subsample = subsampling(args)
        self.encoder_embed_linear = getattr(args, "encoder_embed_linear", False)
        self.encoder_embed_norm = getattr(args, "encoder_embed_norm", False)
        if self.encoder_embed_linear:
            self.linear = nn.Linear(dim, dim)
        if self.encoder_embed_norm:
            self.embed_ln = LayerNorm(dim)

        self.attn_type = getattr(args, "encoder_attention_type", "selfattn")
        if self.attn_type == "rel_pos":
            self.embed_positions = RelPositionalEncoding(
                args.max_source_positions, args.encoder_embed_dim
            )
        elif self.attn_type in ["rel_selfattn", "rel_pos_legacy"]:
            self.embed_positions = LegacyRelPositionalEncoding(
                args.encoder_embed_dim, args.dropout, args.max_source_positions
            )
        elif self.attn_type == "rope":
            self.embed_positions = None
        else:  # Use absolute positional embedding
            self.embed_positions = PositionalEmbedding(
                args.max_source_positions, args.encoder_embed_dim, self.padding_idx
            )

        self.layers = nn.ModuleList(
            [S2TTransformerEncoderLayer(args) for _ in range(args.encoder_layers)]
        )
        self.layer_padding_mask = args.layer_padding_mask

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(dim)
        else:
            self.layer_norm = None

        if args.use_enc_dlcl:
            self.history = DynamicLinearCombination(args, is_encoder=True)
        else:
            self.history = None

        # self.use_ctc = "sate" in args.arch or getattr(args, "ctc_weight", 0) > 0
        self.use_ctc = getattr(args, "ctc_weight", 0) > 0
        if self.use_ctc:
            self.ctc_layer = args.ctc_layer
            self.inter_ctc = True if self.ctc_layer != 0 and self.ctc_layer != args.encoder_layers else False
            if self.inter_ctc:
                logger.info("Intermedia CTC loss in layer %d" % self.ctc_layer)
            self.ctc = CTC(dim,
                           dictionary_size=len(task.source_dictionary),
                           dropout=args.dropout,
                           blank_id=self.blank_idx,
                           need_layernorm=True if self.inter_ctc else False,
                           dictionary = task.source_dictionary,)

            if getattr(args, "share_ctc_and_embed", False) and \
                    task.source_dictionary == task.target_dictionary and \
                    embed_tokens is not None and dim == embed_tokens.embedding_dim:
                self.ctc.ctc_projection.weight = embed_tokens.weight

        self.interleaved_ctc_drop_prob = args.interleaved_ctc_drop_prob
        self.sae_ground_truth_ratio = getattr(args, "sae_ground_truth_ratio", 0)
        self.share_interleaved_ctc = getattr(args, "share_interleaved_ctc", False)
        self.interleaved_ctc_layers = []
        self.use_inter_ctc = False
        if args.interleaved_ctc_layers is not None:
            self.use_inter_ctc = True
            interleaved_ctc_layers = args.interleaved_ctc_layers.split(",")
            for layer_idx in interleaved_ctc_layers:
                layer_idx = int(layer_idx)
                if layer_idx <= 0:
                    layer_idx += args.encoder_layers
                self.interleaved_ctc_layers.append(layer_idx)

                logger.info("Interleaved CTC loss in layer %d" % layer_idx)

            if not (self.use_ctc and self.share_interleaved_ctc):
                if not self.share_interleaved_ctc:
                    for layer_idx in self.interleaved_ctc_layers:
                        inter_layer_norm = LayerNorm(dim)
                        inter_ctc = CTC(dim,
                                        dictionary_size=len(task.source_dictionary),
                                        dropout=args.dropout,
                                        blank_id=self.blank_idx,
                                        )
                        setattr(self, "inter_ctc%d" % layer_idx, inter_ctc)
                        setattr(self, "inter_layer_norm%d" % layer_idx, inter_layer_norm)
                else:
                    self.ctc = CTC(dim,
                                   dictionary_size=len(task.source_dictionary),
                                   dropout=args.dropout,
                                   blank_id=self.blank_idx,
                                   )
                    if getattr(args, "share_ctc_and_embed", False) and \
                            task.source_dictionary == task.target_dictionary and \
                            embed_tokens is not None and dim == embed_tokens.embedding_dim:
                        self.ctc.ctc_projection.weight = embed_tokens.weight

            strategy = {
                "embed_norm": getattr(args, "sae_embed_norm", False),
                "out_norm": getattr(args, "sae_out_norm", False),
                "ctc_compress_strategy": getattr(args, "ctc_compress_strategy", None),
                "ctc_temperature": getattr(args, "sae_ctc_temperature", 1.0),
                "distribution_cutoff": getattr(args, "sae_distribution_cutoff", None),
                "gumbel": getattr(args, "sae_gumbel", False),
                "distribution_hard": getattr(args, "sae_distribution_hard", None),
                "gt_ratio": self.sae_ground_truth_ratio,
                "drop_prob": getattr(args, "sae_drop_prob", 0),
            }

            if not self.share_interleaved_ctc:
                for layer_idx in self.interleaved_ctc_layers:
                    sae = Adapter(dim, args.sae_adapter,
                                  len(task.source_dictionary),
                                  strategy=strategy,
                                  )
                    inter_ctc = getattr(self, "inter_ctc%d" % layer_idx)

                    if args.share_sae_and_ctc and hasattr(sae, "embed_adapter"):
                        sae.embed_adapter.weight = inter_ctc.ctc_projection.weight
                    setattr(self, "sae%d" % layer_idx, sae)
            else:
                self.sae = Adapter(dim, args.sae_adapter,
                                   len(task.source_dictionary),
                                   strategy=strategy,
                                   )
                if args.share_sae_and_ctc and hasattr(self.sae, "embed_adapter"):
                    self.sae.embed_adapter.weight = self.ctc.ctc_projection.weight

        # mixup
        self.mixup = getattr(args, "inter_mixup", False)
        if self.mixup:
            str_mixup_layer = args.inter_mixup_layer
            if len(str_mixup_layer.split(",")) == 1:
                self.mixup_layer = int(str_mixup_layer)
            else:
                self.mixup_layer = [int(layer) for layer in str_mixup_layer.split(",")]
            self.mixup_prob = args.inter_mixup_prob
            self.mixup_ratio = args.inter_mixup_ratio
            self.mixup_keep_org = args.inter_mixup_keep_org
            self.mixup_decoder_emb = args.inter_mixup_decoder_emb

            beta = args.inter_mixup_beta
            from torch.distributions import Beta
            self.beta = Beta(torch.Tensor([beta]), torch.Tensor([beta]))
            logger.info("Use mixup in layer %s with beta %.2f, prob %.2f, ratio %.2f, keep original data %r." % (
                str_mixup_layer, beta, self.mixup_prob, self.mixup_ratio, self.mixup_keep_org))

        # gather cosine similarity
        self.gather_cos_sim = getattr(args, "gather_cos_sim", False)
        self.dis = 2
        self.cos_sim = dict()

        # debug the variance
        self.debug_var = False

        self.update_num = 0
        self.curr_temp = 0

        strategy = {
            "embed_norm": getattr(args, "adapter_embed_norm", False),
            "out_norm": getattr(args, "adapter_out_norm", False),
            "ctc_compress_strategy": getattr(args, "ctc_compress_strategy", None),
            "distribution_cutoff": getattr(args, "adapter_distribution_cutoff", None),
            "drop_prob": getattr(args, "adapter_drop_prob", 0),
        }

        self.shrink_adapter = Shrink(args.encoder_embed_dim,
                               args.adapter,
                               len(task.source_dictionary),
                               blank_id=self.blank_idx,
                               strategy=strategy)

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.update_num = num_updates

    def set_ctc_infer(self, ctc_infer, post_process, src_dict=None, tgt_dict=None, path=None):
        if hasattr(self, "ctc"):
            assert src_dict is not None
            self.ctc.set_infer(ctc_infer, post_process, src_dict,
                               path=path + ".ctc" if path is not None else None)

    def ctc_valid(self, lprobs, targets, input_lengths,
                  dictionary, lang="source"):
        if hasattr(self, "ctc"):
            return self.ctc.valid(lprobs, targets, input_lengths,
                                  dictionary)

        logger.error("No ctc module in textual encoder")

    def set_debug_var(self, debug_var_flag):
        self.debug_var = debug_var_flag

    @staticmethod
    def pooling_ratio():
        return 4

    def add_to_dict(self, x, dis, idx):
        sim = 0
        seq_len = x.size(0)
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        for i in range(dis, seq_len - dis):
            a = x[i, :, :]
            for j in range(-dis, dis + 1):
                if j == 0:
                    continue
                b = x[i + j, :, :]
                sim_j = cos(a, b).mean()
                sim += sim_j
        sim = sim / 2 / dis / (seq_len - 2 * dis)

        if idx not in self.cos_sim:
            self.cos_sim[idx] = []
        self.cos_sim[idx].append(float(sim))

    def apply_mixup(self, x, encoder_padding_mask):
        batch = x.size(1)
        org_indices = np.arange(batch)

        mixup_size = int(batch * self.mixup_ratio)
        mixup_flag = []
        if mixup_size <= batch:
            #随机打散
            mixup_index1 = np.random.permutation(mixup_size)
            mixup_index2 = np.random.permutation(mixup_size)
        else:
            mixup_index1 = np.random.randint(0, batch, mixup_size)
            mixup_index2 = np.random.randint(0, batch, mixup_size)

        if self.mixup_keep_org:
            idx1 = np.append(org_indices, mixup_index1)
            idx2 = np.append(org_indices, mixup_index2)
            mixup_flag.extend([0] * len(org_indices))
            mixup_flag.extend([1] * len(mixup_index1))
        else:
            keep_indices = []
            for i in org_indices:
                if i not in mixup_index1 and i not in mixup_index2:
                    keep_indices.append(i)
            idx1 = np.append(keep_indices, mixup_index1)
            idx2 = np.append(keep_indices, mixup_index2)
            mixup_flag.extend([0] * len(keep_indices))
            mixup_flag.extend([1] * len(mixup_index1))

        idx1 = torch.from_numpy(idx1).to(x.device).long()
        idx2 = torch.from_numpy(idx2).to(x.device).long()

        x1 = x[:, idx1]
        x2 = x[:, idx2]

        coef = self.beta.sample([len(idx1)]).to(x.device).type_as(x).view(-1)
        mixup_coef = coef.view(1, -1, 1)
        x = mixup_coef * x1 + (1 - mixup_coef) * x2
        x = x.contiguous()

        pad1 = encoder_padding_mask[idx1]
        pad2 = encoder_padding_mask[idx2]
        encoder_padding_mask = pad1 & pad2
        input_lengths = (~encoder_padding_mask).sum(-1)
        mixup_flag = torch.Tensor(mixup_flag).to(x.device).bool()

        mixup = {
            "ratio": self.mixup_ratio,
            "keep_org": self.mixup_keep_org,
            "coef": coef,
            "index1": idx1,
            "index2": idx2,
            "mixup_flag": mixup_flag,
            "mixup_decoder_emb": self.mixup_decoder_emb,
        }
        return x, encoder_padding_mask, input_lengths, mixup

    def show_debug(self, x, text=None):
        if not self.debug_var:
            return

        if text:
            logger.info("--- Variance of %s: %f." % (text, x.var()))
        else:
            logger.info("--- Variance: %f." % (x.var()))

    def forward(self, src_tokens, src_lengths=None, **kwargs):

        layer_idx = -1
        mixup = None
        if self.mixup:
            if type(self.mixup_layer) is list:
                mixup_layer = choice(self.mixup_layer)
            else:
                mixup_layer = self.mixup_layer

        if self.history is not None:
            self.history.clean()

        # (B, T, D) -> (T, B, D)
        x = src_tokens.transpose(0, 1)
        input_lengths = src_lengths

        self.show_debug(x, "input x")
        # gather cosine similarity
        cos_sim_idx = -1
        dis = self.dis
        if self.gather_cos_sim:
            self.add_to_dict(x, dis, cos_sim_idx)

        if self.training and self.mixup and layer_idx == mixup_layer:
            new = torch.rand(1)
            if torch.rand(1) < self.mixup_prob:
                encoder_padding_mask = lengths_to_padding_mask(input_lengths)
                x, encoder_padding_mask, input_lengths, mixup = self.apply_mixup(x, encoder_padding_mask)

        # down-sampling
        x, input_lengths = self.subsample(x, input_lengths)
        self.show_debug(x, "x after subsampling")

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        if encoder_padding_mask is not None:
            x = x * (1 - encoder_padding_mask.transpose(0, 1).unsqueeze(-1).type_as(x))

        if self.encoder_embed_norm:
            x = self.embed_ln(x)
            self.show_debug(x, "x after embed norm")

        # embedding scaling
        x = self.embed_scale * x
        self.show_debug(x, "x after scale")

        # position embedding
        if self.attn_type in ["rel_pos", "rel_pos_legacy", "rel_selfattn"]:
            positions = self.embed_positions(x)

        elif self.attn_type == "rope":
            positions = None

        else:
            positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
            self.show_debug(positions, "position embedding")
            x += positions
            positions = None
        self.show_debug(x, "x after position embedding")

        if self.encoder_embed_linear:
            x = self.linear(x)
            self.show_debug(x, "x after embed linear")

        x = self.dropout_module(x)

        # add emb into history
        if self.history is not None:
            self.history.push(x)

        # gather cosine similarity
        cos_sim_idx = (cos_sim_idx + 10) // 10 * 10 - 1
        if self.gather_cos_sim:
            cos_sim_idx += 1
            self.add_to_dict(x, dis, cos_sim_idx)

        layer_idx += 1
        ctc_logit = None
        interleaved_ctc_logits = []

        if self.training and self.mixup and layer_idx == mixup_layer:
            if torch.rand(1) <= self.mixup_prob:
                x, encoder_padding_mask, input_lengths, mixup = self.apply_mixup(x, encoder_padding_mask)

        self.show_debug(x, "x before encoding")
        for layer in self.layers:
            if self.history is not None:
                x = self.history.pop()

            if self.layer_padding_mask and encoder_padding_mask is not None and not torch.all(encoder_padding_mask):
                mask_pad = encoder_padding_mask.unsqueeze(2)
                if mask_pad is not None:
                    x = x.transpose(0, 1)
                    x.masked_fill_(mask_pad, 0.0)
                    x = x.transpose(0, 1)

            # encoder layer
            x = layer(x, encoder_padding_mask, pos_emb=positions)
            layer_idx += 1
            self.show_debug(x, "x after layer %d" % layer_idx)

            if self.training and self.mixup and layer_idx == mixup_layer:
                if torch.rand(1) < self.mixup_prob:
                    x, encoder_padding_mask, input_lengths, mixup = self.apply_mixup(x, encoder_padding_mask)

            if self.use_ctc and self.inter_ctc and self.ctc_layer == layer_idx:
                ctc_logit = self.ctc(x.clone(), encoder_padding_mask, "Source Layer %d" % layer_idx)

            # interleaved CTC
            if layer_idx in self.interleaved_ctc_layers:
                if self.interleaved_ctc_drop_prob > 0:
                    p = torch.rand(1).uniform_()
                    if p < self.interleaved_ctc_drop_prob:
                        break

                if self.share_interleaved_ctc:
                    inter_ctc = self.ctc
                    sae = self.sae
                    layer_norm = self.layer_norm
                else:
                    inter_ctc = getattr(self, "inter_ctc%d" % layer_idx)
                    sae = getattr(self, "sae%d" % layer_idx)
                    layer_norm = getattr(self, "inter_layer_norm%d" % layer_idx)

                norm_x = layer_norm(x)
                logit = inter_ctc(norm_x, encoder_padding_mask, "Source Layer %d" % layer_idx)
                interleaved_ctc_logits.append(logit)

                # CTC alignment
                oracle = None
                oracle_mask = None
                force_emit = None
                if self.sae_ground_truth_ratio > 0:
                    ctc_alignment_oracle = kwargs.get("ctc_alignment_oracle", None)
                    if ctc_alignment_oracle is not None and ctc_alignment_oracle["source"] is not None:
                        oracle, best_aligns_pad = ctc_alignment_oracle["source"]
                        oracle_mask = (torch.rand(oracle.size(),
                                                  device=oracle.device) < self.sae_ground_truth_ratio).bool()
                        force_emit = best_aligns_pad.masked_fill(~oracle_mask, -1)

                if sae.adapter_type != "none":
                    x, encoder_padding_mask = sae([norm_x, logit], encoder_padding_mask, oracle, oracle_mask)
                    self.show_debug(x, "x after sae")

            # gather cosine similarity
            if self.gather_cos_sim:
                cos_sim_idx += 1
                self.add_to_dict(x, dis, cos_sim_idx)

            if self.history is not None:
                self.history.push(x)


        if self.history is not None:
            x = self.history.pop()

        self.show_debug(x, "x after encoding")
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        self.show_debug(x, "x after encoding layer norm")

        if self.training and self.mixup and layer_idx == mixup_layer:
            if torch.rand(1) < self.mixup_prob:
                x, encoder_padding_mask, input_lengths, mixup = self.apply_mixup(x, encoder_padding_mask)

        padding = None
        if self.use_ctc and ctc_logit is None:
            ctc_logit = self.ctc(x, encoder_padding_mask, "Source output", is_top=True)
            self.show_debug(x, "x after ctc")


        return {
            "encoder_out": [x],  # T x B x C
            "ctc_logit": [] if ctc_logit is None else [ctc_logit],  # T x B x C
            "interleaved_ctc_logits": interleaved_ctc_logits,  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            # "oracle": [oracle, oracle_mask, force_emit],
            "mixup": mixup,
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

class S2TDualEncoder(FairseqEncoder):
    """Speech-to-text Transformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args, task=None, embed_tokens=None):
        super().__init__(None)
        self.task = task
        self.do_weighted_shrink = True
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.asr_encoder = ShrinkTransformerEncoder(args, task)

        self.Q_length = args.Q_length
        #cross_attention
        if self.Q_length:
            self.Q = nn.Parameter(torch.zeros(args.Q_length,args.encoder_embed_dim))
            nn.init.normal_(self.Q, mean=0, std=args.encoder_embed_dim ** -0.5)
            setattr(args, "use_s2_attn_norm", True)
            setattr(args, "encoder_collaboration_mode", "parallel")
            self.fuse_layer_num = args.fuse_layer_num
            self.fuse_layers = nn.ModuleList(
                [TransformerS2EncoderLayer(args) for _ in range(self.fuse_layer_num)])
            self.sign_norm = LayerNorm(args.encoder_embed_dim)

            self.gloss_fuse_layers = nn.ModuleList(
                [TransformerS2EncoderLayer(args) for _ in range(self.fuse_layer_num)])
            self.gloss_norm = LayerNorm(args.encoder_embed_dim)



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

    def forward(self, src_tokens, src_lengths,gloss_src_tokens, gloss_src_lengths,**kwargs):
        sign_encoder_out = self.asr_encoder(src_tokens, src_lengths)
        gloss_encoder_out = self.mt_encoder(gloss_src_tokens, gloss_src_lengths, sign_encoder_out)
        if self.Q_length:
            sign_x = sign_encoder_out["encoder_out"][0]
            sign_encoder_padding_mask = sign_encoder_out["encoder_padding_mask"][0]
            gloss_x = gloss_encoder_out["encoder_out"][0]
            gloss_encoder_padding_mask = gloss_encoder_out["encoder_padding_mask"][0]
            Q_padding_mask = torch.zeros(sign_x.size(1),self.Q_length,device=sign_x.device)!=0
            Q = self.Q.repeat(sign_x.size(1),1,1).transpose(0,1)
            shrink_x = self.fuse_layers[0](
                Q,
                encoder_padding_mask=Q_padding_mask,
                s2=sign_x,
                s2_encoder_padding_mask=sign_encoder_padding_mask,
            )
            for i in range(1, self.fuse_layer_num):
                shrink_x = self.fuse_layers[i](
                    shrink_x,
                    encoder_padding_mask=Q_padding_mask,
                )
            shrink_gloss = self.gloss_fuse_layers[0](
                Q,
                encoder_padding_mask=Q_padding_mask,
                s2=gloss_x,
                s2_encoder_padding_mask=gloss_encoder_padding_mask,
            )
            for i in range(1, self.fuse_layer_num):
                shrink_gloss = self.gloss_fuse_layers[i](
                    shrink_gloss,
                    encoder_padding_mask=Q_padding_mask,
                )
            # sign_shrink = self.sign_norm(sign_x)
            # gloss_shrink = self.sign_norm(sign_x)
            # sign_encoder_out["shrink_encoder_out"] = [shrink_x]
            # sign_encoder_out["shrink_encoder_padding_mask"] = [Q_padding_mask]
            # gloss_encoder_out["shrink_encoder_out"] = [shrink_gloss]
            # gloss_encoder_out["shrink_encoder_padding_mask"] = [Q_padding_mask]
            sign_encoder_out["encoder_out"] = [shrink_x]
            sign_encoder_out["encoder_padding_mask"] = [Q_padding_mask]
            gloss_encoder_out["encoder_out"] = [shrink_gloss]
            gloss_encoder_out["encoder_padding_mask"] = [Q_padding_mask]
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



@register_model_architecture(model_name="sign2text_dual", arch_name="sign2text_dual")
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

    # mixup
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

    args.pds_dropout = getattr(args, "pds_dropout", args.dropout)
    args.pds_fusion = getattr(args, "pds_fusion", False)
    args.pds_fusion_method = getattr(args, "pds_fusion_method", "all_conv")
    #Fuse
    args.fuse_layer_num = getattr(args, "fuse_layer_num", 0)
    args.Q_length = getattr(args, "Q_length", 0)
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


@register_model_architecture("sign2text_dual", "sign2text_dual_s")
def sign2text_dual_s(args):
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


@register_model_architecture("sign2text_dual", "sign2text_dual_s_relative")
def sign2text_dual_s_relative(args):
    args.max_encoder_relative_length = 100
    args.k_only = True
    sign2text_dual_s(args)


@register_model_architecture("sign2text_dual", "sign2text_dual_xs")
def sign2text_dual_xs(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 4)
    args.dropout = getattr(args, "dropout", 0.3)
    sign2text_dual_s(args)


@register_model_architecture("sign2text_dual", "sign2text_dual_sp")
def sign2text_dual_sp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    sign2text_dual_s(args)


@register_model_architecture("sign2text_dual", "sign2text_dual_m")
def sign2text_dual_m(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    base_architecture(args)


@register_model_architecture("sign2text_dual", "sign2text_dual_mp")
def sign2text_dual_mp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    sign2text_dual_m(args)


@register_model_architecture("sign2text_dual", "sign2text_dual_l")
def sign2text_dual_l(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.2)
    base_architecture(args)


@register_model_architecture("sign2text_dual", "sign2text_dual_lp")
def sign2text_dual_lp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    sign2text_dual_l(args)
