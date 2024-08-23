import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from random import choice

import torch
import torch.nn as nn

from fairseq import checkpoint_utils, utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import Embedding, TransformerDecoder, TransformerEncoder
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
    S2TTransformerEncoderLayer,
    DynamicLinearCombination,
    #LegacyRelPositionalEncoding,
)

from torch import Tensor

logger = logging.getLogger(__name__)


@register_model("g2t")
class G2TTransformerModel(FairseqEncoderDecoderModel):
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
        # Transformer
        parser.add_argument(
            "--activation-fn",
            type=str,
            default="relu",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-type",
            type=str,
            default="selfattn",
            choices=[
                "local",
                "selfattn",
                "reduced",
                "rel_selfattn",
                "relative",
                "rel_pos_legacy",
                "rel_pos",
                "rope",
                "abs",
                "transfer",
                "reduced_rel_pos",
            ],
            help="transformer encoder self-attention layer type"
        )
        # transfer
        parser.add_argument(
            "--relative-pos-enc",
            action="store_true",
            help="use relative position encoding for attention",
        )
        parser.add_argument(
            "--linear-att",
            action="store_true",
            help="use linear attention",
        )

        # reduced attention
        parser.add_argument(
            "--attention-reduced-method",
            type=str,
            default="conv",
            help="reduction method for attention",
        )
        parser.add_argument(
            "--attention-reduced-q",
            action="store_true",
            help="use reduction for query or not"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-type",
            type=str,
            default="selfattn",
            choices=[
                "selfattn",
                "rel_selfattn",
                "relative",
                "local",
            ],
            help="transformer decoder self-attention layer type"
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
            help="apply layernorm before each decoder block",
        )
        parser.add_argument(
            "--share-decoder-input-output-embed",
            action="store_true",
            help="share decoder input and output embeddings",
        )
        parser.add_argument('--share-all-embeddings',
                            action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings",
        )
        parser.add_argument(
            "--encoder-no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings in encoder",
        )
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--max-encoder-relative-length', type=int, default=-1,
                            help='the max relative length')
        parser.add_argument('--max-decoder-relative-length', type=int, default=-1,
                            help='the max relative length')
        parser.add_argument('--k-only', default=False, action='store_true',
                            help='select the relative mode to map relative position information')
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--load-pretrained-decoder-from",
            type=str,
            metavar="STR",
            help="model to take decoder weights from (for initialization)",
        )
        parser.add_argument(
            "--encoder-freeze-module",
            type=str,
            metavar="STR",
            help="freeze the module of the encoder",
        )
        parser.add_argument(
            "--decoder-freeze-module",
            type=str,
            metavar="STR",
            help="freeze the module of the decoder",
        )
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        # DLCL
        parser.add_argument(
            "--use-enc-dlcl",
            default=False,
            action='store_true',
            help="use dlcl encoder",
        )
        parser.add_argument(
            "--use-dec-dlcl",
            default=False,
            action='store_true',
            help="use dlcl encoder",
        )
        parser.add_argument('--init-value', type=str, default='avg', choices=['avg', 'one'],
                            help='how to init the learned weight matrix')
        parser.add_argument('--weight-type', type=str, default='scalar',
                            help='type of learned weight [scalar, scalar_n(n>1), vector]')
        parser.add_argument('--encoder-learnable', type=eval, default='True',
                            help='enable to learn weights for encoder')
        parser.add_argument('--decoder-learnable', type=eval, default='True',
                            help='enable to learn weights for decoder')
        parser.add_argument('--normalize-learned-weight', type=eval, default='False',
                            help='normalize learned weight by softmax')
        parser.add_argument('--normalize-embedding', type=eval, default='False',
                            help='normalize the input of embedding')
        parser.add_argument('--history-dropout', type=float, default=0.0, metavar='D',
                            help='dropout for history output')
        parser.add_argument('--history-window-size', type=int, default='-1',
                            help='how many past layers are considered. -1 means all')
        # CTC
        parser.add_argument(
            "--ctc-layer",
            default=0,
            type=int,
            help="the position of the ctc loss",
        )
        parser.add_argument(
            "--share-ctc-and-embed",
            action="store_true",
            help="share the weight of ctc and embedding",
        )

        # local modeling
        parser.add_argument(
            '--hard-mask-window',
            type=float,
            metavar="D",
            default=0,
            help='window size of local mask'
        )
        parser.add_argument(
            '--gauss-mask-sigma',
            type=float,
            metavar="D",
            default=0,
            help='standard deviation of the gauss mask'
        )
        parser.add_argument(
            '--init-mask-weight',
            type=float,
            metavar="D",
            default=0.5,
            help='initialized weight for local mask'
        )
        parser.add_argument(
            "--layer-padding-mask",
            default=False,
            type=bool,
            help="mask the padding to 0 before each layer"
        )

        # Conformer setting
        parser.add_argument(
            "--encoder-activation-fn",
            type=str,
            default="relu",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--macaron-style",
            default=False,
            type=bool,
            help="Whether to use macaron style for positionwise layer",
        )
        # Attention
        parser.add_argument(
            "--zero-triu",
            default=False,
            type=bool,
            help="If true, zero the upper triangular part of attention matrix.",
        )

        parser.add_argument(
            "--encoder-embed-linear",
            action="store_true",
            help="use linear transform after down-sampling",
        )
        parser.add_argument(
            "--encoder-embed-norm",
            action="store_true",
            help="use layer norm after down-sampling",
        )


        # Mixup
        parser.add_argument(
            "--inter-mixup",
            action="store_true",
            help="use mixup or not",
        )
        parser.add_argument(
            "--inter-mixup-layer",
            default="-1",
            type=str,
            help="the layers to apply mixup",
        )
        parser.add_argument(
            "--inter-mixup-decoder-layer",
            default="0",
            type=str,
            help="the layers to apply mixup in the decoder",
        )
        parser.add_argument(
            "--inter-mixup-beta",
            default=0.5,
            type=float,
            help="the coefficient beta of mixup",
        )
        parser.add_argument(
            "--inter-mixup-prob",
            default=1,
            type=float,
            help="the probability of mixup",
        )
        parser.add_argument(
            "--inter-mixup-ratio",
            default=1,
            type=float,
            help="the ratio of mixup",
        )
        parser.add_argument(
            "--inter-mixup-keep-org",
            action="store_true",
            help="keep original batch",
        )
        parser.add_argument(
            "--inter-mixup-decoder-emb",
            action="store_true",
            help="mix the embedding in the decoder",
        )
        #mixup decoder可能是单独
        pass

    @classmethod
    def build_encoder(cls, args, task=None, embed_tokens=None):
        if args.inter_mixup:
            encoder = TransformerMixupEncoder(args, task, embed_tokens)
        else:
            encoder = TransformerEncoder(args, task, embed_tokens)
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from, strict=False
            )
            logger.info(
                f"loaded pretrained encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )

        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        if args.inter_mixup:
            decoder = TransformerDecoderMixupScriptable(args, task.target_dictionary, embed_tokens)
        else:
            decoder = TransformerDecoderScriptable(args, task.target_dictionary, embed_tokens)

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

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        return decoder_out
class TransformerMixupEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens, return_fc=False):
        self.args = args
        super().__init__(
            args,
            dictionary,
            embed_tokens,
            return_fc=return_fc,
        )
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

    def forward_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        layer_idx = 0
        mixup = None
        if self.mixup:
            if type(self.mixup_layer) is list:
                mixup_layer = choice(self.mixup_layer)
            else:
                mixup_layer = self.mixup_layer
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = (
            torch.tensor(src_tokens.device.type == "xla") or encoder_padding_mask.any()
        )
        # Torchscript doesn't handle bool Tensor correctly, so we need to work around.
        if torch.jit.is_scripting():
            has_pads = torch.tensor(1) if has_pads else torch.tensor(0)

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        x = x * (
            1 - encoder_padding_mask.unsqueeze(-1).type_as(x) * has_pads.type_as(x)
        )

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []
        fc_results = []

        if return_all_hiddens:
            encoder_states.append(x)
        if self.training and self.mixup and layer_idx == mixup_layer:
            if torch.rand(1) <= self.mixup_prob:
                x, encoder_padding_mask, input_lengths, mixup = self.apply_mixup(x, encoder_padding_mask)

        # encoder layers
        for layer in self.layers:
            lr = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None
            )

            if isinstance(lr, tuple) and len(lr) == 2:
                x, fc_result = lr
            else:
                x = lr
                fc_result = None

            if return_all_hiddens and not torch.jit.is_scripting():
                assert encoder_states is not None
                encoder_states.append(x)
                fc_results.append(fc_result)
            #mixup
            layer_idx += 1
            if self.training and self.mixup and layer_idx == mixup_layer:
                if torch.rand(1) < self.mixup_prob:
                    x, encoder_padding_mask, input_lengths, mixup = self.apply_mixup(x, encoder_padding_mask)


        if self.layer_norm is not None:
            x = self.layer_norm(x)


        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = (
            src_tokens.ne(self.padding_idx)
            .sum(dim=1, dtype=torch.int32)
            .reshape(-1, 1)
            .contiguous()
        )
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "fc_results": fc_results,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
            "mixup": mixup,
        }

    def apply_mixup(self, x, encoder_padding_mask):
        batch = x.size(1)
        org_indices = np.arange(batch)

        mixup_size = int(batch * self.mixup_ratio)
        mixup_flag = []
        if mixup_size <= batch:
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




class TransformerDecoderMixupScriptable(TransformerDecoder):
    def __init__(
            self,
            args,
            dictionary,
            embed_tokens,
            no_encoder_attn=False,
            output_projection=None,
    ):
        self.args = args
        super().__init__(
            args,
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
        )
        self.mixup = getattr(args, "inter_mixup", False)
        if self.mixup:
            self.mixup_decoder_emb = args.inter_mixup_decoder_emb
            str_mixup_layer = getattr(args, "inter_mixup_decoder_layer", "0")
            if len(str_mixup_layer.split(",")) == 1:
                self.mixup_layer = int(str_mixup_layer)
            else:
                self.mixup_layer = [int(layer) for layer in str_mixup_layer.split(",")]
            logger.info("Use mixup in the decoder layer %s, mixup decoder embedding %r." % (
                    str_mixup_layer, self.mixup_decoder_emb))

    def apply_mixup(self, encoder_out, x, self_attn_padding_mask):
        mixup = encoder_out["mixup"]

        coef = mixup["coef"]
        idx1 = mixup["index1"]
        idx2 = mixup["index2"]
        flag = mixup["mixup_flag"]

        if mixup["mixup_decoder_emb"]:
            x1 = x[:, idx1]
            x2 = x[:, idx2]
            mixup_coef = coef.view(1, -1, 1)
            x = mixup_coef * x1 + (1 - mixup_coef) * x2
            x = x.contiguous()

            if self_attn_padding_mask is not None:
                pad1 = self_attn_padding_mask[idx1]
                pad2 = self_attn_padding_mask[idx2]
                self_attn_padding_mask = pad1 & pad2

        else:
            # 没有把target向量也加权平均，a顺序的向量和b顺序的向量mixup之后，其标签还是原来a顺序的target，b顺序的target，32长度的原来target变成了96长度的target。
            mix_idx1 = idx1[flag]
            mix_idx2 = idx2[flag]
            org_idx = idx1[~flag]
            x1 = x[:, mix_idx1]
            x2 = x[:, mix_idx2]

            if self_attn_padding_mask is not None:
                pad1 = self_attn_padding_mask[mix_idx1]
                pad2 = self_attn_padding_mask[mix_idx2]

            decoder_mixup_flag1 = [0] * len(org_idx)
            decoder_mixup_flag2 = [0] * len(org_idx)
            if len(org_idx) != 0:
                org_x = x[:, org_idx]
                x = torch.cat([org_x, x1, x2], dim=1)
                if self_attn_padding_mask is not None:
                    org_pad = self_attn_padding_mask[org_idx]
                    self_attn_padding_mask = torch.cat([org_pad, pad1, pad2], dim=0)
            else:
                x = torch.cat([x1, x2], dim=1)
                if self_attn_padding_mask is not None:
                    self_attn_padding_mask = torch.cat([pad1, pad2], dim=0)

            decoder_mixup_flag1.extend([1] * len(mix_idx1))
            decoder_mixup_flag1.extend([0] * len(mix_idx2))
            decoder_mixup_flag2.extend([0] * len(mix_idx1))
            decoder_mixup_flag2.extend([1] * len(mix_idx2))
            mixup["decoder_mixup_flag1"] = torch.Tensor(decoder_mixup_flag1).to(x.device).bool()
            mixup["decoder_mixup_flag2"] = torch.Tensor(decoder_mixup_flag2).to(x.device).bool()

            encoder_rep = encoder_out["encoder_out"][0]
            mixup_encoder_rep = encoder_rep[:, flag, :]
            encoder_out["encoder_out"][0] = torch.cat([encoder_rep, mixup_encoder_rep], dim=1)

            padding = encoder_out["encoder_padding_mask"][0]
            mixup_padding = padding[flag, :]
            encoder_out["encoder_padding_mask"][0] = torch.cat([padding, mixup_padding], dim=0)

        return encoder_out, x, self_attn_padding_mask, mixup

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
    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]


        layer_idx = -1

        bak_encoder_out = encoder_out["encoder_out"][0]
        bak_encoder_padding_mask = encoder_out["encoder_padding_mask"][0]
        do_mixup = False
        mixup_layer = 0
        mixup = None
        if "mixup" in encoder_out and encoder_out["mixup"] is not None:
            do_mixup = True
            if type(self.mixup_layer) is list:
                from random import choice
                mixup_layer = choice(self.mixup_layer)
            else:
                mixup_layer = self.mixup_layer

        if do_mixup and layer_idx == mixup_layer:
            logger.warning("To DO!!!")

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # Prevent torchscript exporting issue for dynamic quant embedding
        prev_output_tokens = prev_output_tokens.contiguous()
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        layer_idx += 1
        if do_mixup and layer_idx == mixup_layer:
            encoder_out, x, self_attn_padding_mask, mixup = self.apply_mixup(encoder_out, x, self_attn_padding_mask)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

            layer_idx += 1
            if do_mixup and layer_idx == mixup_layer:
                encoder_out, x, self_attn_padding_mask, mixup = self.apply_mixup(encoder_out, x, self_attn_padding_mask)


        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        if do_mixup:
            encoder_out["encoder_out"][0] = bak_encoder_out
            encoder_out["encoder_padding_mask"][0] = bak_encoder_padding_mask

        return x, {"attn": [attn], "inner_states": inner_states, "mixup": mixup}
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


@register_model_architecture(model_name="g2t", arch_name="g2t_transformer")
def base_architecture(args):

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
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

    args.encoder_embed_linear = getattr(args, "encoder_embed_linear", False)
    args.encoder_embed_norm = getattr(args, "encoder_embed_norm", False)

    args.layer_padding_mask = getattr(args, "layer_padding_mask", False)

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

    # mixup
    args.inter_mixup = getattr(args, "inter_mixup", False)
    args.inter_mixup_layer = getattr(args, "inter_mixup_layer", "-1")
    args.inter_mixup_decoder_layer = getattr(args, "inter_mixup_decoder_layer", "0")
    args.inter_mixup_beta = getattr(args, "inter_mixup_beta", 0.5)
    args.inter_mixup_prob = getattr(args, "inter_mixup_prob", 1)
    args.inter_mixup_ratio = getattr(args, "inter_mixup_ratio", 0.3)
    args.inter_mixup_keep_org = getattr(args, "inter_mixup_keep_org", False)
    args.inter_mixup_decoder_emb = getattr(args, "inter_mixup_decoder_emb", False)


@register_model_architecture("g2t", "g2t_transformer_s")
def sign2text_transformer_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)





