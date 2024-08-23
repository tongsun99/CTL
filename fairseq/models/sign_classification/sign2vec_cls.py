# Copyright (c) ASAPP Inc.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import contextlib
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from omegaconf import MISSING, II, open_dict
from typing import Any, Optional

from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.tasks import FairseqTask
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
)
from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES
from fairseq.modules import (
    LayerNorm,
    PositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from fairseq.models.sign_to_text.s2t_transformer import S2TTransformerEncoder

logger = logging.getLogger(__name__)


@dataclass
class Sign2VecClsConfig(FairseqDataclass):
    # encoder   
    s2ttransformer_checkpoint: Optional[str] = field(
        default="/home2/tsun/code/vac_s2t/data/pami/ckpt/checkpoint_best.sh", metadata={"help": "path to load s2t transformer checkpoint"}
    )
    freeze_encoder: bool = field(
        default=True, metadata={"help": "freeze encoder"}
    )

    # for sequence classification
    pool_method: str = field(default="avg", metadata={"help": "pooling method"})
    classifier_dropout: float = field(default=0.0, metadata={"help": "dropout"})


@register_model("sign2vec_seq_cls", dataclass=Sign2VecClsConfig)
class Sign2VecSeqCls(BaseFairseqModel):
    def __init__(
        self,
        cfg: Sign2VecClsConfig,
        sign_encoder: BaseFairseqModel,
        pooler: nn.Module,
        classifier: nn.Module,
    ):
        super().__init__()
        self.cfg = cfg
        self.sign_encoder = sign_encoder
        self.pooler = pooler
        self.classifier = classifier

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: Sign2VecClsConfig, task: FairseqTask):
        """Build a new model instance."""
        # encoder
        state = checkpoint_utils.load_checkpoint_to_cpu(cfg.s2ttransformer_checkpoint) 
        trans_task_args = state["cfg"]["task"]
        trans_task = tasks.setup_task(trans_task_args)
        state["cfg"]["model"].load_pretrained_encoder_from = None
        state["cfg"]["model"].load_pretrained_decoder_from = None
        model = trans_task.build_model(state["cfg"]["model"])
        model.load_state_dict(state["model"], strict=True)
        if cfg.freeze_encoder:
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
        model.share_memory()
        sign_encoder = model.encoder
        d = state["cfg"]["model"].encoder_embed_dim

        if cfg.pool_method == "avg":
            pooler = AvgPooler()
        elif cfg.pool_method == "self_attn":
            pooler = SelfAttnPooler(d)
        elif cfg.pool_method == "transformer_avg":
            pooler = TransformerAvgPooler(state["cfg"]["model"])
        elif cfg.pool_method == "transformer_cls":
            pooler = TransformerClsPooler(state["cfg"]["model"])
        elif cfg.pool_method == "transformer_cls2":
            pooler = TransformerClsPooler2(state["cfg"]["model"])
        else:
            raise NotImplementedError(f"pooler_type={cfg.pool_method}")

        num_classes = len(task.label2id)
        classifier = nn.Sequential(
            nn.Dropout(cfg.classifier_dropout),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(cfg.classifier_dropout),
            nn.Linear(d, num_classes),
            # nn.ReLU()   # test
        )

        return cls(cfg, sign_encoder, pooler, classifier)

    def get_logits(self, net_output, normalize=False):
        logits = net_output["encoder_out"]
        if self.blank_weight != 0:
            if self.blank_mode == "add":
                logits[..., 0] += self.blank_weight
            elif self.blank_mode == "set":
                logits[..., 0] = self.blank_weight
            else:
                raise Exception(f"invalid blank mode {self.blank_mode}")

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            logits[net_output["padding_mask"].T][..., 0] = float("inf")
            logits[net_output["padding_mask"].T][..., 1:] = float("-inf")

        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)

        return logits

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = self.get_logits(net_output)

        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def forward(self, **kwargs):
        src_indices = kwargs["source"]   # (B, T, D)
        if kwargs["padding_mask"] is not None:
            src_lengths = torch.sum(~kwargs["padding_mask"], dim=-1)  # (B)   
        else:   # eval
            src_lengths = torch.tensor([src_indices.size(1)], device=src_indices.device)    # (B) (1)

        x = self.sign_encoder(src_indices, src_lengths) # (T, B, D)
        padding_mask = (
            x["encoder_padding_mask"][0].transpose(0, 1) if x["encoder_padding_mask"][0] is not None else None
        )   # T,B
        pooled = self.pooler(x["encoder_out"][0], padding_mask) # (1, D)
        pooled = self.classifier(pooled)
        x["pooled"] = pooled
        return x

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class AvgPooler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, encoder_out, padding_mask):
        """
        encoder_out: T, B, C
        padding_mask: T, B (True for padded positions)
        """
        if padding_mask is None:
            return encoder_out.mean(dim=0)
        else:
            dtype = encoder_out.dtype
            encoder_out[padding_mask, :] = 0.0
            lengths = (~padding_mask).float().sum(dim=0)
            out = encoder_out.float().sum(dim=0) / lengths.unsqueeze(-1)
            return out.to(dtype)


class SelfAttnPooler(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)

    def forward(self, encoder_out, padding_mask):
        """
        encoder_out: T, B, C
        padding_mask: T, B (True for padded positions)
        """
        dtype = encoder_out.dtype
        attn_weights = self.proj(encoder_out).squeeze(-1).float()
        if padding_mask is not None:
            attn_weights[padding_mask] = float("-inf")
        attn_weights = attn_weights.softmax(dim=0)
        out = torch.einsum("tb,tbc->bc", attn_weights.float(), encoder_out.float())
        return out.to(dtype)

class TransformerAvgPooler(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.transformer = TransformerEncoderLayer(args)
    
    def forward(self, encoder_out, padding_mask):
        """
        encoder_out: T, B, C
        padding_mask: T, B (True for padded positions)
        """
        encoder_out = self.transformer(encoder_out, padding_mask.transpose(0, 1))
        if padding_mask is None:
            return encoder_out.mean(dim=0)
        else:
            dtype = encoder_out.dtype
            encoder_out[padding_mask, :] = 0.0
            lengths = (~padding_mask).float().sum(dim=0)
            out = encoder_out.float().sum(dim=0) / lengths.unsqueeze(-1)
            return out.to(dtype)
        
        
class TransformerClsPooler(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.transformer = TransformerEncoderLayer(args)
    
    
    def forward(self, encoder_out, padding_mask):
        """
        encoder_out: T, B, C
        padding_mask: T, B (True for padded positions)
        """
        encoder_out = self.transformer(encoder_out, padding_mask.transpose(0, 1))
        dtype = encoder_out.dtype
        out = encoder_out[0]
        return out.to(dtype)

class TransformerClsPooler2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.transformer = TransformerEncoderLayer(args)
        self.cls_token = nn.Parameter(torch.zeros(1, args.encoder_embed_dim))
    
    
    def forward(self, encoder_out, padding_mask):
        """
        encoder_out: T, B, C
        padding_mask: T, B (True for padded positions)
        """
        encoder_out = torch.cat([self.cls_token.expand(1, encoder_out.shape[1], -1), encoder_out], dim=0)
        padding_mask = torch.cat([torch.zeros(1, padding_mask.shape[1], dtype=torch.bool, device=padding_mask.device), padding_mask], dim=0)
        encoder_out = self.transformer(encoder_out, padding_mask.transpose(0, 1))
        dtype = encoder_out.dtype
        out = encoder_out[0]
        return out.to(dtype)

