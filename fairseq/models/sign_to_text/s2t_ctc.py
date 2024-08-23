import logging
from typing import Dict, Optional

import torch

from fairseq import checkpoint_utils, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)

from fairseq.models.sign_to_text import (
    S2TTransformerModel,
    S2TTransformerEncoder,
    S2TSATEModel,
    S2TSATEEncoder,
)

from torch import Tensor

logger = logging.getLogger(__name__)


@register_model("sign2text_ctc")
class S2TCTCModel(FairseqEncoderModel):

    def __init__(self, encoder):
        super().__init__(encoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        S2TTransformerModel.add_args(parser)

        # encoder
        parser.add_argument(
            "--encoder-type",
            default="transformer",
            type=str,
            help="encoder type",
        )
        pass

    @classmethod
    def build_encoder(cls, args, task=None):
        encoder = S2TCTCEncoder(args, task)
        if getattr(args, "load_pretrained_encoder_from", None):
            logger.info(
                f"loaded pretrained encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder.encoder, checkpoint=args.load_pretrained_encoder_from, strict=False
            )

        return encoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        encoder = cls.build_encoder(args, task)
        if getattr(args, "encoder_freeze_module", None):
            utils.freeze_parameters(encoder, args.encoder_freeze_module)
            logging.info("freeze the encoder module: {}".format(args.encoder_freeze_module))

        return cls(encoder)

    def get_normalized_probs(
            self,
            net_output,
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (T, B, D) tensor
        if isinstance(net_output, list):
            logits = net_output[0]
        else:
            logits = net_output["ctc_logit"][0]
        if log_probs:
            return utils.log_softmax(logits, dim=-1)
        else:
            return utils.softmax(logits, dim=-1)

    def forward(self, src_tokens, src_lengths, prev_output_tokens=None):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)

        return encoder_out


class S2TCTCEncoder(FairseqEncoder):
    """Speech-to-text Transformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args, task=None):
        super().__init__(None)

        setattr(args, "ctc_weight", 1.0)
        encoder_type = getattr(args, "encoder_type", "transformer")
        if encoder_type == "transformer":
            self.encoder = S2TTransformerEncoder(args, task)
        elif encoder_type == "sate":
            self.encoder = S2TSATEEncoder(args, task)
        else:
            logger.error("Unsupported architecture: %s." % encoder_type)

        return

    def set_ctc_infer(self, ctc_infer, post_process, src_dict=None, tgt_dict=None, path=None):
        self.encoder.set_ctc_infer(ctc_infer, post_process, src_dict=src_dict, tgt_dict=tgt_dict, path=path)

    def ctc_valid(self, lprobs, targets, input_lengths, dictionary, lang="source"):
        return self.encoder.ctc_valid(lprobs, targets, input_lengths, dictionary, lang)

    def forward(self, src_tokens, src_lengths, **kwargs):

        return self.encoder(src_tokens, src_lengths, **kwargs)

    def reorder_encoder_out(self, encoder_out, new_order):
        return self.encoder.reorder_encoder_out(encoder_out, new_order)

@register_model_architecture(model_name="sign2text_ctc", arch_name="sign2text_ctc")
def base_architecture(args):
    # Convolutional subsampler
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
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_type = getattr(args, "encoder_attention_type", "selfattn")
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)

    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.encoder_no_scale_embedding = getattr(args, "encoder_no_scale_embedding", False)

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
    args.inter_mixup = getattr(args, "inter_mixup", False)
    args.inter_mixup_layer = getattr(args, "inter_mixup_layer", None)
    args.inter_mixup_beta = getattr(args, "inter_mixup_beta", 0.5)
    args.inter_mixup_prob = getattr(args, "inter_mixup_prob", 1)
    args.inter_mixup_ratio = getattr(args, "inter_mixup_ratio", 0.3)
    args.inter_mixup_keep_org = getattr(args, "inter_mixup_keep_org", False)

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


@register_model_architecture("sign2text_ctc", "sign2text_ctc_s")
def sign2text_ctc_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)


@register_model_architecture("sign2text_ctc", "sign2text_ctc_s_relative")
def sign2text_ctc_s_relative(args):
    args.max_encoder_relative_length = 100
    args.k_only = True
    sign2text_ctc_s(args)


@register_model_architecture("sign2text_ctc", "sign2text_ctc_xs")
def sign2text_ctc_xs(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 4)
    args.dropout = getattr(args, "dropout", 0.3)
    sign2text_ctc_s(args)


@register_model_architecture("sign2text_ctc", "sign2text_ctc_sp")
def sign2text_ctc_sp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    sign2text_ctc_s(args)


@register_model_architecture("sign2text_ctc", "sign2text_ctc_m")
def sign2text_ctc_m(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    base_architecture(args)


@register_model_architecture("sign2text_ctc", "sign2text_ctc_mp")
def sign2text_ctc_mp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    sign2text_ctc_m(args)


@register_model_architecture("sign2text_ctc", "sign2text_ctc_l")
def sign2text_ctc_l(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.2)
    base_architecture(args)


@register_model_architecture("sign2text_ctc", "sign2text_ctc_lp")
def sign2text_ctc_lp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    sign2text_ctc_l(args)
