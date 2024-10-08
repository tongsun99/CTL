import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import groupby
from fairseq.data.sign.sign_dataset import _collate_frames
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.modules import LayerNorm

logger = logging.getLogger(__name__)


class CTCCompressStrategy:
    @staticmethod
    def avg(prob_ctc, predicted, new_lengths, dtype, device):
        new_maxlen = max(new_lengths)
        weights_matrix = torch.zeros((prob_ctc.shape[0], prob_ctc.shape[1], new_maxlen), dtype=dtype)
        for b_idx, pred in enumerate(predicted):
            processed_inputs_cnt = 0
            for t_idx, same in enumerate(pred):
                new_processed_inputs_cnt = processed_inputs_cnt + same[1]
                weights_matrix[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, t_idx] = 1.0 / same[1]
                processed_inputs_cnt = new_processed_inputs_cnt
        return weights_matrix.to(device)

    @staticmethod
    def weighted(prob_ctc, predicted, new_lengths, dtype, device):
        new_maxlen = max(new_lengths)
        weights_matrix = torch.zeros((prob_ctc.shape[0], prob_ctc.shape[1], new_maxlen), dtype=dtype, device=device)
        for b_idx, pred in enumerate(predicted):
            processed_inputs_cnt = 0
            for t_idx, same in enumerate(pred):
                new_processed_inputs_cnt = processed_inputs_cnt + same[1]
                # Get the probabilities of the prediction for the different time steps as weight
                weights = prob_ctc[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, same[0]]
                weights_matrix[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, t_idx] = \
                    weights / weights.sum()
                processed_inputs_cnt = new_processed_inputs_cnt
        return weights_matrix

    @staticmethod
    def softmax(prob_ctc, predicted, new_lengths, dtype, device):
        new_maxlen = max(new_lengths)
        weights_matrix = torch.zeros((prob_ctc.shape[0], prob_ctc.shape[1], new_maxlen), dtype=dtype, device=device)
        for b_idx, pred in enumerate(predicted):
            processed_inputs_cnt = 0
            for t_idx, same in enumerate(pred):
                new_processed_inputs_cnt = processed_inputs_cnt + same[1]
                # Get the probabilities of the prediction for the different time steps as weight
                weights = F.softmax(
                    prob_ctc[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, same[0]], dtype=torch.float32
                )
                weights_matrix[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, t_idx] = \
                    weights / weights.sum()
                processed_inputs_cnt = new_processed_inputs_cnt
        return weights_matrix


class Adapter(nn.Module):
    def __init__(self, dim, adapter_type, dictionary_size, embed_tokens=None, strategy=None):
        super().__init__()

        dim = dim
        self.adapter_type = adapter_type
        self.cal_linear = False
        self.cal_context = False
        self.shrink = False

        if self.adapter_type in ["linear", "league", "gated_league", "gated_league2", "league_shrink"]:
            self.cal_linear = True
            self.linear_adapter = nn.Sequential(
                nn.Linear(dim, 2 * dim),
                nn.ReLU(),
                nn.Linear(2 * dim, dim),
                LayerNorm(dim),
            )

        if self.adapter_type in ["context", "league", "gated_league", "gated_league2", "inter_league",
                                 "league_shrink", "inter_league_shrink", "context_shrink"]:
            self.cal_context = True
            self.embed_adapter = nn.Linear(dim, dictionary_size, bias=False)  # reverse for initialization
            nn.init.normal_(self.embed_adapter.weight, mean=0, std=dim ** -0.5)
            self.embed_norm = strategy.get("embed_norm", False)
            if self.embed_norm:
                self.embed_ln = LayerNorm(dim)
            if embed_tokens is not None:
                self.embed_adapter.weight = embed_tokens.weight

        if self.adapter_type == "gated_league":
            self.gate_linear = nn.Linear(2 * dim, dim)
        elif self.adapter_type == "gated_league2":
            self.gate_linear1 = nn.Linear(dim, dim)
            self.gate_linear2 = nn.Linear(dim, dim)

        # additional strategy
        if self.adapter_type in ["shrink", "league_shrink", "inter_league_shrink", "context_shrink"]:
            assert strategy is not None
            ctc_compress_strategy = strategy.get("ctc_compress_strategy", "avg")
            self.ctc_compress = getattr(CTCCompressStrategy, ctc_compress_strategy)
            self.shrink = True
            logger.info("CTC Compress Strategy: %s" % ctc_compress_strategy)

        if self.cal_context or self.shrink:
            self.distribution_cutoff = strategy.get("distribution_cutoff", None)
            self.distribution_temperature = strategy.get("ctc_temperature", 1.0)
            self.gumbel = strategy.get("gumbel", False)
            self.distribution_hard = strategy.get("distribution_hard", False)
            self.ground_truth_ratio = strategy.get("gt_ratio", 0)
            self.drop_prob = strategy.get("drop_prob", 0)

            if self.distribution_cutoff is not None:
                logger.info("Distribution cutoff: %d" % self.distribution_cutoff)
            if self.distribution_temperature != 1.0:
                logger.info("Temperature: %f" % self.distribution_temperature)
            if self.gumbel:
                logger.info("Gumbel softmax.")
            if self.distribution_hard:
                logger.info("Hard distribution.")
            if self.drop_prob != 0:
                logger.info("Drop probability: %f" % self.drop_prob)

        self.out_norm = strategy.get("out_norm", False)
        if self.out_norm:
            self.out_ln = LayerNorm(dim)

    def forward(self, x, padding=None, oracle=None, oracle_mask=None):
        representation, logit = x
        seq_len, bsz, dim = representation.size()

        distribution = None
        if self.cal_context or self.shrink:
            if self.training and self.gumbel:
                distribution = F.gumbel_softmax(logit, tau=self.distribution_temperature, hard=self.distribution_hard)
            else:
                distribution = F.softmax(logit / self.distribution_temperature, dim=-1)

        linear_out = None
        soft_out = None
        out = None
        if self.cal_linear:
            linear_out = self.linear_adapter(representation)
        if self.cal_context:
            vocab_size = distribution.size(-1)
            distribution_2d = distribution.contiguous().view(-1, vocab_size)

            if self.distribution_cutoff is not None:
                pass
                # cutoff = min(int(self.distribution_cutoff), vocab_size - 1)

                # threshold = distribution.sort(dim=-1, descending=True)[0][:, :, cutoff:cutoff+1]
                # distribution_2d = torch.where(
                #     distribution > threshold, distribution, torch.zeros_like(distribution)
                # )

                # threshold = distribution.sort(dim=-1, descending=True)[0][:, :, :cutoff].sum(-1, keepdim=True)
                # distribution_2d = torch.where(
                #     threshold > 0.9, distribution, torch.zeros_like(distribution)
                # )
                # distribution_2d = distribution_2d.view(-1, vocab_size)

                # distribution_2d[:, 0] = 0
                # distribution_2d = distribution_2d / distribution_2d.sum(-1, keepdim=True)

            if self.ground_truth_ratio > 0 and oracle is not None:
                oracle = oracle.unsqueeze(-1)
                oracle_one_hot = (oracle == torch.arange(vocab_size, device=oracle.device).unsqueeze(0)). \
                    to(distribution.dtype).transpose(0, 1)
                oracle_mask = oracle_mask.transpose(0, 1).unsqueeze(-1).repeat(1, 1, vocab_size)
                modify_dist = oracle_mask * oracle_one_hot + ~oracle_mask * distribution
                soft_out = torch.mm(modify_dist.view(-1, vocab_size), self.embed_adapter.weight).view(seq_len, bsz, -1)
            else:
                soft_out = torch.mm(distribution_2d, self.embed_adapter.weight).view(seq_len, bsz, -1)

            if self.embed_norm:
                soft_out = self.embed_ln(soft_out)

        if self.adapter_type == "linear":
            out = linear_out

        elif self.adapter_type == "context":
            out = soft_out

        elif self.adapter_type in ["league"]:
            if self.training and self.drop_prob > 0 and torch.rand(1).uniform_() < self.drop_prob:
                if torch.rand(1).uniform_() < 0.5:
                    out = linear_out
                else:
                    out = soft_out
            else:
                out = linear_out + soft_out

        elif self.adapter_type == "gated_league":
            coef = (self.gate_linear(torch.cat([linear_out, soft_out], dim=-1))).sigmoid()
            out = coef * linear_out + (1 - coef) * soft_out

        elif self.adapter_type in ["inter_league", "inter_league_shrink"]:
            out = representation + soft_out

        elif self.adapter_type == "none":
            out = representation

        elif self.adapter_type in ["shrink", "league_shrink", "inter_league_shrink", "context_shrink"]:
            if self.adapter_type in ["league_shrink", "inter_league_shrink"]:
                representation = linear_out + soft_out
            elif self.adapter_type in ["context_shrink"]:
                representation = soft_out

            lengths = (~padding).long().sum(-1)
            with torch.no_grad():
                batch_predicted = []
                prob_ctc = distribution.transpose(0, 1)  # T x B x D -> B x T x D
                for b in range(prob_ctc.shape[0]):
                    predicted = prob_ctc[b][: lengths[b]].argmax(-1).tolist()
                    batch_predicted.append([(p[0], len(list(p[1]))) for p in groupby(predicted)])

                new_lengths = [len(p) for p in batch_predicted]
                weights_matrix = self.ctc_compress(prob_ctc, batch_predicted, new_lengths,
                                                   prob_ctc.dtype, prob_ctc.device)

            # x is T x B x C -> B x C x T; weights_matrix is B x T x T'
            representation = representation.permute(1, 2, 0)
            compressed_output = representation.bmm(weights_matrix).type_as(representation)  # B x C x T'
            out = compressed_output.permute(2, 0, 1)

            out_lengths = lengths.new(new_lengths)
            padding = lengths_to_padding_mask(out_lengths)

        else:
            out = None
            logging.error("Unsupported adapter type: {}.".format(self.adapter_type))

        if self.out_norm:
            out = self.out_ln(out)

        return out, padding

class Shrink(nn.Module):
    def __init__(self, dim, adapter_type, dictionary_size, blank_id,embed_tokens=None, strategy=None):
        super().__init__()

        dim = dim
        self.adapter_type = adapter_type
        self.pad = 1
        self.distribution_temperature = 1.0
        self.blank_id = blank_id

        ctc_compress_strategy = strategy.get("ctc_compress_strategy", "avg")
        self.ctc_compress = getattr(CTCCompressStrategy, ctc_compress_strategy)
        logger.info("CTC Compress Strategy: %s" % ctc_compress_strategy)

        self.out_norm = strategy.get("out_norm", False)
        if self.out_norm:
            self.out_ln = LayerNorm(dim)

    def forward(self, x, padding=None, oracle=None, oracle_mask=None):
        representation, logit = x
        _,_, C = logit.size()
        _, _, dim = representation.size()
        logit = logit.transpose(0, 1)
        representation = representation.transpose(0,1)
        lengths = (~padding).long().sum(-1)

        if padding is not None:
            # replace paddings' logits s.t. they predict pads
            one_hot = F.one_hot(torch.LongTensor([self.blank_id]), C).type_as(logit)
            logit[padding] = one_hot

        feature_shrink = []
        gloss_fake = []
        for repre_single, b in zip(representation,range(logit.shape[0])):
            predicted = logit[b].argmax(-1)
            pre_mask = (predicted != self.blank_id)
            feature_single = torch.masked_select(repre_single, pre_mask.unsqueeze(1)).reshape(-1,dim)
            feature_shrink.append(feature_single)

        new_feature = _collate_frames(feature_shrink)

        shrink_lengths = torch.tensor([x.size(0) for x in feature_shrink], dtype=torch.long)

        with torch.no_grad():
            batch_predicted = []  # T x B x D -> B x T x D
            for b in range(logit.shape[0]):
                predicted = logit[b][: lengths[b]].argmax(-1)
                predicted_shrink = predicted[predicted != self.blank_id].tolist()

                batch_predicted.append([(p[0], len(list(p[1]))) for p in groupby(predicted_shrink)])
                _gloss_fake = torch.LongTensor([p[0] for p in groupby(predicted_shrink)])
                gloss_fake.append(_gloss_fake)

            new_lengths = [len(p) for p in batch_predicted]
            weights_matrix = self.ctc_compress(new_feature, batch_predicted, new_lengths,
                                               new_feature.dtype, new_feature.device)

        new = weights_matrix.detach().cpu().numpy()

        # x is T x B x C -> B x C x T; weights_matrix is B x T x T'
        new_feature = new_feature.permute(0,2,1)
        compressed_output = new_feature.bmm(weights_matrix).type_as(new_feature)  # B x C x T'
        out = compressed_output.permute(2, 0, 1)

        out_lengths = lengths.new(new_lengths)
        padding = lengths_to_padding_mask(out_lengths)


        if self.out_norm:
            out = self.out_ln(out)

        gloss_fake_lengths = [len(p) for p in gloss_fake]
        return out, padding, gloss_fake, gloss_fake_lengths


