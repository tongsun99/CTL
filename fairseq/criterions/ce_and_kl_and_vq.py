# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.data.data_utils import post_process
from fairseq.logging.meters import safe_round
from fairseq.dataclass import FairseqDataclass
from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from fairseq.models.sign_to_text import S2TConcatModel
from omegaconf import II



@register_criterion(
    "ce_and_kl_and_vq")
class LabelSmoothedCrossEntropyCriterionWith_KL_VQ(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(self, task, label_smoothing,
                 sentence_avg,
                 kl_weight=0.0,
                 gloss_translation_weight=0.0,
                 distance_weight=0.0,
                 vq_weight=0.0,
                 ad_weight=0.0,
                 memory_num=32,
                 save_dir=None,
                 cal_all_fake_loss=False,
                 cal_mixup_loss=True,
                 mixup_consistent_weight=0.0,):
        super().__init__(task, sentence_avg, label_smoothing,
                         report_accuracy=True,
                         cal_mixup_loss=cal_mixup_loss,
                         mixup_consistent_weight=mixup_consistent_weight)

        self.report_accuracy = True
        self.kl_weight = kl_weight
        self.save_dir = save_dir

        self.distance_weight = distance_weight
        self.gloss_translation_weight = gloss_translation_weight
        self.cal_all_fake_loss = cal_all_fake_loss
        self.vq_weight = vq_weight
        self.ad_weight = ad_weight
        self.memory_num = memory_num
        self.zero_infinity = True
        self.type = "contrastive"
        self.contrastive_temp = 1.0
        self.teacher_mode = False
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument(
            "--kl-weight",
            default=0.0,
            type=float,
        )
        parser.add_argument(
            "--gloss-translation-weight",
            default=0.0,
            type=float,
        )
        parser.add_argument(
            "--distance-weight",
            default=0.0,
            type=float,
        )
        parser.add_argument(
            "--vq-weight",
            default=0.0,
            type=float,
        )
        parser.add_argument(
            "--ad-weight",
            default=0.0,
            type=float,
        )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_input = sample["net_input"]

        #truth
        net_input["gloss_src_tokens"] = sample["gloss"]["tokens"]
        net_input["gloss_src_lengths"] = sample["gloss"]["lengths"]
        # sign_encoder_out, gloss_encoder_out = model.encoder(**net_input)
        #
        # net_output = model.decoder(prev_output_tokens=net_input["prev_output_tokens"], encoder_out=sign_encoder_out)
        #
        # gloss_decoder_output = model.decoder(prev_output_tokens=net_input["prev_output_tokens"], encoder_out=gloss_encoder_out)

        sign_encoder_out, gloss_encoder_out,net_output,gloss_decoder_output = model(**net_input)
        sign_loss, sign_nll_loss, other_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        loss = sign_loss

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        n_tokens = sample["ntokens"]
        n_sentences = sample["target"].size(0)
        gloss_ntokens = sample["gloss"]["ntokens"]
        if "mixup" in net_output[1] and net_output[1]["mixup"] is not None:
            mixup = net_output[1]["mixup"]
            ratio = mixup["ratio"]

            if mixup["keep_org"]:
                n_tokens = int(n_tokens * (1 + ratio))
                sample_size = int(sample_size * (1 + ratio)) if self.sentence_avg else n_tokens
                n_sentences = int(n_sentences * (1 + ratio))
            else:
                if ratio > 1:
                    n_tokens = int(n_tokens * ratio)
                    sample_size = int(sample_size * ratio) if self.sentence_avg else n_tokens
                    n_sentences = int(n_sentences * ratio)

        logging_output = {
            "sign_loss": utils.item(sign_loss.data) if reduce else sign_loss.data,
            "sign_nll_loss": utils.item(sign_nll_loss.data) if reduce else sign_nll_loss.data,
            "ntokens": n_tokens,
            "nsentences": n_sentences,
            "sample_size": sample_size,
        }
        if len(other_loss) != 0:
            for key, value in other_loss.items():
                loss += value
                logging_output[key] = utils.item(value.data)

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)


        if self.kl_weight:
            self.temperature = 1.0
            kl_loss = self._get_self_distill_loss(model, gloss_decoder_output, net_output, sample,
                                                      self.temperature)
            loss = loss + self.kl_weight * kl_loss
            logging_output["kl_loss"] = utils.item(kl_loss.data) if reduce else kl_loss.data

        if self.gloss_translation_weight:
            gloss_loss, gloss_nll_loss, _ = self.compute_loss(model, gloss_decoder_output, sample, reduce=reduce)
            loss = loss + self.gloss_translation_weight * gloss_loss
            logging_output["gloss_loss"] = utils.item(gloss_loss.data) if reduce else gloss_loss.data
            logging_output["gloss_nll_loss"] = utils.item(gloss_nll_loss.data) if reduce else gloss_nll_loss.data

        if self.distance_weight and self.vq_weight == 0.0:
            gloss_d = gloss_encoder_out["gloss_distribution"]
            sign_d = sign_encoder_out["sign_distribution"]
            kl_distance_loss = self._get_kl(sign_d,gloss_d)
            logging_output["distance_loss"] = utils.item(kl_distance_loss.data) if reduce else kl_distance_loss.data
            loss = loss + kl_distance_loss * self.distance_weight

        if self.ad_weight:
            ad_loss = self.compute_ad_loss(sign_encoder_out, gloss_encoder_out, reduce)
            logging_output["ad_loss"] = utils.item(ad_loss.data) if reduce else ad_loss.data
            loss = loss + ad_loss * self.ad_weight

        if self.vq_weight:
            vq_loss = self.compute_vq_loss(sign_encoder_out, gloss_encoder_out, reduce)
            logging_output["vq_loss"] = utils.item(vq_loss.data) if reduce else vq_loss.data
            loss = loss + vq_loss * self.vq_weight

        logging_output["loss"] = utils.item(loss.data) if reduce else loss.data

        return loss, sample_size, logging_output


    def _get_kl(self,d1, d2, temperature=1.0):
        #mask没有做！
        p_loss = F.kl_div(F.log_softmax(d1 / temperature, dim=-1), F.softmax(d2 / temperature, dim=-1), log_target=False, reduction='none')
        q_loss = F.kl_div(F.log_softmax(d2 / temperature, dim=-1), F.softmax(d1 / temperature, dim=-1), log_target=False, reduction='none')

        p_loss = p_loss.sum(-1).sum()
        q_loss = q_loss.sum(-1).sum()
        loss = (p_loss + q_loss) / 2
        return loss

    def compute_vq_loss(self, speech_encoder_out, text_encoder_out, reduce=True):
        if self.memory_num > 0:
            speech_vq_info = speech_encoder_out["vq_info"]
            text_vq_info = text_encoder_out["vq_info"]
            probs = speech_vq_info["vq_logits"]  # (BxT) x G x V
            lprobs = torch.log(probs + 1e-8)
            if self.teacher_mode:
                targets = text_vq_info["vq_logits"]  # (BxT) x G x V
                loss = torch.sum(-lprobs * targets, dim=-1)  # (BxT) X G
            else:
                targets = text_vq_info["vq_targets"]  # (BxT) X G
                targets = targets.unsqueeze(-1)
                loss = -lprobs.gather(dim=-1, index=targets)
            loss = loss.mean(-1)  # (BxT)
            if reduce:
                loss = loss.sum()
            return loss
        else:
            raise NotImplementedError
    def compute_ad_loss(self, speech_encoder_out, text_encoder_out, reduce=True):
        if self.memory_num>0:
            input1 = speech_encoder_out["fixed_encoder_out"][0]
            input2 = text_encoder_out["fixed_encoder_out"][0]
            assert input1.shape == input2.shape
            input1 = input1.transpose(0, 1)
            input2 = input2.transpose(0, 1)  # [batch, seqlen, dim]
            if self.type == "contrastive":
                batch_size, seqlen, _ = input1.shape
                logits = torch.cosine_similarity(
                    input1.float().unsqueeze(2),
                    input2.float().unsqueeze(1),
                    dim=-1
                ).type_as(input1)
                logits /= self.contrastive_temp
                target = torch.arange(seqlen)[None].repeat(batch_size, 1) \
                    .to(logits.device)
                loss = F.cross_entropy(logits, target,
                                       reduction='sum' if reduce else "none")
            elif self.type == "mse":
                loss = 0.5 * (input1 - input2) ** 2
                loss = loss.sum(-1)
                if reduce:
                    loss = loss.sum()
            else:
                raise NotImplementedError
            return loss
        else:
            raise NotImplementedError
    def _get_self_distill_loss(self,model,gloss_decoder_output, net_output, sample, temperature=1.0):
        _, target = self.get_lprobs_and_target(model, net_output, sample)
        loss = 0

        #输出之前需不需要log正则化
        # student_logit = model.get_normalized_probs(net_output, log_probs=True, sample=sample)
        # teacher_logit = model.get_normalized_probs(gloss_decoder_output, log_probs=True, sample=sample)


        student_logit = net_output[0]
        teacher_logit = gloss_decoder_output[0]
        # if self.gloss_detach:
        #     teacher_logit = teacher_logit.detach()


        if student_logit.shape[0] != teacher_logit.shape[0]:
            raise ValueError("can not do KD")
        if "mixup" in net_output[1] and net_output[1]["mixup"] is not None:
            mixup = net_output[1]["mixup"]
            idx1 = mixup["index1"]
            idx2 = mixup["index2"]
            mixup_flag = mixup["mixup_flag"]
            mixup_idx1 = idx1[mixup_flag]
            mixup_idx2 = idx2[mixup_flag]
            org_idx = idx1[~mixup_flag]

            seq_len = target.size(1)
            student_logit = student_logit.view(-1, seq_len, student_logit.size(-1))
            teacher_logit = teacher_logit.view(-1, seq_len, teacher_logit.size(-1))

            if mixup["mixup_decoder_emb"]:
                mixup_student_logit = student_logit[mixup_flag, :, :]
                mixup_teacher_logit = teacher_logit[mixup_flag, :, :]
            else:
                #暂时不考虑这种情况
                raise EOFError
                # decoder_mixup_flag1 = mixup["decoder_mixup_flag1"]
                # decoder_mixup_flag2 = mixup["decoder_mixup_flag2"]
                # mixup_student_logit = [student_logit[decoder_mixup_flag1, :, :], student_logit[decoder_mixup_flag2, :, :]]
                # mixup_teacher_logit = [teacher_logit[decoder_mixup_flag1, :, :], teacher_logit[decoder_mixup_flag2, :, :]]

            org_student_logit = student_logit[org_idx, :, :]
            org_teacher_logit = teacher_logit[org_idx, :, :]



            if len(org_idx) > 0:
                org_target = target[org_idx]
                padding_mask = org_target.eq(self.padding_idx)
                p_loss = F.kl_div(F.log_softmax(org_student_logit / temperature, dim=-1),
                                  F.softmax(org_teacher_logit / temperature, dim=-1), log_target=False, reduction='none')
                q_loss = F.kl_div(F.log_softmax(org_teacher_logit / temperature, dim=-1),
                                  F.softmax(org_student_logit / temperature, dim=-1), log_target=False, reduction='none')

                p_loss = p_loss.sum(-1).masked_fill_(padding_mask, 0.0).sum()
                q_loss = q_loss.sum(-1).masked_fill_(padding_mask, 0.0).sum()
                loss = (p_loss + q_loss) / 2

            if self.cal_kl_mixup_loss:
                mixup_targets_1 = target[mixup_idx1]
                mixup_targets_2 = target[mixup_idx2]
                padding_mask_1 = mixup_targets_1.eq(self.padding_idx)
                padding_mask_2 = mixup_targets_2.eq(self.padding_idx)
                padding_mask = padding_mask_1 & padding_mask_2
                p_loss = F.kl_div(F.log_softmax(mixup_student_logit / temperature, dim=-1),
                                  F.softmax(mixup_teacher_logit / temperature, dim=-1), log_target=False, reduction='none')
                q_loss = F.kl_div(F.log_softmax(mixup_teacher_logit / temperature, dim=-1),
                                  F.softmax(mixup_student_logit / temperature, dim=-1), log_target=False, reduction='none')

                p_loss = p_loss.sum(-1).masked_fill_(padding_mask, 0.0).sum()
                q_loss = q_loss.sum(-1).masked_fill_(padding_mask, 0.0).sum()
                mixup_loss = (p_loss + q_loss) / 2
                loss = loss + mixup_loss



        else:
            padding_mask = target.eq(self.padding_idx)
            p_loss = F.kl_div(F.log_softmax(student_logit / temperature, dim=-1), F.softmax(teacher_logit / temperature, dim=-1), log_target=False, reduction='none')
            q_loss = F.kl_div(F.log_softmax(teacher_logit / temperature, dim=-1), F.softmax(student_logit / temperature, dim=-1), log_target=False, reduction='none')

            p_loss = p_loss.sum(-1).masked_fill_(padding_mask, 0.0).sum()
            q_loss = q_loss.sum(-1).masked_fill_(padding_mask, 0.0).sum()
            loss = (p_loss + q_loss) / 2
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        sign_loss_sum = utils.item(
            sum(log.get("sign_loss", 0) for log in logging_outputs)
        )
        sign_nll_loss_sum = utils.item(
            sum(log.get("sign_nll_loss", 0) for log in logging_outputs)
        )
        gloss_loss_sum = utils.item(
            sum(log.get("gloss_loss", 0) for log in logging_outputs)
        )
        gloss_nll_loss_sum = utils.item(
            sum(log.get("gloss_nll_loss", 0) for log in logging_outputs)
        )
        mixup_consistent_loss_sum = utils.item(
            sum(log.get("mixup_consistent_loss", 0) for log in logging_outputs)
        )
        kl_loss_sum = utils.item(
            sum(log.get("kl_loss", 0) for log in logging_outputs)
        )
        vq_loss = utils.item(
            sum(log.get("vq_loss", 0) for log in logging_outputs)
        )
        ad_loss = utils.item(
            sum(log.get("ad_loss", 0) for log in logging_outputs)
        )
        distance_loss = utils.item(
            sum(log.get("distance_loss", 0) for log in logging_outputs)
        )
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        n_sentences = utils.item(sum(log.get("nsentences", 0) for log in logging_outputs))
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "sign_loss", sign_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "sign_nll_loss", sign_nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        if gloss_loss_sum != 0:
            metrics.log_scalar(
                "gloss_loss", gloss_loss_sum / ntokens / math.log(2), ntokens, round=3
            )
        if gloss_nll_loss_sum != 0:
            metrics.log_scalar(
                "gloss_nll_loss", gloss_nll_loss_sum / ntokens / math.log(2), ntokens, round=3
            )
        if mixup_consistent_loss_sum != 0:
            metrics.log_scalar(
                "mixup_consistent_loss", mixup_consistent_loss_sum / n_sentences / math.log(2), n_sentences, round=3
            )
        if kl_loss_sum != 0:
            metrics.log_scalar(
                "kl_loss", kl_loss_sum / ntokens / math.log(2), n_sentences, round=3
            )#改动，如果mixup数据不做kl，可能除下来会小了二倍
        if ad_loss != 0:
            metrics.log_scalar(
                "ad_loss", ad_loss / ntokens / math.log(2), n_sentences, round=3
            )
        if distance_loss != 0:
            metrics.log_scalar(
                "distance_loss", distance_loss / ntokens / math.log(2), n_sentences, round=3
            )
        if vq_loss != 0:
            metrics.log_scalar(
                "vq_loss", vq_loss / ntokens / math.log(2), n_sentences, round=3
            )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["sign_nll_loss"].avg)
        )


        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
