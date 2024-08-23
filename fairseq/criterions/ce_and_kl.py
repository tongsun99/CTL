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

@dataclass
class KL_LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    mixup_consistent_weight: float = field(
        default=0.0,
        metadata={"help": "the weight for consistency regularization of mixup"},
    )
    cal_mixup_loss: bool = field(
        default=True,
        metadata={"help": "calculate the loss for the mixed samples"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")




@register_criterion(
    "ce_and_kl")
class LabelSmoothedCrossEntropyCriterionWith_KL(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(self, task, label_smoothing,
                 sentence_avg,
                 kl_weight=0.0,
                 gloss_translation_weight=0.0,
                 distance_loss_weight=0.0,
                 save_dir=None,
                 cal_all_fake_loss=False,
                 cal_mixup_loss=True,
                 cal_kl_mixup_loss=False,
                 mixup_consistent_weight=0.0,
                 gloss_decoder_detach=False):
        super().__init__(task, sentence_avg, label_smoothing,
                         report_accuracy=True,
                         cal_mixup_loss=cal_mixup_loss,
                         mixup_consistent_weight=mixup_consistent_weight)

        self.report_accuracy = True
        self.kl_weight = kl_weight
        self.save_dir = save_dir
        self.gloss_detach = gloss_decoder_detach
        self.cal_kl_mixup_loss = cal_kl_mixup_loss
        self.distance_loss_weight = distance_loss_weight
        self.gloss_translation_weight = gloss_translation_weight
        self.cal_all_fake_loss = cal_all_fake_loss
        #self.distance_loss = nn.MSELoss(size_average=False, reduce=True, reduction='sum')

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)

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
        sign_encoder_out, gloss_encoder_out = model.encoder(**net_input)
        #fake
        net_input["gloss_src_tokens"] = sample["fake_gloss"]["tokens"]
        net_input["gloss_src_lengths"] = sample["fake_gloss"]["lengths"]
        fake_sign_encoder_out, fake_gloss_encoder_out = model.encoder(**net_input)

        net_output = model.decoder(prev_output_tokens=net_input["prev_output_tokens"], encoder_out=sign_encoder_out)
        fake_net_output = model.decoder(prev_output_tokens=net_input["prev_output_tokens"], encoder_out=fake_sign_encoder_out)

        gloss_decoder_output = model.decoder(prev_output_tokens=net_input["prev_output_tokens"], encoder_out=gloss_encoder_out)
        sign_loss, sign_nll_loss, other_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        # if self.cal_all_fake_loss:
        #     sign_fake_loss, sign_fake_nll_loss, _ = self.compute_loss(model, fake_net_output, sample, reduce=reduce)
        #     sign_loss = sign_loss + sign_fake_loss
        #     sign_nll_loss = sign_nll_loss + sign_fake_nll_loss
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
            if self.cal_all_fake_loss:
                fake_kl_loss = self._get_self_distill_loss(model, gloss_decoder_output, fake_net_output, sample,
                                                               self.temperature)
                loss = loss + self.kl_weight * fake_kl_loss
            logging_output["kl_loss"] = utils.item(kl_loss.data) if reduce else kl_loss.data

        if self.gloss_translation_weight:
            gloss_loss, gloss_nll_loss, _ = self.compute_loss(model, gloss_decoder_output, sample, reduce=reduce)
            loss = loss + self.gloss_translation_weight * gloss_loss
            logging_output["gloss_loss"] = utils.item(gloss_loss.data) if reduce else gloss_loss.data
            logging_output["gloss_nll_loss"] = utils.item(gloss_nll_loss.data) if reduce else gloss_nll_loss.data

        if self.distance_loss_weight:
            gloss_hidden = gloss_encoder_out["encoder_out"][0]
            sign_hidden = sign_encoder_out["shrink_encoder_out"][0]
            distance_loss = (1-torch.cosine_similarity(gloss_hidden,sign_hidden,dim=-1)).sum()
            logging_output["distance_loss"] = utils.item(distance_loss.data) if reduce else distance_loss.data
            loss = loss + distance_loss * self.distance_loss_weight

        logging_output["loss"] = utils.item(loss.data) if reduce else loss.data

        return loss, sample_size, logging_output

    # def gen_targets(self,align):
    #     gloss_max_len = 0
    #     for s in align:
    #         gloss_max_len = len(s) if gloss_max_len < len(s) else gloss_max_len
    #     bsz = len(align)
    #     targets = []
    #     for single_align in align:
    #         target = np.arange(len(single_align))
    #         target = np.pad(target,(0,gloss_max_len-len(single_align)),'constant', constant_values=(0,-1))
    #         index_word = 0
    #         for gloss_word, word_align in single_align.items():
    #             if(len(word_align)==0):
    #                 target[index_word] = -1
    #             index_word +=1
    #         target = torch.from_numpy(target).unsqueeze(0)
    #         targets.append(target)
    #     targets = torch.cat(targets, dim=0).cuda()
    #     return targets
    #
    # def _compute_contrast_loss(self, model, sign_encoder_out, gloss_encoder_out, sample, reduce=True,log_probs=False):
    #     sign_feature = sign_encoder_out["encoder_out"][0].permute(1,0,2)
    #     gloss_feature = gloss_encoder_out["encoder_out"][0].permute(1,0,2)
    #     sign_mask = gloss_encoder_out["encoder_padding_mask"][0]
    #     # zero_feature = torch.zeros(sign_feature.shape[0],1,sign_feature.shape[2]).to(device = sign_feature.device)
    #     # zero_mask = torch.zeros(sign_mask.shape[0],1) == 0
    #     # new_sign_feature =torch.cat((zero_feature, sign_feature),1)
    #     # new_sign_mask = torch.cat((zero_mask,sign_mask),1)
    #     #sign_feature[sign_mask] = 1/100000
    #     #gloss_feature[sign_mask] = 1/100000
    #     sim = sign_feature.matmul(gloss_feature.permute(0,2,1))
    #     # if log_probs:
    #     #     sim_sign = F.log_softmax(sim, dim=-1)
    #     #     sim_gloss = F.log_softmax(sim, dim=-1)
    #     # else:
    #     #     sim_sign =  F.softmax(sim, dim=-1)
    #     #     sim_gloss =  F.softmax(sim, dim=-1)
    #     arr = torch.arange(0,sign_mask.shape[1])
    #     target = arr.repeat(sign_mask.shape[0]).to(device=sign_mask.device)
    #
    #     sim_s = sim.view(-1, sim.size(-1))
    #     #sim_g = sim_gloss.view(-1, sim_gloss.size(-1))
    #
    #     new = sim.detach().cpu().numpy()
    #     new2 = sim_s.detach().cpu().numpy()
    #
    #     sign_con_smooth_loss, sign_con_loss = label_smoothed_nll_loss(
    #         sim_s,
    #         target,
    #         self.eps,
    #         ignore_index=None,
    #         reduce=reduce,
    #     )
    #     gloss_con_smooth_loss, gloss_con_loss = label_smoothed_nll_loss(
    #         sim_g,
    #         target,
    #         self.eps,
    #         ignore_index=None,
    #         reduce=reduce,
    #     )
    #     return sign_con_smooth_loss + gloss_con_smooth_loss, sign_con_loss + gloss_con_loss


    def _get_kl(self,d1, d2, temperature=1.0):

        p_loss = F.kl_div(F.log_softmax(d1 / temperature, dim=-1), F.softmax(d2 / temperature, dim=-1), log_target=False, reduction='none')
        q_loss = F.kl_div(F.log_softmax(d2 / temperature, dim=-1), F.softmax(d1 / temperature, dim=-1), log_target=False, reduction='none')

        p_loss = p_loss.sum(-1).sum()
        q_loss = q_loss.sum(-1).sum()
        loss = (p_loss + q_loss) / 2
        return loss


    def _get_self_distill_loss(self,model,gloss_decoder_output, net_output, sample, temperature=1.0):
        _, target = self.get_lprobs_and_target(model, net_output, sample)
        loss = 0

        #输出之前需不需要log正则化
        # student_logit = model.get_normalized_probs(net_output, log_probs=True, sample=sample)
        # teacher_logit = model.get_normalized_probs(gloss_decoder_output, log_probs=True, sample=sample)


        student_logit = net_output[0]
        teacher_logit = gloss_decoder_output[0]
        if self.gloss_detach:
            teacher_logit = teacher_logit.detach()


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
        encoder_kl_loss_sum = utils.item(
            sum(log.get("encoder_kl_loss", 0) for log in logging_outputs)
        )
        con_loss_sum = utils.item(
            sum(log.get("con_loss", 0) for log in logging_outputs)
        )
        enc_loss_sum = utils.item(
            sum(log.get("encoder_loss", 0) for log in logging_outputs)
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
        if sign_loss_sum != loss_sum:
            metrics.log_scalar(
                "sign_loss", sign_loss_sum / ntokens / math.log(2), ntokens, round=3
            )
        metrics.log_scalar(
            "sign_nll_loss", sign_nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "gloss_loss", gloss_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
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
        if encoder_kl_loss_sum != 0:
            metrics.log_scalar(
                "encoder_kl_loss_sum", encoder_kl_loss_sum / ntokens / math.log(2), n_sentences, round=3
            )
        if distance_loss != 0:
            metrics.log_scalar(
                "distance_loss", distance_loss / ntokens / math.log(2), n_sentences, round=3
            )
        if con_loss_sum != 0:
            metrics.log_scalar(
                "con_loss_sum", con_loss_sum / ntokens / math.log(2), n_sentences, round=3
            )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["sign_nll_loss"].avg)
        )
        if enc_loss_sum != 0:
            metrics.log_scalar("enc_loss", enc_loss_sum, sample_size, round=3)


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
