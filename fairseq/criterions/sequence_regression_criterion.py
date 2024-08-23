# Copyright (c) ASAPP Inc.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# MODOFY: sign sequence classification criterion

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask
from fairseq.logging.meters import safe_round
from fairseq.utils import is_xla_tensor
from sklearn.metrics import f1_score
from functools import partial
import json

# https://discuss.pytorch.org/t/custom-tweedie-loss-throwing-an-error-in-pytorch/76349/6
def tweedieloss(predicted, observed, p=1.5):
    '''
    Custom loss fuction designed to minimize the deviance using stochastic gradient descent
    tweedie deviance from McCullagh 1983
    '''
    p = torch.tensor(p, dtype=predicted.dtype, device=predicted.device)
    QLL = torch.pow(predicted, (-p))*(((predicted*observed)/(1-p)) - ((torch.pow(predicted, 2))/(2-p)))
    d = -2 * QLL
    return torch.mean(d)

@dataclass
class SequenceRegressionCriterionConfig(FairseqDataclass):
    report_metric: bool = field(default=True, metadata={"help": "report metric"})
    loss_fn: str = field(default='mse', metadata={"help": "loss function"})
    weight_factor: float = field(default=1.0, metadata={"help": "weight factor for custom_mse loss, >1.0 for pred > gt"})


# add slue_ as prefix of the registerred name in case there are conflicts in future
@register_criterion(
    "slue_sequence_regression", dataclass=SequenceRegressionCriterionConfig
)
class SequenceRegressionCriterion(FairseqCriterion):
    def __init__(self, cfg: SequenceRegressionCriterionConfig, task: FairseqTask):
        super().__init__(task)
        self.cfg = cfg
        self.num_classes = len(task.label2id)
        self.task_cfg = task.cfg
        

    def forward(self, model, sample, reduce=True, log_pred=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])   # @mark 手语回归计算loss
        if self.cfg.loss_fn == 'mse':
            loss = F.mse_loss(
                net_output["pooled"].squeeze(-1),
                sample["target"].to(net_output["pooled"].dtype),
                reduction='none')
        elif self.cfg.loss_fn == 'smooth_l1':
            loss = F.smooth_l1_loss(
                net_output["pooled"].squeeze(-1),
                sample["target"].to(net_output["pooled"].dtype),
                reduction='none')
        # 假设训练集幂律分布的函数
        # 17310.5429356912 * x ** -0.6402147560545366
        elif self.cfg.loss_fn == 'weighted_mse':
            input = net_output["pooled"].squeeze(-1)
            target = sample["target"].to(net_output["pooled"].dtype)
            if self.task_cfg.label_norm == "min_max":
                min = 1; max = 68
                target_before_norm = target * (max - min) + min
            elif self.task_cfg.label_norm == "log":
                target_before_norm = torch.exp(target)
            else:
                target_before_norm = target
            weight = 17310.5429356912 * target_before_norm ** -0.6402147560545366 # 
            weight = 1000 / weight
            loss = torch.sum(weight * (input - target) ** 2)
        elif self.cfg.loss_fn == "tweedie_loss":
            input = net_output["pooled"].squeeze(-1)
            target = sample["target"].to(net_output["pooled"].dtype)
            loss = tweedieloss(input, target, p=1.5)
        elif self.cfg.loss_fn == "poisson_loss":
            input = net_output["pooled"].squeeze(-1)
            target = sample["target"].to(net_output["pooled"].dtype)
            loss = F.poisson_nll_loss(input, target)
        elif self.cfg.loss_fn == "custom_mse":
            input = net_output["pooled"].squeeze(-1)
            target = sample["target"].to(net_output["pooled"].dtype)
            diff = input - target
            loss = self.cfg.weight_factor * torch.square(diff) * (diff > 0).float() + torch.square(diff) * (diff <= 0).float()
        else:
            raise NotImplementedError
        
        # 有bos
        if self.task_cfg.label_norm == "min_max":
            min = 1; max = 68
            pred = net_output["pooled"].squeeze(-1) * (max - min) + min
            gt = sample["target"] * (max - min) + min
        elif self.task_cfg.label_norm == "log":
            pred = torch.exp(net_output["pooled"].squeeze(-1))
            gt = torch.exp(sample["target"])
        else:
            pred = net_output["pooled"].squeeze(-1) + 1
            gt = sample["target"] + 1

        sample_size = loss.numel()
        if reduce:
            loss = loss.sum()

        logging_output = {
            "loss": loss.item() if reduce else loss,
            # "ntokens": sample_size['net_input']['source'].numel(),
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        if self.cfg.report_metric:
            # 1. 平均相对误差
            mape = torch.abs((pred - gt) / (gt + 1e-8))
            mape = torch.mean(mape) * 100
            logging_output["mape"] = mape.item()
            # 2. MSE
            mse = F.mse_loss(pred, gt, reduction='mean')
            logging_output["mse"] = mse.item()
            

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        # ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        mape = utils.item(sum(log.get("mape", 0) for log in logging_outputs))
        mse = utils.item(sum(log.get("mse", 0) for log in logging_outputs))

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        # metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        metrics.log_scalar("mape", mape, round=3)
        metrics.log_scalar("mse", mse, round=3)



    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
