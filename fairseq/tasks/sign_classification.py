# Copyright (c) ASAPP Inc.
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
import logging
import os
import json
from omegaconf import DictConfig
import torch

from dataclasses import dataclass

from fairseq.data import encoders
from fairseq.tasks.audio_pretraining import AudioPretrainingTask, AudioPretrainingConfig
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import GenerationConfig
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional
from omegaconf import MISSING, II
from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from fairseq.tasks import register_task, FairseqTask
from fairseq.data.sign.add_label_dataset_cls import AddLabelDatasetCls
from fairseq.data.sign.feat_sign_dataset import FeatSignDataset
from fairseq.data.text_compressor import TextCompressionLevel


logger = logging.getLogger(__name__)


@dataclass
class SignClassificationConfig(AudioPretrainingConfig):
    data_bin: str = field(
        default="/home2/tsun/data/sign/vacs2t/PHOENIX2014T/pami", metadata={"help": "path to data bin"}
    )
    label_path: str = field(
        default="/home2/tsun/code/vac_s2t/data/sign_cls/configs/remove.json", metadata={"help": "path to label json"}
    )
    all_test: bool = field(
        default=False, metadata={"help": "whether to use all test data"}
    )


@register_task("sign_classification", dataclass=SignClassificationConfig)
class SignClassificationTask(FairseqTask):

    cfg: SignClassificationConfig

    def __init__(
        self,
        cfg: SignClassificationConfig,
    ):
        super().__init__(cfg)
        self.blank_symbol = "<s>"

        self.state.add_factory("label2id", self.load_label2id)
    
    @classmethod
    def setup_task(cls, cfg: SignClassificationConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (SignClassificationConfig): configuration of this task
        """
        return cls(cfg)

    
    def load_label2id(self):
        if self.cfg.label_path is not None:
            label2id = json.load(open(self.cfg.label_path, "r"))
        else:
            label2id = {"pos": 1, "neg": 0}
        return label2id

    def load_dataset(
        self, split: str, task_cfg: SignClassificationConfig = None, **kwargs
    ):
        data_path = self.cfg.data
        data_bin = self.cfg.data_bin
        task_cfg = task_cfg or self.cfg

        # upgrade old task
        if isinstance(task_cfg, Namespace):
            if not hasattr(task_cfg, "autoregressive"):
                task_cfg.autoregressive = not task_cfg.criterion == "ctc"

        text_compression_level = getattr(
            TextCompressionLevel, str(self.cfg.text_compression_level)
        )

        compute_mask = task_cfg.precompute_mask_config is not None
        mask_args = {}
        if compute_mask:
            mask_args = task_cfg.precompute_mask_config
        
        if task_cfg.all_test:
            self.datasets[split] = FeatSignDataset(
                manifest_path=os.path.join(data_path, split + ".json"), # 二分json
                data_bin=os.path.join(data_bin, 'test'),   # 手语pkl test
                sample_rate=task_cfg.sample_rate,   # useless
                max_sample_size=self.cfg.max_sample_size,
                pad=task_cfg.labels is not None or task_cfg.enable_padding,
                normalize=task_cfg.normalize,
                num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
                text_compression_level=text_compression_level,
                compute_mask=compute_mask,
                **mask_args,
            )
        else:
            self.datasets[split] = FeatSignDataset(
                manifest_path=os.path.join(data_path, split + ".json"), # 二分json
                data_bin=os.path.join(data_bin, split),    # 手语pkl
                sample_rate=task_cfg.sample_rate,   # useless
                max_sample_size=self.cfg.max_sample_size,
                pad=task_cfg.labels is not None or task_cfg.enable_padding,
                normalize=task_cfg.normalize,
                num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
                text_compression_level=text_compression_level,
                compute_mask=compute_mask,
                **mask_args,
            )

        assert task_cfg.labels is not None
        
        label_path = os.path.join(data_path, split + ".json")
        skipped_indices = getattr(self.datasets[split], "skipped_indices", set())
        logger.info(f"label2id: {self.label2id}")
        labels = []
        with open(label_path, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                json_obj = json.loads(line)
                if i not in skipped_indices:
                    labels.append(json_obj["label"])

        assert len(labels) == len(self.datasets[split]), (
            f"labels length ({len(labels)}) and dataset length "
            f"({len(self.datasets[split])}) do not match"
        )

        self.datasets[split] = AddLabelDatasetCls(
            self.datasets[split],
            labels,
        )

    @property
    def label2id(self):
        return self.state.label2id
    
    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
    
    def build_model(self, model_cfg: FairseqDataclass, from_checkpoint=False):
        model = super().build_model(model_cfg, from_checkpoint)
        return model
