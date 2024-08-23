# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import io
import logging
import re
import pickle
import gzip
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from fairseq.data import (
    ConcatDataset,
    Dictionary,
    FairseqDataset,
    ResamplingDataset,
    data_utils as fairseq_data_utils,
)
from fairseq.data.sign.data_cfg import S2TDataConfig
from .video_augmentation import *


logger = logging.getLogger(__name__)
def _collate_frames(
    frames: List[torch.Tensor], is_sign_input: bool = False
) -> torch.Tensor:
    """
    Convert a list of 2D frames into a padded 3D tensor
    Args:
        frames (list): list of 2D frames of size L[i]*f_dim. Where L[i] is
            length of i-th frame and f_dim is static dimension of features
    Returns:
        3D tensor of size len(frames)*len_max*f_dim where len_max is max of L[i]
    """
    max_len = max(frame.size(0) for frame in frames)
    if is_sign_input:
        out = frames[0].new_zeros((len(frames), max_len))
    else:
        out = frames[0].new_zeros((len(frames), max_len, frames[0].size(1)))
    for i, v in enumerate(frames):
        out[i, : v.size(0)] = v
    return out


#sign to gloss and text
@dataclass
class VideoToTextDatasetItem(object):
    index: int
    sign: torch.Tensor
    gloss: torch.Tensor
    fake_gloss: torch.Tensor
    text: torch.Tensor
    video: torch.Tensor
    vac_label: torch.Tensor
    signer_id: Optional[int] = None

class VideoToTextDataset(FairseqDataset):
    def __init__(
        self,
        split: str,
        is_train_split: bool,
        cfg: S2TDataConfig,
        n_frames: List[int],
        sign: Optional[List[torch.Tensor]] = None,
        text: Optional[List[str]] = None,
        gloss: Optional[List[str]] = None,
        fake_gloss: Optional[List[str]] = None,
        signer: Optional[List[str]] = None,
        name: Optional[List[str]] = None,
        gloss_dict: Optional[Dictionary] = None,
        text_dict: Optional[Dictionary] = None,
        vac_dict: Optional[Dictionary] = None,
        pre_tokenizer=None,
        bpe_tokenizer=None,
        append_eos=True,
    ):
        self.split, self.is_train_split = split, is_train_split
        self.cfg = cfg
        self.n_frames = n_frames
        self.sign = sign
        self.n_samples = len(sign)

        self.text_dict = text_dict
        self.gloss_dict = gloss_dict
        self.vac_dict = vac_dict



        assert len(n_frames) == self.n_samples > 0
        assert text is None or len(text) == self.n_samples
        assert signer is None or len(signer) == self.n_samples
        assert name is None or len(name) == self.n_samples
        assert (text_dict is None and text is None) or (
            text_dict is not None and text is not None
        )
        assert (gloss_dict is None and gloss is None) or (
            gloss_dict is not None and gloss is not None
        )
        self.gloss = gloss
        self.fake_gloss = fake_gloss
        self.text = text
        self.signer = signer
        self.name = name
        self.shuffle = cfg.shuffle if is_train_split else False

        self.pre_tokenizer = pre_tokenizer
        self.bpe_tokenizer = bpe_tokenizer



        self.src_lens = self.get_src_lens_and_check_oov()
        self.tgt_lens = self.get_tgt_lens_and_check_oov()
        self.append_eos = append_eos

        #video
        self.prefix = "/home3/chu/vac/dataset/phoenix2014T"
        self.vac_dict = vac_dict
        self.transform_mode = split
        #self.inputs_list = np.load(f"../preprocess/phoenix2014T/{split}_info.npy", allow_pickle=True).item()
        # self.inputs_list = np.load(f"{prefix}/annotations/manual/{mode}.corpus.npy", allow_pickle=True).item()
        # self.inputs_list = np.load(f"{prefix}/annotations/manual/{mode}.corpus.npy", allow_pickle=True).item()
        # self.inputs_list = dict([*filter(lambda x: isinstance(x[0], str) or x[0] < 10, self.inputs_list.items())])
        print(split, len(self))
        self.data_aug = self.transform()
        print("")


        logger.info(self.__repr__())

    def get_tokenized_src_text(self, index: int):
        gloss = self.tokenize(self.pre_tokenizer, self.gloss[index])
        gloss = self.tokenize(self.bpe_tokenizer, gloss)
        return gloss

    def get_tokenized_tgt_text(self, index: int):
        text = self.tokenize(self.pre_tokenizer, self.text[index])
        text = self.tokenize(self.bpe_tokenizer, text)
        return text

    def get_src_lens_and_check_oov(self):
        if self.gloss is None:
            return [0 for _ in range(self.n_samples)]
        src_lens = []
        n_tokens, n_oov_tokens = 0, 0
        for i in range(self.n_samples):
            tokenized = self.get_tokenized_src_text(i).split(" ")
            oov_tokens = [
                t
                for t in tokenized
                if self.gloss_dict.index(t) == self.gloss_dict.unk_index
            ]
            n_tokens += len(tokenized)
            n_oov_tokens += len(oov_tokens)
            src_lens.append(len(tokenized))
        logger.info(f"'{self.split}-gloss' has {n_oov_tokens / n_tokens * 100:.2f}% OOV")
        return src_lens

    def get_tgt_lens_and_check_oov(self):
        if self.text is None:
            return [0 for _ in range(self.n_samples)]
        tgt_lens = []
        n_tokens, n_oov_tokens = 0, 0
        for i in range(self.n_samples):
            tokenized = self.get_tokenized_tgt_text(i).split(" ")
            oov_tokens = [
                t
                for t in tokenized
                if self.text_dict.index(t) == self.text_dict.unk_index
            ]
            n_tokens += len(tokenized)
            n_oov_tokens += len(oov_tokens)
            tgt_lens.append(len(tokenized))
        logger.info(f"'{self.split}-text' has {n_oov_tokens / n_tokens * 100:.2f}% OOV")
        return tgt_lens

    def __repr__(self):
        return (
            self.__class__.__name__
            + f'(split="{self.split}", n_samples={self.n_samples:_}, '
            f"shuffle={self.shuffle}, "
        )

    @classmethod
    def tokenize(cls, tokenizer, text: str):
        return text if tokenizer is None else tokenizer.encode(text)

    def get_tokenized_gloss(self, index: int):
        gloss = self.tokenize(self.pre_tokenizer, self.gloss[index])
        gloss = self.tokenize(self.bpe_tokenizer, gloss)
        return gloss

    def get_fake_tokenized_gloss(self, index: int):
        fake_gloss = self.tokenize(self.pre_tokenizer, self.fake_gloss[index])
        fake_gloss = self.tokenize(self.bpe_tokenizer, fake_gloss)
        return fake_gloss

    def get_tokenized_text(self, index: int):
        text = self.tokenize(self.pre_tokenizer, self.text[index])
        text = self.tokenize(self.bpe_tokenizer, text)
        return text

    def pack_frames(self, feature: torch.Tensor):
        if self.n_frames_per_step == 1:
            return feature
        n_packed_frames = feature.shape[0] // self.n_frames_per_step
        feature = feature[: self.n_frames_per_step * n_packed_frames]
        return feature.reshape(n_packed_frames, -1)

    def _get_source_sign(self, index: int) -> torch.Tensor:
        source = self.sign[index]
        return source

    def read_video(self, index, num_glosses=-1):
        # load file info
        import os
        import glob
        import cv2
        img_folder = os.path.join(self.prefix, "features/fullFrame-256x256px/" + self.name[index] + "/*.png")
        img_list = sorted(glob.glob(img_folder))
        label_list = []
        for phase in self.gloss[index].split(" "):
            if phase == '':
                continue
            if phase.upper() in self.vac_dict:
                label_list.append(self.vac_dict.indices[phase.upper()])
        return [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list], label_list

    def transform(self):

        if self.transform_mode == "train":
            print("Apply training transform.")
            return Compose([
                # video_augmentation.CenterCrop(224),
                # video_augmentation.WERAugment('/lustre/wangtao/current_exp/exp/baseline/boundary.npy'),
                RandomCrop(224),
                RandomHorizontalFlip(0.5),
                ToTensor(),
                TemporalRescale(0.2),
                # video_augmentation.Resize(0.5),
            ])
        elif self.transform_mode == "valid" or "test":
            print("Apply testing transform.")
            return Compose([
                CenterCrop(224),
                # video_augmentation.Resize(0.5),
                ToTensor(),
            ])
    def normalize(self, video, label, file_id=None):
        video, label = self.data_aug(video, label, file_id)
        video = video.float() / 127.5 - 1
        return video, label
    def __getitem__(self, index: int) -> VideoToTextDatasetItem:
        sign = self._get_source_sign(index)
        #sign = self.pack_frames(sign)

        # VIDEO
        input_data, label = self.read_video(index)
        input_data, label = self.normalize(input_data, label)
        vac_label = torch.LongTensor(label)
        # input_data, label = self.normalize(input_data, label, fi['fileid'])

        gloss_tokenized = self.get_tokenized_gloss(index)
        fake_gloss_tokenized = self.get_fake_tokenized_gloss(index)
        gloss = self.gloss_dict.encode_line(
            gloss_tokenized, add_if_not_exist=False, append_eos=False
        ).long()
        fake_gloss = self.gloss_dict.encode_line(
            fake_gloss_tokenized, add_if_not_exist=False, append_eos=False
        ).long()
        #在这没加eos
        tokenized = self.get_tokenized_text(index)
        text = self.text_dict.encode_line(
            tokenized, add_if_not_exist=False, append_eos=self.append_eos
        ).long()

        signer_id = None
        # if self.signer_to_id is not None:
        #     signer_id = self.signer_to_id[self.signers[index]]
        return VideoToTextDatasetItem(
            index=index, sign=sign, gloss=gloss, fake_gloss=fake_gloss,text=text,
            video=input_data,
            vac_label=vac_label,
            signer_id=signer_id
        )

    def __len__(self):
        return self.n_samples


    def collate_video(self,video):
        if len(video[0].shape) > 3:
            max_len = max(v.size(0) for v in video)
            video_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 12 for vid in video])
            left_pad = 6
            right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 6
            max_len = max_len + left_pad + right_pad
            padded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video)
        else:
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            padded_video = [torch.cat(
                (
                    vid,
                    vid[-1][None].expand(max_len - len(vid), -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video).permute(0, 2, 1)
        return padded_video, video_length

    def collater(
        self, samples: List[VideoToTextDatasetItem], return_order: bool = False
    ) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)

        # sort samples by descending number of frames
        frames = _collate_frames([x.sign for x in samples])
        n_frames = torch.tensor([x.sign.size(0) for x in samples], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        # sort video
        padded_video, video_length = self.collate_video([x.video for x in samples])
        n_video = torch.tensor([x.video.size(0) for x in samples], dtype=torch.long)
        n_video, order = n_video.sort(descending=True)
        indices = indices.index_select(0, order)
        video = padded_video.index_select(0, order)

        vac_label = fairseq_data_utils.collate_tokens(
            [x.vac_label for x in samples],
            self.vac_dict.pad(),
            self.vac_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=False,
        )
        vac_label = vac_label.index_select(0, order)
        vac_label_length = torch.tensor(
            [x.vac_label.size(0) for x in samples], dtype=torch.long
        ).index_select(0, order)


        gloss = fairseq_data_utils.collate_tokens(
            [x.gloss for x in samples],
            self.gloss_dict.pad(),
            self.gloss_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=False,
        )
        gloss = gloss.index_select(0, order)
        gloss_lengths = torch.tensor(
            [x.gloss.size(0) for x in samples], dtype=torch.long
        ).index_select(0, order)


        fake_gloss = fairseq_data_utils.collate_tokens(
            [x.fake_gloss for x in samples],
            self.gloss_dict.pad(),
            self.gloss_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=False,
        )
        fake_gloss = fake_gloss.index_select(0, order)
        fake_gloss_lengths = torch.tensor(
            [x.fake_gloss.size(0) for x in samples], dtype=torch.long
        ).index_select(0, order)

        text = fairseq_data_utils.collate_tokens(
            [x.text for x in samples],
            self.text_dict.pad(),
            self.text_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=False,
        )

        text = text.index_select(0, order)
        text_lengths = torch.tensor(
            [x.text.size(0) for x in samples], dtype=torch.long
        ).index_select(0, order)

        prev_output_tokens = fairseq_data_utils.collate_tokens(
            [x.text for x in samples],
            self.text_dict.pad(),
            self.text_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=True,
        )
        prev_output_tokens = prev_output_tokens.index_select(0, order)
        ntokens = sum(x.text.size(0) for x in samples)
        gloss_ntokens = sum(x.gloss.size(0) for x in samples)
        fake_gloss_ntokens = sum(x.fake_gloss.size(0) for x in samples)
        signer = None
        # if self.signer_to_id is not None:
        #     signer = (
        #         torch.tensor([s.signer_id for s in samples], dtype=torch.long)
        #         .index_select(0, order)
        #         .view(-1, 1)
        #     )

        net_input = {
            "video": video,
            "video_length": n_video,
            "src_tokens": frames,
            "src_lengths": n_frames,
            "prev_output_tokens": prev_output_tokens,
        }
        out = {
            "id": indices,
            "net_input": net_input,
            "signer": signer,
            "vac_label": vac_label,
            "vac_label_length": vac_label_length,
            "gloss": {
                "tokens": gloss,
                "lengths": gloss_lengths,
                "ntokens": gloss_ntokens,
            },
            "fake_gloss": {
                "tokens": fake_gloss,
                "lengths": fake_gloss_lengths,
                "ntokens": fake_gloss_ntokens,
            },
            "target": text,
            "target_lengths": text_lengths,
            "ntokens": ntokens,
            "nsentences": len(samples),
        }
        if return_order:
            out["order"] = order
        return out

    def num_tokens(self, index):
        return self.n_frames[index]

    def size(self, index):
        return self.n_frames[index], self.src_lens[index], self.tgt_lens[index]

    @property
    def sizes(self):
        return np.array(self.n_frames)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        # first by descending order of # of frames then by original/random order
        order.append([-n for n in self.n_frames])
        return np.lexsort(order)

    def prefetch(self, indices):
        raise False
class VideoToTextDatasetCreator(object):

    @classmethod
    def _load_pickle_file(cls,filename):
        with gzip.open(filename, "rb") as f:
            loaded_object = pickle.load(f)
            return loaded_object
    @classmethod
    def _load_samples_from_pickle(cls, root: str, split: str):
        pickle_path = Path(root) / f"fake_2014T_{split}"
        if not pickle_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {pickle_path}")
        samples = []
        tmp = cls._load_pickle_file(pickle_path)

        for s in tmp:
            #sample = {}
            #seq_id = s["name"]
            sample = {
                "name": s["name"],
                "signer_id": s["signer_id"],
                "gloss": s["gloss"].lower(),
                "fake_gloss": s["fake"],
                "text": s["text"],
                "sign": s["sign"],
                "n_frames": len(s["sign"]),
            }
            samples.append(sample)
        if len(samples) == 0:
            raise ValueError(f"Empty manifest: {pickle_path}")
        return samples

    @classmethod
    def _from_pickle(
        cls,
        root: str,
        cfg: S2TDataConfig,
        split: str,
        gloss_dict,
        text_dict,
        vac_dict,
        is_train_split: bool,
        pre_tokenizer,
        bpe_tokenizer,
    ) -> VideoToTextDataset:

        samples = cls._load_samples_from_pickle(root, split)
        sign_root = Path(cfg.sign_root)
        name = [s["name"] for s in samples]
        n_frames = [int(s["n_frames"]) for s in samples]
        all_text = [s["text"] for s in samples]
        all_gloss = [s["gloss"] for s in samples]
        all_fake_gloss = [s["gloss"] for s in samples]
        signer = [s["signer_id"] for s in samples]
        all_sign_feature = [s["sign"] for s in samples]
        return VideoToTextDataset(
            split,
            is_train_split,
            cfg,
            n_frames,
            sign = all_sign_feature,
            text = all_text,
            gloss = all_gloss,
            fake_gloss=all_fake_gloss,
            signer = signer,
            name = name,
            gloss_dict = gloss_dict,
            text_dict=text_dict,
            vac_dict=vac_dict,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
        )

    @classmethod
    def from_pickle(
        cls,
        root: str,
        cfg: S2TDataConfig,
        splits: str,
        gloss_dict,
        text_dict,
        vac_dict,
        pre_tokenizer,
        bpe_tokenizer,
        is_train_split: bool,
    ) -> VideoToTextDataset:
        datasets = [
            cls._from_pickle(
                root,
                cfg,
                split,
                gloss_dict,
                text_dict,
                vac_dict,
                is_train_split,
                pre_tokenizer,
                bpe_tokenizer,
            )
            for split in splits.split(",")
        ]
        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
