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


# sign to gloss and text
@dataclass
class SignAndGlossTranslationDatasetItem(object):
    index: int
    sign: torch.Tensor
    gloss: torch.Tensor
    text: torch.Tensor
    signer_id: Optional[int] = None


@dataclass
class SignToTextDatasetItem(object):
    index: int
    sign: torch.Tensor
    gloss: torch.Tensor
    fake_gloss: torch.Tensor
    text: torch.Tensor
    signer_id: Optional[int] = None


class SignToTextDataset(FairseqDataset):
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
        id: Optional[List[str]] = None,
        gloss_dict: Optional[Dictionary] = None,
        text_dict: Optional[Dictionary] = None,
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

        assert len(n_frames) == self.n_samples > 0
        assert text is None or len(text) == self.n_samples
        assert signer is None or len(signer) == self.n_samples
        assert id is None or len(id) == self.n_samples
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
        self.id = id
        self.shuffle = cfg.shuffle if is_train_split else False

        self.pre_tokenizer = pre_tokenizer
        self.bpe_tokenizer = bpe_tokenizer

        self.src_lens = self.get_src_lens_and_check_oov()
        self.tgt_lens = self.get_tgt_lens_and_check_oov()
        self.append_eos = append_eos

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
        logger.info(
            f"'{self.split}-gloss' has {n_oov_tokens / n_tokens * 100:.2f}% OOV"
        )
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

    def __getitem__(self, index: int) -> SignToTextDatasetItem:
        sign = self._get_source_sign(index)
        # sign = self.pack_frames(sign)

        gloss_tokenized = self.get_tokenized_gloss(index)
        fake_gloss_tokenized = self.get_fake_tokenized_gloss(index)
        gloss = self.gloss_dict.encode_line(
            gloss_tokenized, add_if_not_exist=False, append_eos=False
        ).long()
        fake_gloss = self.gloss_dict.encode_line(
            fake_gloss_tokenized, add_if_not_exist=False, append_eos=False
        ).long()
        # 在这没加eos
        tokenized = self.get_tokenized_text(index)
        text = self.text_dict.encode_line(
            tokenized, add_if_not_exist=False, append_eos=self.append_eos
        ).long()

        signer_id = None
        # if self.signer_to_id is not None:
        #     signer_id = self.signer_to_id[self.signers[index]]
        return SignToTextDatasetItem(
            index=index,
            sign=sign,
            gloss=gloss,
            fake_gloss=fake_gloss,
            text=text,
            signer_id=signer_id,
        )

    def __len__(self):
        return self.n_samples

    def collater(
        self, samples: List[SignToTextDatasetItem], return_order: bool = False
    ) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)
        frames = _collate_frames([x.sign for x in samples])
        # sort samples by descending number of frames
        n_frames = torch.tensor([x.sign.size(0) for x in samples], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

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
            "src_tokens": frames,
            "src_lengths": n_frames,
            "prev_output_tokens": prev_output_tokens,
        }
        out = {
            "id": indices,
            "net_input": net_input,
            "signer": signer,
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


class SignToTextDatasetCreator(object):
    # mandatory columns
    KEY_ID, KEY_SIGN, KEY_N_FRAMES = "name", "sign", "n_frames"
    KEY_TEXT = "text"
    # optional columns
    KEY_SIGNER, KEY_GLOSS = "signer_id", "gloss"
    KEY_FAKE_GLOSS = "fake_gloss"
    # default values
    DEFAULT_SIGNER = DEFAULT_SRC_TEXT = ""

    @classmethod
    def _load_pickle_file(cls, filename):
        with gzip.open(filename, "rb") as f:
            loaded_object = pickle.load(f)
            return loaded_object

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[Dict],
        cfg: S2TDataConfig,
        gloss_dict,
        text_dict,
        pre_tokenizer,
        bpe_tokenizer,
    ) -> SignToTextDataset:
        sign_root = Path(cfg.sign_root)
        id = [s[cls.KEY_ID] for s in samples]
        n_frames = [int(s[cls.KEY_N_FRAMES]) for s in samples]
        all_text = [s[cls.KEY_TEXT] for s in samples]
        all_gloss = [s.get(cls.KEY_GLOSS, cls.DEFAULT_SRC_TEXT) for s in samples]
        all_fake_gloss = [
            s.get(cls.KEY_FAKE_GLOSS, cls.DEFAULT_SRC_TEXT) for s in samples
        ]
        signer = [s.get(cls.KEY_SIGNER, cls.DEFAULT_SIGNER) for s in samples]
        all_sign_feature = [s[cls.KEY_SIGN] for s in samples]
        # @mark: 获取id, n_frames, text, gloss, fake_gloss, signer, sign_feature
        return SignToTextDataset(
            split_name,
            is_train_split,
            cfg,
            n_frames,
            sign=all_sign_feature,
            text=all_text,
            gloss=all_gloss,
            fake_gloss=all_fake_gloss,
            signer=signer,
            id=id,
            gloss_dict=gloss_dict,
            text_dict=text_dict,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
        )

    @classmethod
    def _load_samples_from_pickle(cls, root: str, split: str):
        # @mark: 从pickle文件中读取数据
        pickle_path = Path(root) / f"{split}"
        if not pickle_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {pickle_path}")
        samples = []
        tmp = cls._load_pickle_file(pickle_path)
        for s in tmp:
            # # data root: fake
            # sample = {
            #     "name": s["name"],
            #     "signer_id": s["signer_id"],
            #     "gloss": s["gloss"].lower(),
            #     "fake_gloss": s["fake"],
            #     "text": s["text"],
            #     "sign": s["sign"],
            #     "n_frames": len(s["sign"]),
            # }   
            # data root: pami
            sample = {
                "name": s["name"],
                "signer": s["signer"],
                "gloss": s["gloss"].lower(),
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
        is_train_split: bool,
        pre_tokenizer,
        bpe_tokenizer,
    ) -> SignToTextDataset:
        samples = cls._load_samples_from_pickle(root, split)
        return cls._from_list(
            split,
            is_train_split,
            samples,
            cfg,
            gloss_dict,
            text_dict,
            pre_tokenizer,
            bpe_tokenizer,
        )

    @classmethod
    def from_pickle(
        cls,
        root: str,
        cfg: S2TDataConfig,
        splits: str,
        gloss_dict,
        text_dict,
        pre_tokenizer,
        bpe_tokenizer,
        is_train_split: bool,
    ) -> SignToTextDataset:
        datasets = [
            cls._from_pickle(
                root,
                cfg,
                split,
                gloss_dict,
                text_dict,
                is_train_split,
                pre_tokenizer,
                bpe_tokenizer,
            )
            for split in splits.split(",")
        ]
        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
