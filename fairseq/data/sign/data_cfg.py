# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from argparse import Namespace
from pathlib import Path
from typing import Dict, Optional

from fairseq.data import Dictionary


def get_config_from_yaml(yaml_path: Path):
    try:
        import yaml
    except ImportError:
        print("Please install PyYAML: pip install PyYAML")
    config = {}
    if yaml_path.is_file():
        try:
            with open(yaml_path) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        except Exception as e:
            raise Exception(f"Failed to load config from {yaml_path.as_posix()}: {e}")
    else:
        raise FileNotFoundError(f"{yaml_path.as_posix()} not found")

    return config


class S2TDataConfig(object):
    """Wrapper class for data config YAML"""

    def __init__(self, yaml_path: Path):
        self.config = get_config_from_yaml(yaml_path)
        self.root = yaml_path.parent

    def _auto_convert_to_abs_path(self, x):
        if isinstance(x, str):
            if not Path(x).exists() and (self.root / x).exists():
                return (self.root / x).as_posix()
        elif isinstance(x, dict):
            return {k: self._auto_convert_to_abs_path(v) for k, v in x.items()}
        return x

    @property
    def vocab_filename(self):
        """fairseq vocabulary file under data root"""
        return self.config.get("vocab_filename", "dict.txt")

    @property
    def vac_vocab_filename(self):
        """fairseq vocabulary file under data root"""
        return self.config.get("vac_vocab_filename", "dict.txt")
    @property
    def input_feat_per_channel(self):
        """The dimension of input features (per audio channel)"""
        return self.config.get("input_feat_per_channel", 80)

    @property
    def input_channels(self):
        """The number of channels in the input audio"""
        return self.config.get("input_channels", 1)
    @property
    def shuffle(self) -> bool:
        """Shuffle dataset samples before batching"""
        return self.config.get("shuffle", False)

    @property
    def pre_tokenizer(self) -> Dict:
        """Pre-tokenizer to apply before subword tokenization. Returning
        a dictionary with `tokenizer` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        tokenizer = self.config.get("pre_tokenizer", {"tokenizer": None})
        return self._auto_convert_to_abs_path(tokenizer)

    @property
    def bpe_tokenizer(self) -> Dict:
        """Subword tokenizer to apply after pre-tokenization. Returning
        a dictionary with `bpe` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        tokenizer = self.config.get("bpe_tokenizer", {"bpe": None})
        return self._auto_convert_to_abs_path(tokenizer)

    @property
    def prepend_tgt_lang_tag(self) -> bool:
        """Prepend target lang ID token as the target BOS (e.g. for to-many
        multilingual setting). During inference, this requires `--prefix-size 1`
        to force BOS to be lang ID token."""
        return self.config.get("prepend_tgt_lang_tag", False)

    @property
    def prepend_bos_and_append_tgt_lang_tag(self) -> bool:
        """Prepend BOS and append target lang ID token to the target (e.g. mBART with language token pretraining)."""
        return self.config.get("prepend_bos_and_append_tgt_lang_tag", False)

    @property
    def use_audio_input(self):
        """Needed by the dataset loader to see if the model requires
        raw audio as inputs."""
        return self.config.get("use_audio_input", False)


    @property
    def sign_root(self):
        """sign paths in the manifest TSV can be relative and this provides
        the root path. Set this to empty string when using absolute paths."""
        return self.config.get("sign_root", "")

