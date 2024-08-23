import csv
import argparse
import os
from pathlib import Path
import zipfile
from functools import reduce
from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional, Union
import io
import numpy as np
import pandas as pd
import sentencepiece as sp
from fairseq.data.audio.audio_utils import (
    convert_waveform, _get_kaldi_fbank, _get_torchaudio_fbank, is_npy_data,
    is_sf_audio_data
)
import torch
import soundfile as sf
from tqdm import tqdm


UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 3
BOS_TOKEN, BOS_TOKEN_ID = "<s>", 0
EOS_TOKEN, EOS_TOKEN_ID = "</s>", 2
PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 1


def gen_vocab(
    input_path: Path, output_path_prefix: Path, model_type="bpe",
    vocab_size=1000, special_symbols: Optional[List[str]] = None
):
    # Train SentencePiece Model
    arguments = [
        f"--input={input_path.as_posix()}",
        f"--model_prefix={output_path_prefix.as_posix()}",
        f"--model_type={model_type}",
        f"--vocab_size={vocab_size}",
        "--character_coverage=1.0",
        f"--num_threads={cpu_count()}",
        f"--unk_id={UNK_TOKEN_ID}",
        f"--bos_id={BOS_TOKEN_ID}",
        f"--eos_id={EOS_TOKEN_ID}",
        f"--pad_id={PAD_TOKEN_ID}",
    ]
    if special_symbols is not None:
        _special_symbols = ",".join(special_symbols)
        arguments.append(f"--user_defined_symbols={_special_symbols}")
    sp.SentencePieceTrainer.Train(" ".join(arguments))
    # Export fairseq dictionary
    spm = sp.SentencePieceProcessor()
    spm.Load(output_path_prefix.as_posix() + ".model")
    vocab = {i: spm.IdToPiece(i) for i in range(spm.GetPieceSize())}
    assert (
        vocab.get(UNK_TOKEN_ID) == UNK_TOKEN
        and vocab.get(PAD_TOKEN_ID) == PAD_TOKEN
        and vocab.get(BOS_TOKEN_ID) == BOS_TOKEN
        and vocab.get(EOS_TOKEN_ID) == EOS_TOKEN
    )
    vocab = {
        i: s
        for i, s in vocab.items()
        if s not in {UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN}
    }
    with open(output_path_prefix.as_posix() + ".vocab", "w") as f_out:
        for _, s in sorted(vocab.items(), key=lambda x: x[0]):
            f_out.write(f"{s} 1\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_bin', type=str)
    parser.add_argument('--vocab_size', type=int)

    args = parser.parse_args()

    data_path = os.path.join(args.data_bin, "tgt_train.txt")
    piece_path = os.path.join(args.data_bin, f"mypiece_{args.vocab_size}")

    gen_vocab(
        Path(data_path),
        Path(piece_path),
        model_type='bpe',
        vocab_size=args.vocab_size,
        special_symbols=None,
    )
