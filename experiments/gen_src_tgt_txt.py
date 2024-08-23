import argparse
import gzip
import pickle
import numpy as np
import torch
import os


def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object


def main(**kwargs):
    data_path = kwargs["data_path"]
    src_txt_path = kwargs["src_txt_path"]
    tgt_txt_path = kwargs["tgt_txt_path"]
    sample_rate = kwargs["sample_rate"]

    tmp = load_dataset_file(data_path)

    samples = {}
    for index, s in enumerate(tmp):
        seq_id = s["name"]
        if seq_id in samples:
            assert samples[seq_id]["name"] == s["name"]
            assert samples[seq_id]["text"] == s["text"]
        else:
            samples[seq_id] = {
                "name": s["name"],
                "text": s["text"],
            }
            with open(src_txt_path, "a") as f:
                f.write(data_path + ' ' + s["name"] + ' ' + str(index) + ' ' + str(sample_rate) + "\n")

            with open(tgt_txt_path, "a") as f:
                f.write(s["text"] + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_bin', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--sample_rate', type=int)

    args = parser.parse_args()

    main(
        data_path=os.path.join(args.data_bin, args.split),
        src_txt_path=os.path.join(args.data_bin, f'src_{args.split}.txt'),
        tgt_txt_path=os.path.join(args.data_bin, f'tgt_{args.split}.txt'),
        sample_rate=args.sample_rate,
    )
