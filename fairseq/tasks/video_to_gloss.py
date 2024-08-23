# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import json
import os.path as op
import numpy as np
from argparse import Namespace
from pathlib import Path
from fairseq import metrics, utils
from fairseq.data import Dictionary, encoders
from fairseq.data.sign import (
    S2TDataConfig,
    SignToTextDataset,
    SignToTextDatasetCreator,
    VideoToTextDataset,
    VideoToTextDatasetCreator,
)
from fairseq.scoring.tokenizer import EvaluationTokenizer
from fairseq.tasks import LegacyFairseqTask, register_task
import torch
EVAL_BLEU_ORDER = 4
logger = logging.getLogger(__name__)
from fairseq.models.sign_to_text import S2TDualModel

@register_task("video_to_gloss")
class VideoToGlossTask(LegacyFairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument("data", help="manifest root path")
        parser.add_argument(
            "--config-yaml",
            type=str,
            default="config.yaml",
            help="Configuration YAML filename (under manifest root)",
        )
        parser.add_argument(
            "--max-source-positions",
            default=6000,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-target-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )
        parser.add_argument(
            "--use-aligned-text",
            default=False,
            action="store_true",
            help="use aligned text for loss",
        )

        # options for reporting BLEU during validation
        parser.add_argument(
            "--eval-bleu",
            default=False,
            action="store_true",
            help="evaluation with BLEU scores",
        )
        parser.add_argument(
            "--eval-bleu-args",
            default="{}",
            type=str,
            help='generation args for BLUE scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string',
        )
        parser.add_argument(
            "--eval-bleu-detok",
            default="space",
            type=str,
            help="detokenize before computing BLEU (e.g., 'moses'); required if using --eval-bleu; "
                 "use 'space' to disable detokenization; see fairseq.data.encoders for other options",
        )

        parser.add_argument(
            "--eval-bleu-detok-args",
            default="{}",
            type=str,
            help="args for building the tokenizer, if needed, as JSON string",
        )
        parser.add_argument(
            "--eval-tokenized-bleu",
            default=False,
            action="store_true",
            help="compute tokenized BLEU instead of sacrebleu",
        )
        parser.add_argument(
            "--eval-bleu-remove-bpe",
            nargs="?",
            const="@@ ",
            default=None,
            help="remove BPE before computing BLEU",
        )
        parser.add_argument(
            "--eval-bleu-print-samples",
            default=False,
            action="store_true",
            help="print sample generations during validation",
        )

        # options for reporting WER during validation
        parser.add_argument(
            "--eval-wer",
            default=False,
            action="store_true",
            help="evaluation with WER scores",
        )
        parser.add_argument(
            "--eval-wer-remove-bpe",
            nargs="?",
            const="@@ ",
            default=None,
            help="remove BPE before computing WER",
        )
        parser.add_argument(
            "--eval-wer-args",
            default="{}",
            type=str,
            help='generation args for WER scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string',
        )
        parser.add_argument(
            "--eval-wer-tok-args",
            default="{}",
            type=str,
            help='tokenizer args for WER scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string',
        )
        parser.add_argument(
            "--eval-wer-detok-args",
            default="{}",
            type=str,
            help="args for building the tokenizer, if needed, as JSON string",
        )
        parser.add_argument(
            "--eval-wer-print-samples",
            default=False,
            action="store_true",
            help="print sample gloss generations during validation",
        )

    def __init__(self, args, tgt_dict, src_dict,vac_dict):
        super().__init__(args)
        self.tgt_dict = tgt_dict
        self.src_dict = src_dict
        self.vac_dict = vac_dict
        self.data_cfg = S2TDataConfig(Path(args.data) / args.config_yaml)

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = S2TDataConfig(Path(args.data) / args.config_yaml)
        dict_path = op.join(args.data, data_cfg.vocab_filename)
        if not op.isfile(dict_path):
            raise FileNotFoundError(f"Dict not found: {dict_path}")
        src_dict = tgt_dict = Dictionary.load(dict_path)
        #if(args.ctc_weight > 0):
        #src_dict.add_symbol("<ctc_blank>")
        logger.info(
            f"text dictionary size ({data_cfg.vocab_filename}): " f"{len(tgt_dict):,}"
            f"gloss dictionary size ({data_cfg.vocab_filename}): " f"{len(src_dict):,}"
        )

        vac_vocab_filename = getattr(data_cfg, "vac_vocab_filename", None)
        dict_path = op.join(args.data, vac_vocab_filename)
        if not op.isfile(dict_path):
            raise FileNotFoundError(f"Dict not found: {dict_path}")
        vac_dict = Dictionary.load(dict_path)
        logger.info(
            f"vac dictionary size ({vac_vocab_filename}): " f"{len(vac_dict):,}"
        )

        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')
        return cls(args, tgt_dict, src_dict, vac_dict)

    def build_criterion(self, args):
        from fairseq import criterions
        return criterions.build_criterion(args, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        self.pre_tokenizer = self.build_tokenizer(self.args)
        self.bpe_tokenizer = self.build_bpe(self.args)
        self.datasets[split] = VideoToTextDatasetCreator.from_pickle(
            self.args.data,
            self.data_cfg,
            split,
            self.src_dict,
            self.tgt_dict,
            self.vac_dict,
            self.pre_tokenizer,
            self.bpe_tokenizer,
            is_train_split=is_train_split,
        )

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def vac_dictionary(self):
        return self.vac_dict

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions

    def build_model(self, args):
        args.input_feat_per_channel = self.data_cfg.input_feat_per_channel
        args.input_channels = self.data_cfg.input_channels

        model = super(VideoToGlossTask, self).build_model(args)

        if self.args.eval_wer:
            try:
                import editdistance as ed
            except ImportError:
                raise ImportError("Please install editdistance to use WER scorer")
            self.ed = ed

            detok_args = json.loads(self.args.eval_wer_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.args.eval_bleu_detok, **detok_args)
            )

            wer_tok_args = json.loads(self.args.eval_wer_tok_args)
            self.wer_tokenizer = EvaluationTokenizer(
                tokenizer_type=wer_tok_args.get("wer_tokenizer", "none"),
                lowercase=wer_tok_args.get("wer_lowercase", False),
                punctuation_removal=wer_tok_args.get("wer_remove_punct", False),
                character_tokenization=wer_tok_args.get("wer_char_level", False),
            )
            wer_gen_args = json.loads(self.args.eval_wer_args)
            from fairseq.models.sign_to_text.modules.ctcdecoder import CTCDecoder
            blank_idx = self.vac_dictionary.pad_index
            self.wer_sequence_generator = CTCDecoder(model, wer_gen_args,
                             self.vac_dictionary,
                             blank_idx)

        return model

    def build_tokenizer(self, args):
        logger.info(f"pre-tokenizer: {self.data_cfg.pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.pre_tokenizer))

    def build_bpe(self, args):
        logger.info(f"tokenizer: {self.data_cfg.bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.bpe_tokenizer))

    def build_src_bpe(self, args):
        logger.info(f"src tokenizer: {self.data_cfg.src_bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.src_bpe_tokenizer))

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_wer:
            distance, ref_length = self._inference_with_wer(self.wer_sequence_generator, sample, model)
            logging_output["_wer_distance"] = distance
            logging_output["_wer_ref_length"] = ref_length
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_wer:
            distance = sum(log.get("_wer_distance", 0) for log in logging_outputs)
            ref_length = sum(log.get("_wer_ref_length", 0) for log in logging_outputs)
            if ref_length > 0:
                metrics.log_scalar("wer", 100.0 * distance / ref_length)
    def _inference_with_wer(self, generator, sample, model):

        # def decode(toks):
        #     s = self.target_dictionary.string(
        #         toks.int().cpu(),
        #         self.cfg.eval_wer_remove_bpe,
        #         escape_unk=True,
        #     )
        #     if self.tokenizer:
        #         s = self.tokenizer.decode(s)
        #     return s
        #
        # num_word_errors, num_char_errors = 0, 0
        # num_chars, num_words = 0, 0
        # gen_out = self.inference_step(generator, [model], sample, None)
        # for i in range(len(gen_out)):
        #     hyp = decode(gen_out[i][0]["tokens"])
        #     ref = decode(
        #         utils.strip_pad(sample["target"][i], self.target_dictionary.pad()),
        #     )
        #     num_char_errors += self.ed.eval(hyp, ref)
        #     num_chars += len(ref)
        #     hyp_words = hyp.split()
        #     ref_words = ref.split()
        #     num_word_errors += self.ed.eval(hyp_words, ref_words)
        #     num_words += len(ref_words)



        def decode(toks, escape_unk=False):
            s = self.src_dict.string(
                toks.int().cpu(),
                self.args.eval_wer_remove_bpe,
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["gloss"]["tokens"][i], self.src_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )

        distance = 0
        ref_length = 0
        if self.args.eval_wer_print_samples:
            logger.info("gloss hypothesis: " + hyps[0])
            logger.info("gloss reference: " + refs[0])
        for hyp, ref in zip(hyps, refs):
            # ref = ref.replace("<<unk>>", "@")
            # hyp = hyp.replace("<<unk>>", "@")
            # ref_items = self.wer_tokenizer.tokenize(ref).split()
            # hyp_items = self.wer_tokenizer.tokenize(hyp).split()
            distance += 0
            ref_length += len(ref.split())
        return distance, ref_length