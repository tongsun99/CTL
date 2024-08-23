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
from fairseq.data.sign.sign_dataset import (
    S2TDataConfig,
    SignToTextDataset,
    SignToTextDatasetCreator,
)
from fairseq.scoring.tokenizer import EvaluationTokenizer
from fairseq.tasks import LegacyFairseqTask, register_task
import torch

EVAL_BLEU_ORDER = 4
logger = logging.getLogger(__name__)
from fairseq.models.sign_to_text import S2TDualModel


@register_task("sign_to_gloss_and_text")
class SignToGlossAndTextTask(LegacyFairseqTask):
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

    def __init__(self, args, tgt_dict, src_dict=None):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.blank_symbol = "<ctc_blank>"
        self.data_cfg = S2TDataConfig(Path(args.data) / args.config_yaml)

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = S2TDataConfig(Path(args.data) / args.config_yaml)
        dict_path = op.join(args.data, data_cfg.vocab_filename)
        if not op.isfile(dict_path):
            raise FileNotFoundError(f"Dict not found: {dict_path}")
        src_dict = tgt_dict = Dictionary.load(dict_path)
        logger.info(
            f"text dictionary size ({data_cfg.vocab_filename}): "
            f"{len(tgt_dict):,}"
            f"gloss dictionary size ({data_cfg.vocab_filename}): "
            f"{len(src_dict):,}"
        )
        # if getattr(data_cfg, "share_src_and_tgt", False):
        #     asr_vocab_filename = data_cfg.vocab_filename
        # else:
        #     asr_vocab_filename = getattr(data_cfg, "asr_vocab_filename", None)
        # if asr_vocab_filename is not None:
        #     dict_path = op.join(args.data, asr_vocab_filename)
        #     if not op.isfile(dict_path):
        #         raise FileNotFoundError(f"Dict not found: {dict_path}")
        #     src_dict = Dictionary.load(dict_path)
        #     logger.info(
        #         f"asr dictionary size ({asr_vocab_filename}): " f"{len(src_dict):,}"
        #     )

        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')
        return cls(args, tgt_dict, src_dict)

    def build_criterion(self, args):
        from fairseq import criterions

        if self.data_cfg.prepend_tgt_lang_tag and args.ignore_prefix_size != 1:
            raise ValueError(
                'Please set "--ignore-prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        return criterions.build_criterion(args, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        # @mark 加载tokenizer
        is_train_split = split.startswith("train")
        self.pre_tokenizer = self.build_tokenizer(self.args)
        self.bpe_tokenizer = self.build_bpe(self.args)
        self.datasets[split] = SignToTextDatasetCreator.from_pickle(
            self.args.data,
            self.data_cfg,
            split,
            self.src_dict,
            self.tgt_dict,
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

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions

    def build_model(self, args):
        args.input_feat_per_channel = self.data_cfg.input_feat_per_channel
        args.input_channels = self.data_cfg.input_channels

        model = super(SignToGlossAndTextTask, self).build_model(args)

        if self.args.eval_bleu:
            detok_args = json.loads(self.args.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.args.eval_bleu_detok, **detok_args)
            )
            self.args.eval_bleu_args = '{"beam": 4, "lenpen": 0.6}'
            gen_args = json.loads(self.args.eval_bleu_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
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

            blank_idx = self.source_dictionary.index(self.blank_symbol)
            self.wer_sequence_generator = CTCDecoder(
                model, wer_gen_args, self.source_dictionary, blank_idx
            )

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

    # def get_interactive_tokens_and_lengths(self, lines, encode_fn):
    #     n_frames = [get_features_or_waveform(p).shape[0] for p in lines]
    #     return lines, n_frames
    #
    # def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
    #     return SpeechToTextDataset(
    #         "interactive", False, self.data_cfg, src_tokens, src_lengths
    #     )

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        # if self.args.eval_wer:
        #     distance, ref_length = self._inference_with_wer(self.wer_sequence_generator, sample, model)
        #     logging_output["_wer_distance"] = distance
        #     logging_output["_wer_ref_length"] = ref_length
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_wer:
            distance = sum(log.get("_wer_distance", 0) for log in logging_outputs)
            ref_length = sum(log.get("_wer_ref_length", 0) for log in logging_outputs)
            # if ref_length > 0:
            #     metrics.log_scalar("wer", 100.0 * distance / ref_length)

        if self.args.eval_bleu:

            def sum_logs(key):
                import torch

                result = sum(log.get(key, 0) for log in logging_outputs)
                if torch.is_tensor(result):
                    result = result.cpu()
                return result

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect

                    try:
                        from sacrebleu.metrics import BLEU

                        comp_bleu = BLEU.compute_bleu
                    except ImportError:
                        # compatibility API for sacrebleu 1.x
                        import sacrebleu

                        comp_bleu = sacrebleu.compute_bleu

                    fn_sig = inspect.getfullargspec(comp_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = comp_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth,
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.args.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])

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

    # def valid_step(self, sample, model, criterion):
    #     loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
    #
    #     if self.args.eval_bleu:
    #         hyps, refs = self._inference(self.sequence_generator, sample, model)
    #         bleu = self._cal_bleu(hyps, refs)
    #         logging_output["_bleu_sys_len"] = bleu.sys_len
    #         logging_output["_bleu_ref_len"] = bleu.ref_len
    #         # we split counts into separate entries so that they can be
    #         # summed efficiently across workers using fast-stat-sync
    #         assert len(bleu.counts) == EVAL_BLEU_ORDER
    #         for i in range(EVAL_BLEU_ORDER):
    #             logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
    #             logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
    #
    #     if self.args.eval_wer:
    #         hyps, refs = self.wer_inference(self.wer_sequence_generator, sample, model)
    #         distance, ref_length = self._cal_wer(hyps, refs)
    #         logging_output["_wer_distance"] = distance
    #         logging_output["_wer_ref_length"] = ref_length
    #
    #     return loss, sample_size, logging_output
    #
    # def reduce_metrics(self, logging_outputs, criterion):
    #     super().reduce_metrics(logging_outputs, criterion)
    #     if self.args.eval_wer:
    #         distance = sum(log.get("distance", 0) for log in logging_outputs)
    #         ref_length = sum(log.get("ref_length", 0) for log in logging_outputs)
    #         if ref_length > 0:
    #             metrics.log_scalar("wer", 100.0 * distance / ref_length)
    #
    #     if self.args.eval_bleu:
    #
    #         def sum_logs(key):
    #             import torch
    #             result = sum(log.get(key, 0) for log in logging_outputs)
    #             if torch.is_tensor(result):
    #                 result = result.cpu()
    #             return result
    #
    #         counts, totals = [], []
    #         for i in range(EVAL_BLEU_ORDER):
    #             counts.append(sum_logs("_bleu_counts_" + str(i)))
    #             totals.append(sum_logs("_bleu_totals_" + str(i)))
    #
    #         if max(totals) > 0:
    #             # log counts as numpy arrays -- log_scalar will sum them correctly
    #             metrics.log_scalar("_bleu_counts", np.array(counts))
    #             metrics.log_scalar("_bleu_totals", np.array(totals))
    #             metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
    #             metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))
    #
    #             def compute_bleu(meters):
    #                 import inspect
    #                 import sacrebleu
    #
    #                 fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
    #                 if "smooth_method" in fn_sig:
    #                     smooth = {"smooth_method": "exp"}
    #                 else:
    #                     smooth = {"smooth": "exp"}
    #                 bleu = sacrebleu.compute_bleu(
    #                     correct=meters["_bleu_counts"].sum,
    #                     total=meters["_bleu_totals"].sum,
    #                     sys_len=meters["_bleu_sys_len"].sum,
    #                     ref_len=meters["_bleu_ref_len"].sum,
    #                     **smooth
    #                 )
    #                 return round(bleu.score, 2)
    #
    #             metrics.log_derived("bleu", compute_bleu)

    # def _inference(self, generator, sample, model):
    #     def decode(toks, escape_unk=False):
    #         s = self.tgt_dict.string(
    #             toks.int().cpu(),
    #             self.args.eval_bleu_remove_bpe,
    #             # The default unknown string in fairseq is `<unk>`, but
    #             # this is tokenized by sacrebleu as `< unk >`, inflating
    #             # BLEU scores. Instead, we use a somewhat more verbose
    #             # alternative that is unlikely to appear in the real
    #             # reference, but doesn't get split into multiple tokens.
    #             unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
    #         )
    #         if self.tokenizer:
    #             s = self.tokenizer.decode(s)
    #         return s
    #
    #     gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
    #     hyps, refs = [], []
    #     for i in range(len(gen_out)):
    #         hyps.append(decode(gen_out[i][0]["tokens"]))
    #         refs.append(
    #             decode(
    #                 utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
    #                 escape_unk=True,  # don't count <unk> as matches to the hypo
    #             )
    #         )
    #     return hyps, refs

    # def _cal_bleu(self, hyps, refs):
    #     import sacrebleu
    #
    #     if self.args.eval_bleu_print_samples:
    #         logger.info("text hypothesis: " + hyps[0])
    #         logger.info("text reference: " + refs[0])
    #     if self.args.eval_tokenized_bleu:
    #         return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
    #     else:
    #         return sacrebleu.corpus_bleu(hyps, [refs])
    #
    # def _cal_wer(self, hyps, refs):
    #     distance = 0
    #     ref_length = 0
    #     if self.args.eval_wer_print_samples:
    #         logger.info("gloss hypothesis: " + hyps[0])
    #         logger.info("gloss reference: " + refs[0])
    #     for hyp, ref in zip(hyps, refs):
    #         ref = ref.replace("<<unk>>", "@")
    #         hyp = hyp.replace("<<unk>>", "@")
    #         ref_items = self.wer_tokenizer.tokenize(ref).split()
    #         hyp_items = self.wer_tokenizer.tokenize(hyp).split()
    #         distance += self.ed.eval(ref_items, hyp_items)
    #         ref_length += len(ref_items)
    #     return distance, ref_length
