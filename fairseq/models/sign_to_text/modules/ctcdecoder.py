import logging
from typing import Dict, Optional, List

import torch

from fairseq import utils

from torch import Tensor
from itertools import groupby
logger = logging.getLogger(__name__)
#
# class CTCDecoder(object):
#
#     def __init__(self, models, args, dictionary, blank_idx):
#         self.dict = dictionary
#         self.vocab_size = len(dictionary)
#
#         self.blank = blank_idx
#         self.pad = dictionary.pad()
#         self.unk = dictionary.unk()
#         self.eos = dictionary.eos()
#
#         self.ctc_self_ensemble = getattr(args, "ctc_self_ensemble", False)
#         self.ctc_inter_logit = getattr(args, "ctc_inter_logit", 0)
#         assert not (self.ctc_self_ensemble is True and self.ctc_inter_logit is True), \
#             "Self ensemble and inference by intermediate logit can not be True at the same time."
#
#         if self.ctc_self_ensemble:
#             logger.info("Using self ensemble for CTC inference")
#         if self.ctc_inter_logit != 0:
#             logger.info("Using intermediate logit %d for CTC inference" % self.ctc_inter_logit)
#
#         self.vocab_size = len(dictionary)
#         #self.beam_size = args.beam
#         self.beam_size = 5
#         # the max beam size is the dictionary size - 1, since we never select pad
#         self.beam_size = min(self.beam_size, self.vocab_size - 1)
#
#         # from fairseq.sequence_generator import EnsembleModel
#         # if isinstance(models, EnsembleModel):
#         #     self.model = models
#         # else:
#         #     self.model = EnsembleModel(models)
#         self.model = models
#         self.model.eval()
#
#         self.lm_model = getattr(args, "kenlm_model", None)
#         self.lm_weight = getattr(args, "lm_weight", 0)
#         if self.lm_model is not None:
#             self.lm_model.eval()
#
#         self.infer = "greedy"
#         if self.beam_size > 1:
#             try:
#                 from torchaudio.models.decoder import ctc_decoder
#                 self.infer = "beam"
#                 self.ctc_decoder = ctc_decoder(
#                     lexicon=None,
#                     tokens=dictionary.symbols,
#                     beam_size=self.beam_size,
#                     nbest = 5,
#                     blank_token="<ctc_blank>",
#                     sil_token="<ctc_sil>",
#                     unk_word="<unk>",
#                 )
#             except ImportError:
#                 logger.warning("Cannot import the CTCBeamDecoder library. We use the greedy search for CTC decoding.")
#
#     def generate(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs):
#
#         net_input = sample["net_input"]
#
#         # bsz: total number of sentences in beam
#         # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
#         src_tokens = net_input["src_tokens"]
#         src_lengths = net_input["src_lengths"]
#         bsz, src_len = src_tokens.size()[:2]
#
#         encoder_outs = self.model.encoder(src_tokens=src_tokens,
#                                   src_lengths=src_lengths)
#
#         if "target_ctc_logit" in encoder_outs and encoder_outs["target_ctc_logit"][0]!=None:
#             ctc_logit = encoder_outs["target_ctc_logit"][0].transpose(0, 1)
#         else:
#             ctc_logit = encoder_outs["ctc_logit"][0].transpose(0, 1)
#         inter_logits = encoder_outs.get("interleaved_ctc_logits", [])
#         inter_logits_num = len(inter_logits)
#
#         if self.ctc_inter_logit != 0:
#             if inter_logits_num != 0:
#                 assert self.ctc_inter_logit <= inter_logits_num
#                 ctc_logit = inter_logits[-self.ctc_inter_logit].transpose(0, 1)
#
#         logit_length = (~encoder_outs["encoder_padding_mask"][0]).long().sum(-1)
#         finalized = []
#         if self.infer == "beam":
#
#             assert len(ctc_logit) == len(logit_length)
#
#             decoded_strings: List[str] = list()
#             # out = self.ctc_decoder(ctc_logit.detach().cpu(), logit_length.detach().cpu())
#             # for hypotheses in out:
#             #     # hypotheses [beam size - 1]
#             #     tokens_ids = hypotheses[-1].tokens.tolist()
#             #     tokens_ids = torch.Tensor(list(filter(lambda a: a != self.vocab_size - 1, tokens_ids)))
#             #
#             #     #str_decoded = self._string_encoder.inverse_transform(tokens_ids)
#             #     decoded_strings.append(tokens_ids)
#             #
#             # assert len(decoded_strings) == len(logit_length)
#             # return decoded_strings
#             #
#             #
#             out_lens = self.ctc_decoder(
#                 utils.softmax(ctc_logit.detach().cpu(), -1)
#             )
#
#             for idx in range(bsz):
#                 hypos = []
#                 #for beam_idx in range(beam_size):
#                 for beam_idx in range(1):
#                     hypo = dict()
#                     length = out_lens[idx][beam_idx]
#                     scores = beam_scores[idx, beam_idx]
#
#                     hypo["tokens"] = beam_results[idx, beam_idx, : length]
#                     hypo["score"] = scores
#                     hypo["attention"] = None
#                     hypo["alignment"] = None
#                     hypo["positional_scores"] = torch.Tensor([scores / length] * length)
#                     hypos.append(hypo)
#                 finalized.append(hypos)
#
#         else:
#             ctc_probs = utils.log_softmax(ctc_logit, -1)
#             if self.ctc_self_ensemble:
#                 if inter_logits_num != 0:
#                     for i in range(inter_logits_num):
#                         inter_logits_prob = utils.log_softmax(inter_logits[i].transpose(0, 1), -1)
#                         ctc_probs += inter_logits_prob
#
#             topk_prob, topk_index = ctc_probs.topk(1, dim=2)
#
#             topk_prob = topk_prob.squeeze(-1)
#             topk_index = topk_index.squeeze(-1)
#
#             real_indexs = topk_index.masked_fill(encoder_outs["encoder_padding_mask"][0], self.blank).cpu()
#             real_probs = topk_prob.masked_fill(topk_index == self.blank, self.blank)
#             scores = -real_probs.sum(-1, keepdim=True).cpu()
#
#             for idx in range(bsz):
#                 hypos = []
#                 hypo = dict()
#
#                 hyp = real_indexs[idx].unique_consecutive()
#                 hyp = hyp[hyp != self.blank]
#                 length = len(hyp)
#
#                 hypo["tokens"] = hyp
#                 hypo["score"] = scores[idx]
#                 hypo["attention"] = None
#                 hypo["alignment"] = None
#                 hypo["positional_scores"] = torch.Tensor([hypo["score"] / length] * length)
#                 hypos.append(hypo)
#                 finalized.append(hypos)
#
#         return finalized

class CTCDecoder(object):

    def __init__(self, models, args, dictionary, blank_idx):
        self.dict = dictionary
        self.vocab_size = len(dictionary)

        self.blank = blank_idx
        self.pad = dictionary.pad()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()

        self.ctc_self_ensemble = getattr(args, "ctc_self_ensemble", False)
        self.ctc_inter_logit = getattr(args, "ctc_inter_logit", 0)
        assert not (self.ctc_self_ensemble is True and self.ctc_inter_logit is True), \
            "Self ensemble and inference by intermediate logit can not be True at the same time."

        if self.ctc_self_ensemble:
            logger.info("Using self ensemble for CTC inference")
        if self.ctc_inter_logit != 0:
            logger.info("Using intermediate logit %d for CTC inference" % self.ctc_inter_logit)

        self.vocab_size = len(dictionary)
        #self.beam_size = args.beam
        self.beam_size = 5
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(self.beam_size, self.vocab_size - 1)

        # from fairseq.sequence_generator import EnsembleModel
        # if isinstance(models, EnsembleModel):
        #     self.model = models
        # else:
        # self.model = EnsembleModel(models)
        # self.model = models[0]
        self.model = models
        self.model.eval()

        self.lm_model = getattr(args, "kenlm_model", None)
        self.lm_weight = getattr(args, "lm_weight", 0)
        if self.lm_model is not None:
            self.lm_model.eval()

        self.infer = "greedy"
        if self.beam_size > 1:
            try:
                from ctcdecode import CTCBeamDecoder
                self.infer = "beam"
                self.ctc_decoder = CTCBeamDecoder(
                    dictionary.symbols,
                    model_path=self.lm_model,
                    alpha=self.lm_weight,
                    beta=0,
                    cutoff_top_n=40,
                    cutoff_prob=1.0,
                    beam_width=self.beam_size,
                    num_processes=20,
                    blank_id=self.blank,
                    log_probs_input=False
                )
            except ImportError:
                logger.warning("Cannot import the CTCBeamDecoder library. We use the greedy search for CTC decoding.")

    def generate(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs):

        net_input = sample["net_input"]

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
        src_tokens = net_input["src_tokens"]
        src_lengths = net_input["src_lengths"]
        bsz, src_len = src_tokens.size()[:2]

        encoder_outs = self.model(src_tokens=src_tokens,
                                  src_lengths=src_lengths)

        if "target_ctc_logit" in encoder_outs:
            ctc_logit = encoder_outs["target_ctc_logit"][0].transpose(0, 1)
        else:
            ctc_logit = encoder_outs["ctc_logit"][0].transpose(0, 1)
        inter_logits = encoder_outs.get("interleaved_ctc_logits", [])
        inter_logits_num = len(inter_logits)

        if self.ctc_inter_logit != 0:
            if inter_logits_num != 0:
                assert self.ctc_inter_logit <= inter_logits_num
                ctc_logit = inter_logits[-self.ctc_inter_logit].transpose(0, 1)

        logit_length = (~encoder_outs["encoder_padding_mask"][0]).long().sum(-1)
        finalized = []
        if self.infer == "beam":
            beam_results, beam_scores, time_steps, out_lens = self.ctc_decoder.decode(
                utils.softmax(ctc_logit, -1), logit_length
            )

            for idx in range(bsz):
                hypos = []
                #for beam_idx in range(self.beam_size):
                for beam_idx in range(1):
                    hypo = dict()
                    length = out_lens[idx][beam_idx]
                    scores = beam_scores[idx, beam_idx]

                    hypo["tokens"] = beam_results[idx, beam_idx, : length]
                    hypo["score"] = scores
                    hypo["attention"] = None
                    hypo["alignment"] = None
                    hypo["positional_scores"] = torch.Tensor([scores / length] * length)
                    hypos.append(hypo)
                finalized.append(hypos)

        # elif self.infer == "greedy":
        else:
            ctc_probs = utils.log_softmax(ctc_logit, -1)
            if self.ctc_self_ensemble:
                if inter_logits_num != 0:
                    for i in range(inter_logits_num):
                        inter_logits_prob = utils.log_softmax(inter_logits[i].transpose(0, 1), -1)
                        ctc_probs += inter_logits_prob

            topk_prob, topk_index = ctc_probs.topk(1, dim=2)

            topk_prob = topk_prob.squeeze(-1)
            topk_index = topk_index.squeeze(-1)

            real_indexs = topk_index.masked_fill(encoder_outs["encoder_padding_mask"][0], self.blank).cpu()
            real_probs = topk_prob.masked_fill(topk_index == self.blank, self.blank)
            scores = -real_probs.sum(-1, keepdim=True).cpu()

            for idx in range(bsz):
                hypos = []
                hypo = dict()

                hyp = real_indexs[idx].unique_consecutive()
                hyp = hyp[hyp != self.blank]
                length = len(hyp)

                hypo["tokens"] = hyp
                hypo["score"] = scores[idx]
                hypo["attention"] = None
                hypo["alignment"] = None
                hypo["positional_scores"] = torch.Tensor([hypo["score"] / length] * length)
                hypos.append(hypo)
                finalized.append(hypos)

        return finalized

class Decode(object):
    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0):
        self.i2g_dict = dict((v, k) for k, v in gloss_dict.indices.items())
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id
        vocab = [chr(x) for x in range(20000, 20000 + num_classes)]
        from ctcdecode import CTCBeamDecoder
        self.ctc_decoder = CTCBeamDecoder(vocab, beam_width=10, blank_id=blank_id,
                                                    num_processes=10)

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        if self.search_mode == "max":
            return self.MaxDecode(nn_output, vid_lgt)
        else:
            return self.BeamSearch(nn_output, vid_lgt, probs)

    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        '''
        CTCBeamDecoder Shape:
                - Input:  nn_output (B, T, N), which should be passed through a softmax layer
                - Output: beam_resuls (B, N_beams, T), int, need to be decoded by i2g_dict
                          beam_scores (B, N_beams), p=1/np.exp(beam_score)
                          timesteps (B, N_beams)
                          out_lens (B, N_beams)
        '''
        if not probs:
            nn_output = nn_output.softmax(-1).cpu()
        vid_lgt = vid_lgt.cpu()
        beam_result, beam_scores, timesteps, out_seq_len = self.ctc_decoder.decode(nn_output, vid_lgt)
        ret_list = []
        for batch_idx in range(len(nn_output)):
            first_result = beam_result[batch_idx][0][:out_seq_len[batch_idx][0]]
            if len(first_result) != 0:
                first_result = torch.stack([x[0] for x in groupby(first_result)])
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(first_result)])
        return ret_list

    def MaxDecode(self, nn_output, vid_lgt):
        index_list = torch.argmax(nn_output, axis=2)
        batchsize, lgt = index_list.shape
        ret_list = []
        for batch_idx in range(batchsize):
            group_result = [x[0] for x in groupby(index_list[batch_idx][:vid_lgt[batch_idx]])]
            filtered = [*filter(lambda x: x != self.blank_id, group_result)]
            if len(filtered) > 0:
                max_result = torch.stack(filtered)
                max_result = [x[0] for x in groupby(max_result)]
            else:
                max_result = filtered
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(max_result)])
        return ret_list
