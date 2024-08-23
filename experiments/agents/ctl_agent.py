import math
import os
import json
import numpy as np
import torch
import yaml
import json
import pickle
import logging
import copy

from fairseq import checkpoint_utils, tasks
from fairseq.file_io import PathManager

try:
    from simuleval import READ_ACTION, WRITE_ACTION, DEFAULT_EOS
    from simuleval.agents import SignAgent
    from simuleval.states import ListEntry, SignStates
except ImportError:
    print("Please install simuleval 'pip install simuleval'")

SHIFT_SIZE = 10
WINDOW_SIZE = 25
SAMPLE_RATE = 25
FEATURE_DIM = 1024
BOW_PREFIX = "\u2581"
SIGN_SEGMENT_SIZE = int(os.environ.get("SIGN_SEGMENT_SIZE", 40))

def longest_common_prefix(tensor1, tensor2):
    tensor1 = tensor1[0]
    tensor2 = tensor2[0]
    min_length = min(len(tensor1), len(tensor2))
    prefix_length = 0

    for i in range(min_length):
        if tensor1[i] == tensor2[i]:
            prefix_length += 1
        else:
            break

    return prefix_length


class TensorListEntry(ListEntry):
    """
    Data structure to store a list of tensor.
    """

    def append(self, value):

        if len(self.value) == 0:
            self.value = value
            return

        self.value = torch.cat([self.value] + [value], dim=0)

    def info(self):
        return {
            "type": str(self.new_value_type),
            "length": self.__len__(),
            "value": "" if type(self.value) is list else self.value.size(),
        }


class CTLAgent(SignAgent):

    sign_segment_size = SIGN_SEGMENT_SIZE  # in ms

    def __init__(self, args):
        super().__init__(args)

        self.eos = DEFAULT_EOS

        self.gpu = getattr(args, "gpu", False)

        self.args = args

        self.load_model_vocab(args)

        self.max_len = args.max_len

        self.force_finish = args.force_finish

        torch.set_grad_enabled(False)
    
    def build_states(self, args, client, sentence_id):
        # Initialize states here, for example add customized entry to states
        # This function will be called at beginning of every new sentence
        states = SignStates(args, client, sentence_id, self)
        self.initialize_states(states)

        states.prev_cur_str = ""

        return states

    def to_device(self, tensor):
        if self.gpu:
            return tensor.cuda()
        else:
            return tensor.cpu()

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--model-path', type=str, required=True,
                            help='path to your pretrained model.')
        parser.add_argument("--data-bin", type=str, required=True,
                            help="Path of data binary")
        parser.add_argument("--config", type=str, default=None,
                            help="Path to config yaml file")
        parser.add_argument("--tgt-splitter-type", type=str, default="SentencePiece",
                            help="Subword splitter type for target text")
        parser.add_argument("--tgt-splitter-path", type=str, default=None,
                            help="Subword splitter model path for target text")
        parser.add_argument("--max-len", type=int, default=200,
                            help="Max length of translation")
        parser.add_argument("--force-finish", default=False, action="store_true",
                            help="Force the model to finish the hypothsis if the source is not finished")
        parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE,
                            help="Sample rate")
        parser.add_argument("--feature-dim", type=int, default=FEATURE_DIM,
                            help="Acoustic feature dimension.")
        parser.add_argument("--pre-decision-ratio", type=int, default=1,
                            help="Pre decision ratio")

        parser.add_argument("--ctl-model-path-list", nargs="+", type=str, required=True,
                            help="Path list to ctl sign regression model")
        
        parser.add_argument("--remove-subword", type=int, required=True,
                           help="Number of tokens for latency control")

        # fmt: on
        return parser

    def load_model_vocab(self, args):

        filename = args.model_path
        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))

        state = checkpoint_utils.load_checkpoint_to_cpu(filename)

        task_args = state["cfg"]["task"]
        task_args.data = args.data_bin

        if args.config is not None:
            task_args.config_yaml = args.config

        task = tasks.setup_task(task_args)

        # build model for ensemble
        state["cfg"]["model"].load_pretrained_encoder_from = None
        state["cfg"]["model"].load_pretrained_decoder_from = None
        self.model = task.build_model(state["cfg"]["model"])
        self.model.load_state_dict(state["model"], strict=True)
        self.model.eval()
        self.model.share_memory()

        if self.gpu:
            self.model.cuda()

        # Set dictionary
        self.dict = {}
        self.dict["tgt"] = task.target_dictionary

        # bpe
        self.bpe = task.build_bpe(task_args)

        # ctl model

        for ctl_model_path in args.ctl_model_path_list:
            if not os.path.exists(ctl_model_path):
                raise IOError("CTL Model file not found: {}".format(ctl_model_path))
            ctl_state = checkpoint_utils.load_checkpoint_to_cpu(ctl_model_path)
            ctl_task_args = ctl_state["cfg"]["task"]
            ctl_task = tasks.setup_task(ctl_task_args)
            ctl_model = ctl_task.build_model(ctl_state["cfg"]["model"])
            ctl_model.load_state_dict(ctl_state["model"], strict=True)
            ctl_model.eval()
            ctl_model.share_memory()
            if self.gpu:
                ctl_model.cuda()
            if not hasattr(self, "ctl_model_list"):
                self.ctl_model_list = []
            self.ctl_model_list.append(ctl_model)

            ctl_label_norm = ctl_task_args.get("label_norm", None)
            ctl_min_len = ctl_task_args.get("min_len", 1)
            ctl_max_len = ctl_task_args.get("max_len", 90)
            # every CTL model is configured the same
            if hasattr(self, "ctl_label_norm"):
                assert self.ctl_label_norm == ctl_label_norm
                assert self.ctl_min_len == ctl_min_len
                assert self.ctl_max_len == ctl_max_len
            self.ctl_label_norm = ctl_label_norm
            self.ctl_min_len = ctl_min_len
            self.ctl_max_len = ctl_max_len
        

    def initialize_states(self, states):
        states.units.source = TensorListEntry()
        states.units.target = ListEntry()
        states.incremental_states = dict()

    def segment_to_units(self, segment, states):
        # encoder: Convert 2D list to tensor list
        features = torch.tensor(segment)
        if features is not None:
            return [features]
        else:
            return []

    def units_to_segment(self, units, states):
        # decoder: Merge sub word to full word. # units: unit_queue
        if self.model.decoder.dictionary.eos() == units[0]:
            return DEFAULT_EOS

        segments = []
        segment = []
        if None in units.value:
            units.value.remove(None)
        if None in states.unit_queue.target.value:
            states.unit_queue.target.value.remove(None)

        tmp = list(units.value)
        for index in tmp:
            if index is None:
                units.pop()
                continue
            token = self.model.decoder.dictionary.string([index])
            if token.startswith(BOW_PREFIX):
                if len(segment) == 0:
                    segment += [token.replace(BOW_PREFIX, "")]
                else:
                    for j in range(len(segment)):
                        states.units.target.append(units.pop())
                    
                    string_to_add = ["".join(segment)]

                    if self.model.decoder.dictionary.eos() == units[0]:
                        string_to_add += [DEFAULT_EOS]
                        segments += string_to_add
                        return segments

                    segments += string_to_add

                    segment = [token.replace(BOW_PREFIX, "")]
            else:
                if index == self.model.decoder.dictionary.eos():
                    segments += ["".join(segment)]
                    segments += [DEFAULT_EOS]
                    return segments

                segment += [token.replace(BOW_PREFIX, "")]
        
        return segments


    def update_model_encoder(self, states):
        if len(states.units.source) == 0:
            return
        src_indices = self.to_device(
            states.units.source.value.unsqueeze(0)
        )   # (B, T, D1024)
        src_lengths = self.to_device(
            torch.LongTensor([states.units.source.value.size(0)])
        )   # (B)

        states.encoder_states = self.model.encoder(src_indices, src_lengths)    # (B, T, D512)
        
        preds = []
        for ctl_model in self.ctl_model_list:
            with torch.no_grad():
                pred = ctl_model(source=src_indices, padding_mask=None)["pooled"].item()
            preds.append(pred)
        pred = sum(preds) / len(preds)
        
        if self.ctl_label_norm == "log":
            states.pred_len = int(round(math.exp(pred)))
        elif self.ctl_label_norm == "min_max":
            states.pred_len = int(round(pred * (self.ctl_max_len - self.ctl_min_len) + self.ctl_min_len))
        else:
            states.pred_len = round(pred) + 1

        torch.cuda.empty_cache()

    def update_states_read(self, states):
        # Happens after a read action.
        self.update_model_encoder(states)
    
    def policy(self, states):
        if not getattr(states, "encoder_states", None):
            return READ_ACTION
        nfeatures = states.units.source.__len__()

        tgt_indices = self.to_device(
            torch.LongTensor(
                [self.model.decoder.dictionary.eos()]
                + [x for x in states.units.target.value if x is not None]
            ).unsqueeze(0)
        )   # (B, T)
        tgt_str = self.model.decoder.dictionary.string(tgt_indices, bpe_symbol="sentencepiece")

        states.incremental_states["steps"] = {
            "src": nfeatures,
            "tgt": tgt_indices.size(1),
        }
        states.incremental_states["online"] = {
            "only": torch.tensor(not states.finish_read())
        }

        if states.finish_read():
            states.pred_len = self.max_len

        if not states.finish_read() and states.pred_len <= states.incremental_states["steps"]["tgt"]:
            return READ_ACTION

        # try to translate
        cur_indices = tgt_indices
        while True:
            x, outputs = self.model.decoder.forward(
                prev_output_tokens=cur_indices,
                encoder_out=states.encoder_states,
                incremental_state=None, 
            )
            lprobs = self.model.get_normalized_probs(
                [x[:, -1:]], log_probs=True
            )
            index = lprobs.argmax(dim=-1)
            if cur_indices.size(1) >= states.pred_len:
                break 
            if index == self.model.decoder.dictionary.eos():
                break
            if cur_indices.size(1) >= self.args.max_len:
                break
            cur_indices = torch.cat([cur_indices, index], dim=-1)
        
        cur_str = self.model.decoder.dictionary.string(cur_indices, bpe_symbol="sentencepiece")

        # remove tokens to control latency
        if not states.finish_read():
            cur_indices_len = cur_indices.size(1)
            after_remove_len = int(cur_indices_len - self.args.remove_subword)
            after_remove_len = max(after_remove_len, len(tgt_indices[0]))
            cur_indices = cur_indices[:, :after_remove_len]
        cur_str = self.model.decoder.dictionary.string(cur_indices, bpe_symbol="sentencepiece")

        # prevent infinite WRITE loops
        if not states.finish_read() and cur_str == states.prev_cur_str:
            return READ_ACTION
        else:
            states.prev_cur_str = cur_str
        # empty unit_queue
        while not states.unit_queue.target.empty():
            states.unit_queue.target.pop()
        for index in cur_indices[:, tgt_indices.size(1):][0]:
            states.unit_queue.target.push(index.item())
        torch.cuda.empty_cache()
        return WRITE_ACTION

    def predict(self, states):
        if states.finish_read():
            return self.model.decoder.dictionary.eos()
        return None
