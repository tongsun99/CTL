import math
import os
import json
import numpy as np
import torch
import yaml

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


class WaitinfAgent(SignAgent):

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
                            help="Number of frames contained in one segment")

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

    def initialize_states(self, states):
        # self.feature_extractor.clear_cache()
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
        # decoder: Merge sub word to full word.
        if self.model.decoder.dictionary.eos() == units[0]:
            return DEFAULT_EOS

        segment = []
        if None in units.value:
            units.value.remove(None)

        for index in units:
            if index is None:
                units.pop()
            token = self.model.decoder.dictionary.string([index])
            if token.startswith(BOW_PREFIX):
                if len(segment) == 0:
                    segment += [token.replace(BOW_PREFIX, "")]
                else:
                    for j in range(len(segment)):
                        units.pop()

                    string_to_return = ["".join(segment)]

                    if self.model.decoder.dictionary.eos() == units[0]:
                        string_to_return += [DEFAULT_EOS]

                    return string_to_return
            else:
                segment += [token.replace(BOW_PREFIX, "")]

        if (
            len(units) > 0
            and self.model.decoder.dictionary.eos() == units[-1]
            or len(states.units.target) > self.max_len
        ):
            tokens = [self.model.decoder.dictionary.string([unit]) for unit in units]
            return ["".join(tokens).replace(BOW_PREFIX, "")] + [DEFAULT_EOS]

        return None

    def update_model_encoder(self, states):
        if len(states.units.source) == 0:
            return
        src_indices = self.to_device(
            states.units.source.value.unsqueeze(0)
        )   # (T, B, D)
        src_lengths = self.to_device(
            torch.LongTensor([states.units.source.value.size(0)])
        )   # (T)

        states.encoder_states = self.model.encoder(src_indices, src_lengths)
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
        )

        states.incremental_states["steps"] = {
            "src": nfeatures,
            "tgt": tgt_indices.size(1),
        }

        states.incremental_states["online"] = {
            "only": torch.tensor(not states.finish_read())
        }

        if not states.finish_read():
            return READ_ACTION

        x, outputs = self.model.decoder.forward(
            prev_output_tokens=tgt_indices,
            encoder_out=states.encoder_states,
            incremental_state=None,
        )

        states.decoder_out = x

        states.decoder_out_extra = outputs

        torch.cuda.empty_cache()

        return WRITE_ACTION

    def predict(self, states):
        decoder_states = states.decoder_out

        lprobs = self.model.get_normalized_probs(
            [decoder_states[:, -1:]], log_probs=True
        )

        index = lprobs.argmax(dim=-1)

        index = index[0, 0].item()

        if (states.finish_read() and states.incremental_states["steps"]["tgt"] > self.args.max_len):
            index = self.model.decoder.dictionary.eos()
            return index

        if (
            self.force_finish
            and index == self.model.decoder.dictionary.eos()
            and not states.finish_read()
        ):
            # If we want to force finish the translation
            # (don't stop before finish reading), return a None
            # self.model.decoder.clear_cache(states.incremental_states)
            index = None

        return index
