#!/usr/bin/env python
"""
 Copyright 2019 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""



from typing import Dict, List, Tuple

import sentencepiece as spm
import torch.nn as nn

import sys
import os
from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
    ActionParser,
    namespace_to_dict,
)
import time
import logging

import numpy as np
import pandas as pd

import torch

from hyperion.hyp_defs import config_logger, float_cpu, set_float_cpu
from hyperion.utils import Utt2Info
from hyperion.io import DataWriterFactory as DWF
from hyperion.io import SequentialAudioReader as AR
from hyperion.np.augment import SpeechAugment

from hyperion.torch.utils import open_device
from hyperion.torch.narchs import AudioFeatsMVN as AF
from hyperion.torch import TorchModelLoader as TML

from hyperion.torch.models.wav2transducer.beam_search import greedy_search, beam_search

def init_device(use_gpu):
    set_float_cpu("float32")
    num_gpus = 1 if use_gpu else 0
    logging.info("initializing devices num_gpus={}".format(num_gpus))
    device = open_device(num_gpus=num_gpus)
    return device


def load_model(model_path, device):
    logging.info("loading model {}".format(model_path))
    model = TML.load(model_path)
    logging.info("transducer-model={}".format(model))
    model.to(device)
    model.eval()
    return model



# def decode_dataset(
#     dl: torch.utils.data.DataLoader,
#     params: AttributeDict,
#     model: nn.Module,
#     sp: spm.SentencePieceProcessor,
# ) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
#     """Decode dataset.
#     Args:
#       dl:
#         PyTorch's dataloader containing the dataset to decode.
#       params:
#         It is returned by :func:`get_params`.
#       model:
#         The neural model.
#       sp:
#         The BPE model.
#     Returns:
#       Return a dict, whose key may be "greedy_search" if greedy search
#       is used, or it may be "beam_7" if beam size of 7 is used.
#       Its value is a list of tuples. Each tuple contains two elements:
#       The first is the reference transcript, and the second is the
#       predicted result.
#     """
#     num_cuts = 0

#     try:
#         num_batches = len(dl)
#     except TypeError:
#         num_batches = "?"

#     if decoding_method == "greedy_search":
#         log_interval = 100
#     else:
#         log_interval = 2

#     results = defaultdict(list)
#     for batch_idx, batch in enumerate(dl):
#         texts = batch["supervisions"]["text"]
#         cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

#         hyps_dict = decode_one_batch(
#             params=params,
#             model=model,
#             sp=sp,
#             batch=batch,
#         )

#         for name, hyps in hyps_dict.items():
#             this_batch = []
#             assert len(hyps) == len(texts)
#             for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
#                 ref_words = ref_text.split()
#                 this_batch.append((cut_id, ref_words, hyp_words))

#             results[name].extend(this_batch)

#         num_cuts += len(texts)

#         if batch_idx % log_interval == 0:
#             batch_str = f"{batch_idx}/{num_batches}"

#             logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
#     return results


def decode_one_batch(
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    x: torch.Tensor,
    decoding_method = "beam_search"
) -> Dict[str, List[List[str]]]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:
        - key: It indicates the setting used for decoding. For example,
               if greedy_search is used, it would be "greedy_search"
               If beam search with a beam size of 7 is used, it would be
               "beam_7"
        - value: It contains the decoding result. `len(value)` equals to
                 batch size. `value[i]` is the decoding result for the i-th
                 utterance in the given batch.
    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict.
    """
    device = model.device
    feature = x #batch["inputs"]
    assert x.shape[0] == 1
    assert feature.ndim == 2

    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    feature_lens = torch.Tensor([x.shape[1]]).int() #batch["supervisions"]
    # feature_lens = supervisions["num_frames"].to(device)

    # encoder_out, encoder_out_lens = model.encoder(x=feature, x_lens=feature_lens)

    # print("feature",feature.shape)
    # print("feature_lens",feature_lens)
    encoder_out, hid_feats, encoder_out_lens = model.forward_feats(x=feature, x_lengths=feature_lens)
    
    hyps = []
    batch_size = encoder_out.size(0)

    encoder_out = encoder_out.permute(0, 2, 1)  # (N, C, T) ->(N, T, C)

    for i in range(batch_size):
        # fmt: off
        encoder_out_i = encoder_out[i:i+1, :encoder_out_lens[i]]
        # fmt: on
        if decoding_method == "greedy_search":
            hyp = greedy_search(model=model, encoder_out=encoder_out_i)
        elif decoding_method == "beam_search":
            hyp = beam_search(
                model=model, encoder_out=encoder_out_i, beam=5
            )
        else:
            raise ValueError(f"Unsupported decoding method: {decoding_method}")
        hyps.append(sp.decode(hyp).split())
    
    logging.info("hyps:{}".format(" ".join(hyps[0])))
    
    if decoding_method == "greedy_search":
        return hyps[0] #{"greedy_search": hyps}
    else:
        return hyps[0] #{f"beam_{params.beam_size}": hyps}


def decode_transducer(
    input_spec,
    output_spec,
    scp_sep,
    model_path,
    bpe_model,
    use_gpu,
    **kwargs
):

    device = init_device(use_gpu)
    model = load_model(model_path, device)

    sp  = spm.SentencePieceProcessor()
    sp.load(bpe_model)
    # blank_id = self.sp.piece_to_id("<blk>")
    # vocab_size = self.sp.get_piece_size()

    # if write_num_frames_spec is not None:
    #     keys = []
    #     info = []

    augmenter = None
    aug_df = None
    num_augs = 1

    ar_args = AR.filter_args(**kwargs)
    logging.info("opening output: %s" % (output_spec))
    # with DWF.create(output_spec, scp_sep=scp_sep) as writer:
    with open(output_spec,"w") as writer:
        logging.info(
            "opening input stream: {} with args={}".format(input_spec, ar_args)
        )
        with AR(input_spec, **ar_args) as reader:
            while not reader.eof():
                t1 = time.time()
                key, x0, fs = reader.read(1)
                if len(key) == 0:
                    break

                x0 = x0[0]
                key0 = key[0]
                t2 = time.time()

                logging.info("processing utt %s" % (key0))
                for aug_id in range(num_augs):
                    t3 = time.time()
                    key, x = key0, x0 #augment(key0, x0, augmenter, aug_df, aug_id)
                    t4 = time.time()
                    with torch.no_grad():
                        x = torch.tensor(
                            x[None, :], dtype=torch.get_default_dtype()
                        ).to(device)

                        t5 = time.time()
                        tot_frames = x.shape[1]

                        logging.info(
                            "utt %s detected %d/%d (%.2f %%) speech frames"
                            % (
                                key,
                                x.shape[1],
                                tot_frames,
                                x.shape[1] / tot_frames * 100,
                            )
                        )


                        t6 = time.time()
                        if x.shape[1] == 0:
                            y = np.zeros((model.embed_dim,), dtype=float_cpu())
                        else:
                            # x = x.transpose(1, 2).contiguous()
                            # x = torch.unsqueeze(x,2)
                            # writer.write(key + ' ' + "abc")
                            y = decode_one_batch(model=model, sp=sp, x=x)
                            writer.write(key + ' ' + ' '.join(y) + "\n")

                            # y = (
                            #     model.extract_embed(
                            #         x,
                            #         chunk_length=chunk_length,
                            #         embed_layer=embed_layer,
                            #     )
                            #     .cpu()
                            #     .numpy()[0]
                            # )

                    t7 = time.time()
                    # writer.write([key], [y])
                    # if write_num_frames_spec is not None:
                    #     keys.append(key)
                    #     info.append(str(x.shape[-1]))

                    t8 = time.time()
                    read_time = t2 - t1
                    tot_time = read_time + t8 - t3
                    logging.info(
                        (
                            "utt %s total-time=%.3f read-time=%.3f "
                            "aug-time=%.3f feat-time=%.3f "
                            "vad-time=%.3f embed-time=%.3f write-time=%.3f "
                            "rt-factor=%.2f"
                        )
                        % (
                            key,
                            tot_time,
                            read_time,
                            t4 - t3,
                            t5 - t4,
                            t6 - t5,
                            t7 - t6,
                            t8 - t7,
                            x0.shape[0] / fs[0] / tot_time,
                        )
                    )

    # if write_num_frames_spec is not None:
    #     logging.info("writing num-frames to %s" % (write_num_frames_spec))
    #     u2nf = Utt2Info.create(keys, info)
    #     u2nf.save(write_num_frames_spec)

    # if aug_info_path is not None:
    #     aug_df = pd.concat(aug_df, ignore_index=True)
    #     aug_df.to_csv(aug_info_path, index=False, na_rep="n/a")


if __name__ == "__main__":

    parser = ArgumentParser(
        description=(
            "Extracts x-vectors from waveform computing " "acoustic features on the fly"
        )
    )

    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--input", dest="input_spec", required=True)
    parser.add_argument("--scp-sep", default=" ", help=("scp file field separator"))

    AR.add_class_args(parser)


    AF.add_class_args(parser, prefix="feats")

    parser.add_argument("--model-path", required=True)

    parser.add_argument("--bpe-model", required=True)

    parser.add_argument("--output", dest="output_spec", required=True)
    parser.add_argument(
        "--use-gpu", default=False, action="store_true", help="extract xvectors in gpu"
    )
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    decode_transducer(**namespace_to_dict(args))
