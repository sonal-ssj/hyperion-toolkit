"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from jsonargparse import ArgumentParser, ActionParser

import torch
import torch.nn as nn

# import torch.nn.functional as nnf

from ...torch_model import TorchModel
from ...utils import remove_silence


class HFWav2XVector(TorchModel):
    """Abstract Base class for x-vector models that use a Hugging Face Model as feature extractor.

    Attributes:
       hf_feats: hugging face model wrapper object.
       xvector: x-vector model object.
       feat_fusion_start: the input to x-vector model will fuse the wav2vec layers from "feat_fusion_start" to
                          the wav2vec "num_layers".
       feat_fusion_method: method to fuse the hidden layers from the wav2vec model, when more
                           than one layer is used.
    """

    def __init__(
        self, hf_feats, xvector, feat_fusion_start=0, feat_fusion_method="weighted-avg"
    ):

        super().__init__()
        self.hf_feats = hf_feats
        self.xvector = xvector
        self.feat_fusion_start = feat_fusion_start
        self.feat_fusion_method = feat_fusion_method
        self._make_fuser()

    def _make_fuser(self):
        if self.feat_fusion_method == "last":
            self.feat_fuser = None
            return

        num_layers = self.hf_feats.num_encoder_layers + 1 - self.feat_fusion_start
        layer_dim = self.hf_feats.hidden_size
        if self.feat_fusion_method == "weighted-avg":
            self.feat_fuser = nn.Parameter(torch.zeros(num_layers))
        elif self.feat_fusion_method == "linear":
            self.feat_fuser = nn.Linear(num_layers, 1, bias=False)
            self.feat_fuser.weights.data = torch.ones(num_layers) / num_layers
        elif self.feat_fusion_method == "cat":
            self.feat_fuser = nn.Linear(num_layers * layer_dim, layer_dim, bias=False)

    def _fuse_hid_feats(self, hid_feats):
        """Fuses the hidden features from the Wav2Vec model.

        Args:
          hid_feats: list of hidden features Tensors from Wav2Vec model.

        Returns:
          Tensor of fused features (batch, channels, time)
        """
        if len(hid_feats) == 1:
            # There is only one layer of features
            return hid_feats[0]

        hid_feats = hid_feats[self.feat_fusion_start :]
        if self.feat_fusion_method == "weighted-avg":
            hid_feats = torch.stack(hid_feats, dim=-1)
            norm_weights = nn.functional.softmax(self.feat_fuser, dim=-1)
            feats = torch.sum(hid_feats * norm_weights, dim=-1)
        elif self.feat_fusion_method == "linear":
            hid_feats = torch.stack(hid_feats, dim=-1)
            feats = self.feat_fuser(hid_feats)
        elif self.feat_fusion_method == "cat":
            hid_feats = torch.cat(hid_feats, dim=-1)
            feats = self.feat_fuser(hid_feats)
        elif self.feat_fusion_method == "last":
            feats = hid_feats[-1]

        return feats

    def update_loss_margin(self, epoch):
        """Updates the value of the margin in AAM/AM-softmax losses
           given the epoch number

        Args:
          epoch: epoch which is about to start
        """
        self.xvector.update_loss_margin(epoch)

    def forward_feats(self, x, x_lengths, return_feat_layers=None):
        return_hid_states = (
            False
            if return_feat_layers is None and self.feat_fusion_method == "last"
            else True
        )
        hf_output = self.hf_feats(x, x_lengths, return_hid_states=return_hid_states)
        feat_lengths = hf_output["hidden_states_lengths"]
        if return_hid_states:
            hid_feats = hf_output["hidden_states"]
            feats = self._fuse_hid_feats(hid_feats)
        else:
            hid_feats = None
            feats = hf_output["last_hidden_state"]

        feats = feats.transpose(1, 2)
        if return_feat_layers is not None:
            # add hidden feats from wav2vec to the output. We transpose to be (batch, C, time)
            # as the hidden features of the x-vector encoder.
            hid_feats = [
                f.transpose(1, 2)
                for i, f in enumerate(hid_feats)
                if i in return_feat_layers
            ]
        else:
            hid_feats = None

        return feats, hid_feats, feat_lengths

    def forward(
        self,
        x,
        x_lengths=None,
        y=None,
        return_feat_layers=None,
        return_enc_layers=None,
        return_classif_layers=None,
        return_logits=True,
    ):
        """Forward function. If returns the logits posteriors of the classes.
        It can also returns the hidden representations in the wav2vec feature extractor,
        the x-vector encoder and the
        classification head. In this case the ouput variable is a dictionary.

        Args:
          x: input features tensor with shape=(batch, in_feats, time)
          x_lengths: time lengths of the features with shape=(batch,)
          y: target classes torch.long tensor with shape=(batch,)
          return_feat_layers: list of integers indicating, which wav2vec layers
                             we should return. If None, no wav2vec layers are returned.
          return_enc_layers: list of integers indicating, which encoder layers
                             we should return. If None, no encoder layers are returned.
          return_enc_layers: list of integers indicating, which classification head layers
                             we should return. If None, no head layers are returned.
          return_logits: if True, it adds the logits to the output dictionary.
        Returns:
          Tensor with class logits with shape=(batch, num_classes) or
          Dictionary with "logits", "h_enc" (list of hidden encoder layers),
          "h_classif" (list hidden classification head layers), "h_feats" (wav2vec features)
        """
        feats, hid_feats, feat_lengths = self.forward_feats(
            x, x_lengths, return_feat_layers
        )
        output = self.xvector(
            feats,
            feat_lengths,
            y,
            return_enc_layers=return_enc_layers,
            return_classif_layers=return_classif_layers,
            return_logits=return_logits,
        )

        if not return_feat_layers:
            return output

        if not isinstance(output, dict):
            # if the xvector just returned the logits we put then into a dictionary
            # to append the hid feats later.
            output["logits"] = output

        output["h_feats"] = hid_feats
        return output

    def extract_embed(
        self,
        x,
        x_lengths=None,
        vad_samples=None,
        chunk_length=0,
        embed_layer=None,
        detach_chunks=False,
    ):

        if vad_samples is not None:
            x, x_lengths = remove_silence(x, x_lengths)

        feats, _, feat_lengths = self.forward_feats(x, x_lengths)
        xvec_chunk_length = int(chunk_length * feats.size(-1) // x.size(-1))
        return self.xvector.extract_embed(
            feats, feat_lengths, xvec_chunk_length, embed_layer, detach_chunks
        )

    @staticmethod
    def filter_args(**kwargs):
        valid_args = (
            "hf_feats",
            "xvector",
            "feat_fusion_start",
            "feat_fusion_method",
        )
        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        return args

    def get_config(self):

        hf_cfg = self.hf_feats.get_config()
        xvec_cfg = self.xvector.get_config()
        del hf_cfg["class_name"]
        del xvec_cfg["class_name"]
        config = {
            "hf_feats": hf_cfg,
            "xvector": xvec_cfg,
            "feat_fusion_start": self.feat_fusion_start,
            "feat_fusion_method": self.feat_fusion_method,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def add_class_args(parser, prefix=None, skip=set()):

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--feat-fusion-start",
            default=0,
            type=int,
            help=(
                "the input to x-vector model will fuse the wav2vec layers from feat_fusion_start to"
                "the wav2vec num_layers"
            ),
        )
        parser.add_argument(
            "--feat-fusion-method",
            default="weighted-avg",
            choices=["weighted-avg", "linear", "cat", "last"],
            help=(
                "method to fuse the hidden layers from the wav2vec model "
                "in [weighted-avg, cat]"
            ),
        )

        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix,
                action=ActionParser(parser=parser),
                help="xvector options",
            )