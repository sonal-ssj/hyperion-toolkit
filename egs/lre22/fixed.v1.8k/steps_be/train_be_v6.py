#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import sys
import os
import logging
from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
    ActionParser,
    namespace_to_dict,
    ActionYesNo,
)
import time
from pathlib import Path

import numpy as np

from hyperion.hyp_defs import config_logger
from hyperion.utils import SegmentSet
from hyperion.utils.math import softmax
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.np.transforms import TransformList, PCA, LNorm
from hyperion.np.classifiers import LinearGBE as GBE
from hyperion.np.metrics import (
    compute_accuracy,
    compute_confusion_matrix,
    print_confusion_matrix,
)

tar_langs = (
    "afr-afr",
    "ara-aeb",
    "ara-arq",
    "ara-ayl",
    "eng-ens",
    "eng-iaf",
    "fra-ntf",
    "nbl-nbl",
    "orm-orm",
    "tir-tir",
    "tso-tso",
    "ven-ven",
    "xho-xho",
    "zul-zul",
)


def compute_metrics(y_true, y_pred, labels):

    acc = compute_accuracy(y_true, y_pred)
    logging.info("training acc: %.2f %%", acc * 100)
    logging.info("non-normalized confusion matrix:")
    C = compute_confusion_matrix(y_true, y_pred, normalize=False)
    print_confusion_matrix(C, labels)
    logging.info("normalized confusion matrix:")
    C = compute_confusion_matrix(y_true, y_pred, normalize=True)
    print_confusion_matrix(C * 100, labels, fmt=".2f")


def train_be(
    v_file,
    train_list,
    cv_v_file,
    cv_list,
    afr_v_file,
    afr_list,
    class_name,
    do_lnorm,
    whiten,
    pca,
    gbe,
    output_dir,
    verbose,
):
    config_logger(verbose)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("loading data")
    train_segs = SegmentSet.load(train_list)
    v_reader = DRF.create(v_file)
    x_trn = v_reader.read(train_segs["id"], squeeze=True)
    del v_reader
    logging.info("loaded %d train samples", x_trn.shape[0])

    segs_cv = SegmentSet.load(cv_list)
    # ary_idx = segs_lre17[class_name] == "ara-ary"
    # segs_ary = segs_lre17.loc[ary_idx]

    segs_cv = SegmentSet.load(cv_list)
    cv_idx = np.zeros((len(segs_cv),), dtype=bool)
    for lang in tar_langs:
        cv_idx_i = segs_cv[class_name] == lang
        cv_idx = np.logical_or(cv_idx, cv_idx_i)

    segs_cv = segs_cv.loc[cv_idx]
    # segs_cv.loc[segs_cv[class_name] == "eng-ine", class_name] = "eng-iaf"

    # v_reader = DRF.create(cv_v_file)
    # x_cv = v_reader.read(segs_cv["id"], squeeze=True)
    # logging.info("loaded %d cv samples", x_cv.shape[0])

    segs_afr = SegmentSet.load(afr_list)
    afr_idx = np.zeros((len(segs_afr),), dtype=bool)
    for lang in tar_langs:
        afr_idx_i = segs_afr[class_name] == lang
        afr_idx = np.logical_or(afr_idx, afr_idx_i)

    segs_afr = segs_afr.loc[afr_idx]

    v_reader = DRF.create(afr_v_file)
    x_afr = v_reader.read(segs_afr["id"], squeeze=True)
    logging.info("loaded %d afr samples", x_afr.shape[0])

    class_ids = (
        list(train_segs[class_name].values)
        # + list(segs_cv[class_name].values)
        + list(segs_afr[class_name].values)
    )
    # x_trn = np.concatenate((x_trn, x_cv, x_afr), axis=0)
    x_trn = np.concatenate((x_trn, x_afr), axis=0)
    labels, y_true = np.unique(class_ids, return_inverse=True)
    logging.info("%d training samples", x_trn.shape[0])

    logging.info("PCA args=%s", str(pca))
    pca_var_r = pca["pca_var_r"]
    pca_dim = pca["pca_dim"]
    if pca_var_r is not None and pca_var_r < 1.0 or pca_dim is not None:
        logging.info("training PCA")
        pca = PCA(**pca)
        pca.fit(x_trn)
        logging.info("PCA dimension: %d", pca.pca_dim)
        logging.info("apply PCA")
        x_trn = pca(x_trn)
    else:
        pca = None

    if do_lnorm:
        lnorm = LNorm()
        if whiten:
            logging.info("training whitening")
            lnorm.fit(x_trn)

        logging.info("apply lnorm")
        x_trn = lnorm(x_trn)
    else:
        lnorm = None

    logging.info("GBE args=%s", str(gbe))
    model = GBE(labels=labels, **gbe)
    model.fit(x_trn, y_true)
    logging.info("trained GBE")
    scores = model(x_trn)
    y_pred = np.argmax(scores, axis=-1)

    compute_metrics(y_true, y_pred, labels)

    logging.info("Saving transforms and GBE")
    transforms = []
    if pca is not None:
        transforms.append(pca)
    if lnorm is not None:
        transforms.append(lnorm)

    if transforms:
        transforms = TransformList(transforms)
        transforms.save(output_dir / "transforms.h5")

    model.save(output_dir / "model_gbe.h5")


if __name__ == "__main__":

    parser = ArgumentParser(description="Train linear GBE Classifier",)

    parser.add_argument("--v-file", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--cv-v-file", required=True)
    parser.add_argument("--cv-list", required=True)
    parser.add_argument("--afr-v-file", required=True)
    parser.add_argument("--afr-list", required=True)
    PCA.add_class_args(parser, prefix="pca")
    GBE.add_class_args(parser, prefix="gbe")
    parser.add_argument("--class-name", default="class_id")
    parser.add_argument("--do-lnorm", default=True, action=ActionYesNo)
    parser.add_argument("--whiten", default=True, action=ActionYesNo)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    train_be(**namespace_to_dict(args))