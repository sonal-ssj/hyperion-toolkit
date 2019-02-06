#!/bin/bash
# Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export PYTHONPATH=$PWD:$PYTHONPATH

rm -rf ./tests/data_out/*
mkdir ./tests/data_out/ark
mkdir ./tests/data_out/h5

#py.test ./tests/hyperion/io
#py.test ./tests/hyperion/feats
#py.test ./tests/hyperion/utils
py.test ./tests/hyperion/helpers
#py.test ./tests


