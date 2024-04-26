#!/bin/bash
# Copyright 2022 Johns Hopkins University (Jesus Villalba)
# Apache 2.0
#
# Downloads NIST scoring tools for LRE22

set -e 
tool=lre-scorer
s_dir=lre-scorer

# shareable link:
# https://drive.google.com/file/d/13pvUhFPGLgqId5yB8i25X__LFXKIU-ju/view?usp=sharing

wget --no-check-certificate "https://drive.google.com/uc?export=download&id=13pvUhFPGLgqId5yB8i25X__LFXKIU-ju" -O $tool.tar.gz
tar xzvf $tool.tar.gz

if [ ! -f $s_dir/scorerLRE22.py ];then
    echo "the scoring tool wasn't dowloaded correctly, download manually"
    exit 1
fi

rm -f $tool.tar.gz

