#!/bin/bash
# Copyright 2020 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
nj=20
cmd=run.pl
feat_config=conf/fbank.conf
use_gpu=false
audio_feat=logfb
center=true
norm_var=false
context=150
attack_type=fgsm
eps=0
snr=100
save_wav_tar_thr=0.4
save_wav_non_thr=0.25
save_wav_path=""

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
set -e

if [ $# -ne 7 ]; then
  echo "Usage: $0 [options] <key> <enroll-file> <test-data-dir> <vector-file> <nnet-model> <output-scores> <output-snr>"
  echo "Options: "
  echo "  --feat-config <config-file>                      # feature extractor config"
  echo "  --audio-feat <logfb|mfcc>                        # feature type"
  echo "  --nj <nj>                                        # number of parallel jobs"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --center <true|false>                            # If true, normalize means in the sliding window cmvn (default:true)"
  echo "  --norm-var <true|false>                          # If true, normalize variances in the sliding window cmvn (default:false)"
  echo "  --use-gpu <bool|false>                           # If true, use GPU."
  echo "  --context <int|150>                              # Left context for short-time cmvn (default: 150)"
  echo "  --attack-type <str|fgsm>                         # Attack type"
  echo "  --eps <float|0>                                  # Attack epsilon"
  echo "  --snr <float|100>                                # Attack SNR"
  echo "  --save-wav-thr <float|0.75>                     # threshold to decide to save adversarial wav to disk"
  echo "  --save-wav-path <str|>                          # path to save adv wavs"
  exit 1;
fi

key_file=$1
enroll_file=$2
test_data=$3
vector_file=$4
nnet_file=$5
output_file=$6
snr_file=$7

output_dir=$(dirname $output_file)
log_dir=$output_dir/log

mkdir -p $log_dir
name=$(basename $output_file)


wav=$test_data/wav.scp
vad=$test_data/vad.scp

required="$wav $feat_config $key $enroll_file $vector_file $vad"

for f in $required; do
  if [ ! -f $f ]; then
    echo "$0: no such file $f"
    exit 1;
  fi
done

num_gpus=0
args=""
if [ "$use_gpu" == "true" ];then
    cmd="$cmd --gpu 1"
    num_gpus=1
    args="--use-gpu"
fi

if [ "$center" == "false" ];then
    args="${args} --mnv-no-norm-mean"
fi
if [ "$norm_var" == "true" ];then
    args="${args} --mvn-norm-var"
fi
args="${args} --mvn-context $context"

if [ -n "${save_wav_path}" ];then
    args="${args} --save-adv-wav-path $save_wav_path --save-adv-wav --save-adv-wav-tar-thr $save_wav_tar_thr --save-adv-wav-non-thr -$save_wav_non_thr"
fi

echo "$0: score $key_file to $output_dir"

$cmd JOB=1:$nj $log_dir/${name}.JOB.log \
    hyp_utils/torch.sh --num-gpus $num_gpus \
    steps_adv/torch-eval-cosine-scoring-from-adv-test-wav.py \
    @$feat_config --audio-feat $audio_feat ${args} \
    --v-file scp:$vector_file \
    --key-file $key_file \
    --enroll-file $enroll_file \
    --test-wav-file $wav \
    --vad scp:$vad \
    --model-path $nnet_file \
    --attack-type $attack_type \
    --attack-snr $snr \
    --attack-eps $eps \
    --score-file $output_file \
    --snr-file $snr_file \
    --seg-part-idx JOB --num-seg-parts $nj || exit 1


for((j=1;j<=$nj;j++));
do
    cat $output_file-$(printf "%03d" 1)-$(printf "%03d" $j)
done | sort -u > $output_file

for((j=1;j<=$nj;j++));
do
    cat $snr_file-$(printf "%03d" 1)-$(printf "%03d" $j)
done | sort -u > $snr_file


