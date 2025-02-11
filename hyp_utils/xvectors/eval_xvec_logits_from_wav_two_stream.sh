#!/bin/bash
#               2019  Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
nj=30
cmd="run.pl"

chunk_length=0     # The chunk size over which the embedding is extracted.
use_gpu=false
write_utt2num_frames=true  # If true writes utt2num_frames.
feat_config=conf/fbank80_stmn_16k.yaml
stage=0
min_utt_length=500
max_utt_length=12000
random_utt_length=false
aug_config=""
num_augs=0
use_bin_vad=true

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ] && [ $# != 4 ]; then
  echo "Usage: $0 [options] <nnet-model> <data> <data2> <logits-dir> [<data-out-dir>]"
  echo " e.g.: $0 --feat-config conf/fbank_mvn.pyconf --aug-config conf/noise_aug.yml exp/xvector_nnet/model.pt data/train exp/logits_train [data/train_aug]"
  echo "main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --use-gpu <bool|false>                           # If true, use GPU."
  echo "  --nj <n|30>                                      # Number of jobs"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --use-bin-vad <bool|true>                        # If true, uses binary VAD from vad.scp"
  echo "  --write-utt2num-frames <bool|tru>                # If true, write utt2num_frames file."
  echo "  --chunk-length <n|0>                             # If provided, applies encoder with specified chunk-length and "
  echo "                                                   # concatenates the chunks outputs before pooling"
  echo "  --feat-config <str>                              # feature/mvn config file"
  echo "  --aug-config <str>                               # augmentation config file"
  echo "  --random-utt-length                              # If true, extracts a random chunk from the utterance between "
  echo "                                                   # min_utt_length and max_utt_length"
  echo "  --min-utt-length <n|0>                           # "
  echo "  --max-utt-length <n|0>                           # "
  

fi

nnet_file=$1
data_dir=$2
data_dir_2=$3
output_dir=$4
data_out_dir=$5

for f in $data_dir/wav.scp ; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

log_dir=$output_dir/log
mkdir -p $log_dir

num_gpus=0
args=""
if [ "$use_gpu" == "true" ];then
    cmd="$cmd --gpu 1"
    num_gpus=1
    args="--use-gpu"
fi

if [ "$use_bin_vad" == "true" ];then
    f=$data_dir/vad.scp 
    [ ! -f $f ] && echo "No such file $f" && exit 1;
    args="${args} --vad scp:$f"
fi

if [ -n "$aug_config" ];then
    args="${args} --aug-cfg $aug_config --num-augs $num_augs --aug-info-path $output_dir/aug_info.JOB.csv"
fi

if [ "$random_utt_length" == "true" ];then
    args="${args} --random-utt-length --min-utt-length $min_utt_length --max-utt-length $max_utt_length"
fi

if [ "$write_utt2num_frames" == "true" ];then
    write_num_frames_opt="--write-num-frames $output_dir/utt2num_frames.JOB"
fi

if [ $stage -le 0 ];then
    $cmd JOB=1:$nj $output_dir/log/eval_logits.JOB.log \
	hyp_utils/conda_env.sh --num-gpus $num_gpus \
	torch-eval-xvec-logits-from-wav-two-stream.py \
	--feats $feat_config ${args} $write_num_frames_opt \
	--part-idx JOB --num-parts $nj \
	--input $data_dir/wav.scp \
    --input2 $data_dir_2/wav.scp \
	--model-path $nnet_file --chunk-length $chunk_length \
	--output ark,scp:$output_dir/logits.JOB.ark,$output_dir/logits.JOB.scp || exit 1;
fi


if [ $stage -le 1 ]; then
  echo "$0: combining logits across jobs"
  for j in $(seq $nj); do cat $output_dir/logits.$j.scp; done > $output_dir/logits.scp || exit 1;
  if [ "$write_utt2num_frames" == "true" ];then
      for n in $(seq $nj); do
	  cat $output_dir/utt2num_frames.$n || exit 1;
      done > $output_dir/utt2num_frames || exit 1
  fi

  if [ -f $output_dir/aug_info.1.csv ];then
      cat $output_dir/aug_info.1.csv > $output_dir/aug_info.csv
      for j in $(seq 2 $nj);
      do
	  tail -n +2 $output_dir/aug_info.$j.csv
      done >> $output_dir/aug_info.csv
  fi
fi

if [ $stage -le 2 ]; then
    if [ -n "$data_out_dir" ];then
	echo "$0: creating data dir $data_out_dir for augmented x-vectors"
	mkdir -p $data_out_dir
	awk -F "," '$1 != "key_aug" { print $1,$2}' $output_dir/aug_info.csv \
	> $data_out_dir/augm2clean 
	awk -v u2s=$data_dir/utt2spk 'BEGIN{
while(getline < u2s)
{
   spk[$1]=$2
}
}
{ print $1,spk[$2]}' $data_out_dir/augm2clean > $data_out_dir/utt2spk
	utils/utt2spk_to_spk2utt.pl $data_out_dir/utt2spk > $data_out_dir/spk2utt
	#cp $output_dir/utt2num_frames $data_out_dir/utt2num_frames_logits
    else
        echo "Checking if File exists..."
        echo $data_dir/utt2num_frames_logits
        if [ ! -f $data_dir/utt2num_frames_logits ]
        then
            echo "File does not exist in Bash"
            #cp $output_dir/utt2num_frames $data_dir/utt2num_frames_logits
            echo "Warning: Original script did cp $output_dir/utt2num_frames $data_dir/utt2num_frames_logits"
            echo "But we are skipping it as mostly $data_dir is not writable "
        else
            echo "File already found. No need to copy again"
            echo "Warning: Original script did cp $output_dir/utt2num_frames $data_dir/utt2num_frames_logits"
            echo "But we are skipping it as mostly $data_dir is not writable and the files are essentially the same "
            echo "------ diff output <start> -----"
            diff $output_dir/utt2num_frames_logits $data_dir/utt2num_frames_logits
            echo "------ diff output <end>-----"
        fi
    fi
fi
