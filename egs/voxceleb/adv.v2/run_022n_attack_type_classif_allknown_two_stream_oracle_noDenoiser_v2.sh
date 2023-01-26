#!/bin/bash
# Copyright
#                2019   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#

# Script to train two stream signature extractor using oracle noise 
# Takes in features in two channels

# Changes to default config: 
# (1) Replaced `s` with `cos_scale` 

#RuntimeError: CUDA out of memory. Tried to allocate 64.00 MiB (GPU 0; 10.92 GiB total capacity; 9.96 GiB already allocated; 19.06 MiB free; 10.12 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
# (2) Changed sign_nnet_batch_size_1gpu to 112 (from 128 default) due to CUDA error 

# Still gives error RuntimeError: CUDA out of memory. Tried to allocate 64.00 MiB (GPU 0; 10.92 GiB total capacity; 9.96 GiB already allocated; 19.06 MiB free; 10.12 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF 
#(2) Changed sign_nnet_batch_size_1gpu to 64 (from 128 default) due to CUDA error

#Still gives error RuntimeError: CUDA out of memory for two stream
#(3) Changed sign_nnet_batch_size_1gpu to 32 due to CUDA error

. ./cmd.sh
. ./path.sh
set -e

stage=1
ngpu=1
# config_file=default_config.sh
# config_file=default_config_reduced_batch_size.sh
config_file=default_config_reduced_batch_size_32_two_stream_v2.sh ## ->> 2 channels
resume=false
interactive=false
num_workers=4
xvec_use_gpu=true
xvec_chunk_length=12800

. parse_options.sh || exit 1;
. $config_file
. datapath.sh

if [ "$xvec_use_gpu" == "true" ];then
    xvec_args="--use-gpu true --chunk-length $xvec_chunk_length"
    xvec_cmd="$cuda_eval_cmd"
else
    xvec_cmd="$train_cmd"
fi

batch_size=$(($sign_nnet_batch_size_1gpu*$ngpu))
grad_acc_steps=$(echo $batch_size $sign_nnet_eff_batch_size | awk '{ print int($2/$1+0.5)}')
log_interval=$(echo 100*$grad_acc_steps | bc)
list_dir=data/$attack_type_split_tag
list_dir_OracleAdvNoise=data/$attack_type_split_tag"_noise_only"

args=""
if [ "$resume" == "true" ];then
    args="--resume"
fi

if [ "$interactive" == "true" ];then
    export cuda_cmd=run.pl
fi

sign_nnet_reldir=$spknet_name/$sign_nnet_name/$attack_type_split_tag"_two_stream_oracle_noDenoiser_v2"
sign_nnet_dir=exp/sign_nnets/$sign_nnet_reldir
sign_dir=exp/signatures/$sign_nnet_reldir
logits_dir=exp/logits/$sign_nnet_reldir
sign_nnet=$sign_nnet_dir/model_ep0020.pth


# Network Training
# if [ $stage -le 1 ]; then
#     echo "Train two-stream signature network on all attacks"
#     mkdir -p $sign_nnet_dir/log
#     $cuda_cmd --gpu $ngpu $sign_nnet_dir/log/train.log \
# 	hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
# 	torch-train-xvec-from-wav-two-stream.py  $sign_nnet_command --cfg $sign_nnet_config \
# 	--audio-path $list_dir/trainval_wav.scp \
# 	--time-durs-file $list_dir/trainval_utt2dur \
# 	--train-list $list_dir/train_utt2attack \
# 	--val-list $list_dir/val_utt2attack \
# 	--class-file $list_dir/class_file \
# 	--audio_path_OracleAdvNoise $list_dir_OracleAdvNoise/trainval_wav.scp \
# 	--time_durs_file_OracleAdvNoise $list_dir/trainval_utt2dur \
# 	--train_list_OracleAdvNoise $list_dir_OracleAdvNoise/train_utt2attack \
# 	--val_list_OracleAdvNoise $list_dir_OracleAdvNoise/val_utt2attack \
# 	--class_file_OracleAdvNoise $list_dir/class_file \
# 	--batch-size $batch_size \
# 	--num-workers $num_workers \
# 	--grad-acc-steps $grad_acc_steps \
# 	--num-gpus $ngpu \
# 	--log-interval $log_interval \
# 	--exp-path $sign_nnet_dir $args
# fi

if [ $stage -le 1 ]; then
    echo "Train two-stream signature network on all attacks"
    mkdir -p $sign_nnet_dir/log
    $cuda_cmd --gpu $ngpu $sign_nnet_dir/log/train.log \
	hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
	torch-train-xvec-from-wav-two-stream.py  $sign_nnet_command --cfg $sign_nnet_config \
	--audio-path $list_dir/trainval_wav.scp \
	--time-durs-file $list_dir/trainval_utt2dur \
	--train-list $list_dir/train_utt2attack \
	--val-list $list_dir/val_utt2attack \
	--class-file $list_dir/class_file \
	--audio-path2 $list_dir_OracleAdvNoise/trainval_wav.scp \
	--time-durs-file2 $list_dir_OracleAdvNoise/trainval_utt2dur \
	--train-list2 $list_dir_OracleAdvNoise/train_utt2attack \
	--val-list2 $list_dir_OracleAdvNoise/val_utt2attack \
	--batch-size $batch_size \
	--num-workers $num_workers \
	--grad-acc-steps $grad_acc_steps \
	--num-gpus $ngpu \
	--log-interval $log_interval \
	--exp-path $sign_nnet_dir $args
fi

exit 


if [ $stage -le 2 ]; then
    echo "Extract signatures on the test set"
    mkdir -p $list_dir/test
    cp $list_dir/test_wav.scp $list_dir/test/wav.scp
    nj=100
    steps_xvec/extract_xvectors_from_wav.sh \
	--cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} --use-bin-vad false \
	--feat-config $feat_config \
	$sign_nnet $list_dir/test \
	$sign_dir/test
fi

proj_dir=$sign_dir/test/tsne_${attack_type_split_tag}
if [ $stage -le 3 ];then
    echo "Make TSNE plots on all test attacks"
    echo "Result will be left in $proj_idr"
    for p in 30 100 250
    do
	for e in 12 64
	do
	    proj_dir_i=$proj_dir/p${p}_e${e}
	    $train_cmd $proj_dir_i/train.log \
		hyp_utils/conda_env.sh steps_visual/proj-attack-tsne.py \
		--train-v-file scp:$sign_dir/test/xvector.scp \
		--train-list $list_dir/test_utt2attack \
		--pca-var-r 0.99 \
		--prob-plot 0.3 --lnorm --tsne.metric cosine --tsne.early-exaggeration $e --tsne.perplexity $p --tsne.init pca \
		--output-path $proj_dir_i &
	done
    done
    wait
fi

if [ $stage -le 4 ]; then
    echo "Eval signature network logits on test attacks"
    mkdir -p $list_dir/test
    nj=100
    steps_xvec/eval_xvec_logits_from_wav.sh \
	--cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} --use-bin-vad false \
	--feat-config $feat_config \
	$sign_nnet $list_dir/test \
	$logits_dir/test
fi

if [ $stage -le 5 ];then
    echo "Compute cofusion matrices"
    echo "Result is left in $logits_dir/test/eval_acc.log"
    $train_cmd $logits_dir/test/eval_acc.log \
        hyp_utils/conda_env.sh steps_backend/eval-classif-perf.py \
        --score-file scp:$logits_dir/test/logits.scp \
        --key-file $list_dir/test_utt2attack \
	--class-file $list_dir/class_file
fi


exit
