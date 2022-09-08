#!/bin/bash
# Copyright
#                2019   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#

# Edits by Sonal Joshi
# Edits to script `run_024_threat_model_classif_allknown.sh` to include
# (1) Denoiser while training signature extractor
# (2) Have trackable model names 
# (3) Add additional parameter to  exisitng scripts `denoiser_model_name_predict`

. ./cmd.sh
. ./path.sh
set -e

stage=1
ngpu=1
config_file=default_config.sh
resume=false
interactive=false
num_workers=4
xvec_use_gpu=false
xvec_chunk_length=12800

# Sonal commented because file not found
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

args=""
if [ "$resume" == "true" ];then
    args="--resume"
fi

if [ "$interactive" == "true" ];then
    export cuda_cmd=run.pl
fi

denoiser_root_dir=/export/b05/sjoshi/codes/train_denoiser/models
denoiser_model_project_name=2
denoiser_model_exp_name=1001
denoiser_model_ckpt=14_0.0006_0.0.pt

denoiser_model_name_predict='P' # Predicts adv perturb

denoiser_model_load_string=G
denoiser_model_G_num_speakers=1
denoiser_model_ctn_layer=8
denoiser_model_encoder_dim=128
denoiser_model_two_stream=False
denoiser_model_two_stream_use_benign=False


denoiser_model_path=${denoiser_root_dir}/BWE-${denoiser_model_project_name}/${denoiser_model_exp_name}/${denoiser_model_ckpt} # GAN Denoiser Model 
denoiser_model_prefix=BWE-${denoiser_model_project_name}_${denoiser_model_exp_name}_${denoiser_model_ckpt}_useBen-${denoiser_model_two_stream_use_benign}


sign_nnet_reldir=$spknet_name_${denoiser_model_prefix}/${sign_nnet_name}/${attack_type_split_tag}
sign_nnet_dir=exp/sign_nnets/$sign_nnet_reldir
sign_dir=exp/signatures/$sign_nnet_reldir
logits_dir=exp/logits/$sign_nnet_reldir
sign_nnet=$sign_nnet_dir/model_ep0020.pth

#torch-train-xvec-from-wav-with-preprocess-denoiser.py  \

	# --denoiser_model_path $denoiser_model_path \
    # --denoiser_model_load_string $denoiser_model_load_string \
	# --denoiser_model_G_num_speakers $denoiser_model_G_num_speakers \
	# --denoiser_model_name_predict $denoiser_model_name_predict \
	# --denoiser_model_ctn_layer $denoiser_model_ctn_layer \
	# --denoiser_model_encoder_dim $denoiser_model_encoder_dim \
	# --denoiser_model_two_stream $denoiser_model_two_stream \
	# --denoiser_model_two_stream_use_benign $denoiser_model_two_stream_use_benign \

# Network Training
if [ $stage -le 1 ]; then
    echo "Train signature network on all attacks (use denoiser)"
	echo $denoiser_model_prefix
    mkdir -p $sign_nnet_dir/log
    $cuda_cmd --gpu $ngpu $sign_nnet_dir/log/train.log \
	hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
	torch-train-xvec-from-wav.py  \
	$sign_nnet_command --cfg $sign_nnet_config \
	--audio-path $list_dir/trainval_wav.scp \
	--time-durs-file $list_dir/trainval_utt2dur \
	--train-list $list_dir/train_utt2attack \
	--val-list $list_dir/val_utt2attack \
	--class-file $list_dir/class_file \
	--batch-size $batch_size \
	--num-workers $num_workers \
	--grad-acc-steps $grad_acc_steps \
	--num-gpus $ngpu \
	--log-interval $log_interval \
	--exp-path $sign_nnet_dir \
	$args
fi

if [ $stage -le 2 ]; then
    echo "Extract signatures on the test set"
    mkdir -p $list_dir/test
    cp $list_dir/test_wav.scp $list_dir/test/wav.scp
	#nj=100
    nj=50 # Reducing number of jobs because otherwise jobs go to error state in CLSP grid
    steps_xvec/extract_xvectors_from_wav_with_preproc_denoiser.sh \
	--cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} --use-bin-vad false \
	--feat-config $feat_config \
	--denoiser_model_path $denoiser_model_path \
    --denoiser_model_load_string $denoiser_model_load_string \
	--denoiser_model_G_num_speakers $denoiser_model_G_num_speakers \
	--denoiser_model_name_predict $denoiser_model_name_predict \
	--denoiser_model_ctn_layer $denoiser_model_ctn_layer \
	--denoiser_model_encoder_dim $denoiser_model_encoder_dim \
	--denoiser_model_two_stream $denoiser_model_two_stream \
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
	#nj=100
    nj=50 # Reducing number of jobs because otherwise jobs go to error state in CLSP grid
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
