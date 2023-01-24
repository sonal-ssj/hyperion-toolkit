#!/bin/bash
# Copyright
#                2019   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
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

args=""
if [ "$resume" == "true" ];then
    args="--resume"
fi

if [ "$interactive" == "true" ];then
    export cuda_cmd=run.pl
fi

# sign_nnet_reldir=$spknet_name/$sign_nnet_name/$attack_type_split_tag"_noise_only"
# sign_nnet_dir=exp/sign_nnets/$sign_nnet_reldir
# sign_dir=exp/signatures/$sign_nnet_reldir
# logits_dir=exp/logits/$sign_nnet_reldir
# sign_nnet=$sign_nnet_dir/model_ep0020.pth

# sign_nnet_reldir=$spknet_name/$sign_nnet_name/$attack_type_split_tag"_noise_only"
# sign_nnet_dir=exp/sign_nnets/$sign_nnet_reldir
# sign_nnet_reldir_sign_with_denoiser=$spknet_name"_preprocess_denoiser_gan_with_denoiser"/$sign_nnet_name/$attack_type_split_tag"_noise_only"
# sign_dir=/export/b15/sjoshi/hyp_red_out/exp/signatures/$sign_nnet_reldir_sign_with_denoiser
# logits_dir=/export/b15/sjoshi/hyp_red_out/exp/logits/$sign_nnet_reldir_sign_with_denoiser
# sign_nnet=$sign_nnet_dir/model_ep0020.pth


# denoiser_model_path=exp/denoiser_model/BWE-2/133/14_0.0002_-0.0.pt # GAN Model trained by SK 
# denoiser_model_n_layers=8
# denoiser_model_load_string=G
# denoiser_return_noise=True



denoiser_root_dir=/export/c05/sjoshi/codes/hyperion-toolkit/egs/voxceleb/adv.v2/exp/denoiser_model
denoiser_model_project_name=2
denoiser_model_exp_name=133
denoiser_model_ckpt=14_0.0002_-0.0.pt

denoiser_model_name_predict='B' # Denoiser Model Predicts benign

denoiser_model_load_string=G
denoiser_model_G_num_speakers=1
denoiser_model_ctn_layer=8
denoiser_model_encoder_dim=128
denoiser_model_two_stream=False
denoiser_model_two_stream_use_benign=False


denoiser_model_path=${denoiser_root_dir}/BWE-${denoiser_model_project_name}/${denoiser_model_exp_name}/${denoiser_model_ckpt} # GAN Denoiser Model 
denoiser_model_prefix=BWE-${denoiser_model_project_name}_${denoiser_model_exp_name}_${denoiser_model_ckpt}_useBen-${denoiser_model_two_stream_use_benign}

list_dir=data/$attack_type_split_tag # Attacks list dir
sign_nnet_reldir=$spknet_name/$sign_nnet_name/$attack_type_split_tag"_noise_only" # Noise only model
sign_nnet_dir=exp/sign_nnets/$sign_nnet_reldir
echo "sign_nnet_dir="$sign_nnet_dir

sign_dir=exp/signatures/$sign_nnet_reldir_${denoiser_model_prefix}
logits_dir=exp/logits/$sign_nnet_reldir_${denoiser_model_prefix}
sign_nnet=$sign_nnet_dir/model_ep0020.pth

# Network Training
if [ $stage -le 1 ]; then
	echo "---- Stage 1 ----"
    echo "Use Trained signature network Oracle Adv Perturbation"
fi


if [ $stage -le 2 ]; then
	echo "---- Stage 2 ----"
    echo "Extract signatures on the test set"
    echo "No need to create new list  wav.scp files as its already there"
    # mkdir -p $list_dir/test
    # cp $list_dir/test_wav.scp $list_dir/test/wav.scp
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
	--denoiser_model_two_stream_use_benign $denoiser_model_two_stream_use_benign \
    $sign_nnet $list_dir/test \
	$sign_dir/test
fi



proj_dir=$sign_dir/test/tsne_${attack_type_split_tag}
if [ $stage -le 3 ];then
	echo "---- Stage 3 ----"
    echo "Make TSNE plots on all test attacks"
    echo "Result will be left in $proj_dir"
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
	echo "---- Stage 4 ----"
    echo "Eval signature network logits on test attacks"
    mkdir -p $list_dir/test
	#nj=100
    nj=50 # Reducing number of jobs because otherwise jobs go to error state in CLSP grid
    steps_xvec/eval_xvec_logits_from_wav_with_preproc_denoiser.sh \
	--cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} --use-bin-vad false \
	--feat-config $feat_config \
	--denoiser_model_path $denoiser_model_path \
    --denoiser_model_load_string $denoiser_model_load_string \
	--denoiser_model_G_num_speakers $denoiser_model_G_num_speakers \
	--denoiser_model_name_predict $denoiser_model_name_predict \
	--denoiser_model_ctn_layer $denoiser_model_ctn_layer \
	--denoiser_model_encoder_dim $denoiser_model_encoder_dim \
	--denoiser_model_two_stream $denoiser_model_two_stream \
	--denoiser_model_two_stream_use_benign $denoiser_model_two_stream_use_benign \
	$sign_nnet $list_dir/test \
	$logits_dir/test
fi

if [ $stage -le 5 ];then
	echo "---- Stage 5 ----"
    echo "Compute confusion matrices"
    echo "Result is left in $logits_dir/test/eval_acc.log"
    $train_cmd $logits_dir/test/eval_acc.log \
        hyp_utils/conda_env.sh steps_backend/eval-classif-perf.py \
        --score-file scp:$logits_dir/test/logits.scp \
        --key-file $list_dir/test_utt2attack \
	    --class-file $list_dir/class_file
fi


exit
