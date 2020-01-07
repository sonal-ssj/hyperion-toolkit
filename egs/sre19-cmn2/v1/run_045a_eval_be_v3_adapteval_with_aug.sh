#!/bin/bash
# Copyright       2018   Johns Hopkins University (Author: Jesus Villalba)
#                
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

#hyperparam tuned with F-TDNN 17M (3a) from SRE18
#spk det back-end
#spk det back-end
lda_tel_dim=150
lda_vid_dim=200
ncoh_tel=400

w_mu1=1
w_B1=0.1
w_W1=0.6
w_mu2=1
w_B2=0.4
w_W2=0.1
num_spks=500
coral_mu=0.5
coral_T=0.75

plda_tel_y_dim=150
plda_tel_z_dim=150

plda_tel_type=splda
coh_tel_data=sre18_dev_unlabeled

. parse_options.sh || exit 1;
. $config_file
. datapath.sh 

plda_tel_data=sre_tel_combined_noreverb

plda_tel_label=${plda_tel_type}y${plda_tel_y_dim}_adapt_v3evalaug_coral_mu${coral_mu}T${coral_T}_a1_mu${w_mu1}B${w_B1}W${w_W1}_a2_M${num_spks}_mu${w_mu2}B${w_B2}W${w_W2}
be_tel_name=lda${lda_tel_dim}_${plda_tel_label}_${plda_tel_data}

xvector_dir=exp/xvectors/$nnet_name
be_tel_dir=exp/be/$nnet_name/$be_tel_name
score_dir=exp/scores/$nnet_name/${be_tel_name}
score_plda_dir=$score_dir/plda

if [ ! -d scoring_software/sre19-cmn2 ];then
    local/download_sre19cmn2_scoring_tool.sh 
fi

if [ $stage -le 1 ]; then

    echo "Train BE"
    steps_be/train_tel_be_v3.sh --cmd "$train_cmd" \
    	--lda_dim $lda_tel_dim \
    	--plda_type $plda_tel_type \
    	--y_dim $plda_tel_y_dim --z_dim $plda_tel_z_dim \
    	--w_mu1 $w_mu1 --w_B1 $w_B1 --w_W1 $w_W1 \
    	--w_mu2 $w_mu2 --w_B2 $w_B2 --w_W2 $w_W2 --num_spks_unlab $num_spks \
	--w_coral_mu $coral_mu --w_coral_T $coral_T \
    	$xvector_dir/$plda_tel_data/xvector.scp \
    	data/$plda_tel_data \
	$xvector_dir/sre18_train_eval_cmn2_combined_noreverb/xvector.scp \
	data/sre18_train_eval_cmn2_combined_noreverb \
	$xvector_dir/sre18_dev_unlabeled/xvector.scp \
    	$sre18_dev_meta $be_tel_dir &
    
    wait
fi

if [ $stage -le 2 ]; then

    #SRE18
    echo "eval SRE18 without S-Norm"
    steps_be/eval_tel_be_v1.sh --cmd "$train_cmd" --plda_type $plda_tel_type \
			       $sre18_dev_trials_cmn2 \
			       data/sre18_dev_enroll_cmn2/utt2spk \
			       $xvector_dir/sre18_dev_cmn2/xvector.scp \
			       $be_tel_dir/lda_lnorm_adapt.h5 \
			       $be_tel_dir/plda_adapt2.h5 \
			       $score_plda_dir/sre18_dev_cmn2_scores &

    steps_be/eval_tel_be_v1.sh --cmd "$train_cmd" --plda_type $plda_tel_type \
			       $sre18_eval_trials_cmn2 \
			       data/sre18_eval_enroll_cmn2/utt2spk \
			       $xvector_dir/sre18_eval_cmn2/xvector.scp \
			       $be_tel_dir/lda_lnorm_adapt.h5 \
			       $be_tel_dir/plda_adapt2.h5 \
			       $score_plda_dir/sre18_eval_cmn2_scores &

    echo "eval SRE19 without S-Norm"
    steps_be/eval_tel_be_v1.sh --cmd "$train_cmd" --plda_type $plda_tel_type \
			       $sre19_eval_trials_cmn2 \
			       data/sre19_eval_enroll_cmn2/utt2spk \
			       $xvector_dir/sre19_eval_cmn2/xvector.scp \
			       $be_tel_dir/lda_lnorm_adapt.h5 \
			       $be_tel_dir/plda_adapt2.h5 \
			       $score_plda_dir/sre19_eval_cmn2_scores &

    wait
    local/score_sre18cmn2.sh data/sre18_dev_test_cmn2 dev $score_plda_dir
    local/score_sre18cmn2.sh data/sre18_eval_test_cmn2 eval $score_plda_dir
    local/score_sre19cmn2.sh data/sre19_eval_test_cmn2 ${score_plda_dir}

fi

if [ $stage -le 3 ];then
    local/calibrate_sre19cmn2_v1dev.sh --cmd "$train_cmd" $score_plda_dir
    local/score_sre18cmn2.sh data/sre18_dev_test_cmn2 dev ${score_plda_dir}_cal_v1dev
    local/score_sre18cmn2.sh data/sre18_eval_test_cmn2 eval ${score_plda_dir}_cal_v1dev
    local/score_sre19cmn2.sh data/sre19_eval_test_cmn2 ${score_plda_dir}_cal_v1dev

    #local/make_sre19cmn2_sub.sh $sre19cmn2_eval_root ${score_plda_dir}_cal_v1dev/sre19_eval_cmn2_scores
fi

score_plda_dir=$score_dir/plda_snorm

if [ $stage -le 4 ]; then

    #SRE18
    echo "SRE18 S-Norm"
    steps_be/eval_tel_be_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_tel_type --ncoh $ncoh_tel \
				     $sre18_dev_trials_cmn2 \
				     data/sre18_dev_enroll_cmn2/utt2spk \
				     $xvector_dir/sre18_dev_cmn2/xvector.scp \
				     data/${coh_tel_data}/utt2spk \
				     $xvector_dir/${coh_tel_data}/xvector.scp \
				     $be_tel_dir/lda_lnorm_adapt.h5 \
				     $be_tel_dir/plda_adapt2.h5 \
				     $score_plda_dir/sre18_dev_cmn2_scores &

    steps_be/eval_tel_be_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_tel_type --ncoh $ncoh_tel \
				     $sre18_eval_trials_cmn2 \
				     data/sre18_eval_enroll_cmn2/utt2spk \
				     $xvector_dir/sre18_eval_cmn2/xvector.scp \
				     data/${coh_tel_data}/utt2spk \
				     $xvector_dir/${coh_tel_data}/xvector.scp \
				     $be_tel_dir/lda_lnorm_adapt.h5 \
				     $be_tel_dir/plda_adapt2.h5 \
				     $score_plda_dir/sre18_eval_cmn2_scores &

    echo "SRE19 S-Norm"
    steps_be/eval_tel_be_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_tel_type --ncoh $ncoh_tel \
				     $sre19_eval_trials_cmn2 \
				     data/sre19_eval_enroll_cmn2/utt2spk \
				     $xvector_dir/sre19_eval_cmn2/xvector.scp \
				     data/${coh_tel_data}/utt2spk \
				     $xvector_dir/${coh_tel_data}/xvector.scp \
				     $be_tel_dir/lda_lnorm_adapt.h5 \
				     $be_tel_dir/plda_adapt2.h5 \
				     $score_plda_dir/sre19_eval_cmn2_scores &

    wait
    local/score_sre18cmn2.sh data/sre18_dev_test_cmn2 dev $score_plda_dir
    local/score_sre18cmn2.sh data/sre18_eval_test_cmn2 eval $score_plda_dir
    local/score_sre19cmn2.sh data/sre19_eval_test_cmn2 ${score_plda_dir}

fi

if [ $stage -le 5 ];then
    local/calibrate_sre19cmn2_v1dev.sh --cmd "$train_cmd" $score_plda_dir
    local/score_sre18cmn2.sh data/sre18_dev_test_cmn2 dev ${score_plda_dir}_cal_v1dev
    local/score_sre18cmn2.sh data/sre18_eval_test_cmn2 eval ${score_plda_dir}_cal_v1dev
    local/score_sre19cmn2.sh data/sre19_eval_test_cmn2 ${score_plda_dir}_cal_v1dev
    #local/make_sre19cmn2_sub.sh $sre19cmn2_eval_root ${score_plda_dir}_cal_v1dev/sre19_eval_cmn2_scores
fi

    
exit
