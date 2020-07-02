#Default configuration parameters for the experiment

#xvector training 
nnet_data=voxceleb2cat_combined
nnet_type=efficientnet-b4
batch_size_1gpu=12
eff_batch_size=512 # effective batch size
min_chunk=400
max_chunk=400
ipe=1
lr=0.01
dropout=0
embed_dim=256
s=30
margin_warmup=20
margin=0.3
se_r=4
effnet_opt="--in-channels 1 --in-kernel-size 3 --in-stride 1 --se-r $se_r --fix-stem-head --mbconv-strides 1 1 2 2 1 2 1"
opt_opt="--opt-optimizer adam --opt-lr $lr --opt-beta1 0.9 --opt-beta2 0.95 --opt-weight-decay 1e-5 --opt-amsgrad --use-amp"
lrs_opt="--lrsch-lrsch-type exp_lr --lrsch-decay-rate 0.5 --lrsch-decay-steps 8000 --lrsch-hold-steps 40000 --lrsch-min-lr 1e-5 --lrsch-warmup-steps 1000 --lrsch-update-lr-on-opt-step"
nnet_name=${nnet_type}_is1_mbs1122121_ser${se_r}_fixsh_e${embed_dim}_arcs${s}m${margin}_do${dropout}_adam_lr${lr}_b${eff_batch_size}_amp.v1
nnet_num_epochs=200
num_augs=5
nnet_dir=exp/xvector_nnets/$nnet_name
nnet=$nnet_dir/model_ep0070.pth
#nnet=$nnet_dir/model_ep0118.pth

#xvector finetuning
ft_batch_size_1gpu=12
ft_eff_batch_size=512 # effective batch size
ft_min_chunk=400
ft_max_chunk=400
ft_ipe=1
ft_lr=0.005
ft_nnet_num_epochs=40
ft_margin_warmup=3
#ft_opt_opt="--opt-optimizer sgd --opt-lr $ft_lr --opt-momentum 0.9 --opt-weight-decay 1e-5 --use-amp --var-batch-size"
ft_opt_opt="--opt-optimizer sgd --opt-lr $ft_lr --opt-momentum 0.9 --opt-weight-decay 1e-5 --use-amp"
#ft_lrs_opt="--lrsch-lrsch-type exp_lr --lrsch-decay-rate 0.5 --lrsch-decay-steps 8000 --lrsch-hold-steps 40000 --lrsch-min-lr 1e-5 --lrsch-warmup-steps 100 --lrsch-update-lr-on-opt-step"
ft_lrs_opt="--lrsch-lrsch-type cos_lr --lrsch-t 2500 --lrsch-t-mul 2 --lrsch-warm-restarts --lrsch-gamma 0.75 --lrsch-min-lr 1e-4 --lrsch-warmup-steps 100 --lrsch-update-lr-on-opt-step"
ft_nnet_name=${nnet_name}.ft_${ft_min_chunk}_${ft_max_chunk}_sgdcos_lr${ft_lr}_b${ft_eff_batch_size}_amp.v1
# ft_opt_opt="--opt-optimizer adam --opt-lr $ft_lr --opt-beta1 0.9 --opt-beta2 0.95 --opt-weight-decay 1e-5 --opt-amsgrad --use-amp"
# ft_lrs_opt="--lrsch-lrsch-type exp_lr --lrsch-decay-rate 0.5 --lrsch-decay-steps 8000 --lrsch-hold-steps 40000 --lrsch-min-lr 1e-5 --lrsch-warmup-steps 1000 --lrsch-update-lr-on-opt-step"
# ft_nnet_name=${nnet_name}.ft_${ft_min_chunk}_${ft_max_chunk}_adam_lr${ft_lr}_b${ft_eff_batch_size}_amp.v2
ft_nnet_dir=exp/xvector_nnets/$ft_nnet_name
ft_nnet=$ft_nnet_dir/model_ep0014.pth


# #xvector finetuning
# ft_batch_size_1gpu=4
# ft_eff_batch_size=128 # effective batch size
# ft_min_chunk=400
# ft_max_chunk=6000
# ft_ipe=0.25
# ft_lr=0.01
# ft_nnet_num_epochs=40
# ft_margin_warmup=3
# ft_opt_opt="--opt-optimizer sgd --opt-lr $ft_lr --opt-momentum 0.9 --opt-weight-decay 1e-5 --use-amp --var-batch-size"
# #ft_lrs_opt="--lrsch-lrsch-type exp_lr --lrsch-decay-rate 0.5 --lrsch-decay-steps 8000 --lrsch-hold-steps 40000 --lrsch-min-lr 1e-5 --lrsch-warmup-steps 100 --lrsch-update-lr-on-opt-step"
# ft_lrs_opt="--lrsch-lrsch-type cos_lr --lrsch-t 2500 --lrsch-t-mul 2 --lrsch-warm-restarts --lrsch-gamma 0.75 --lrsch-min-lr 1e-4 --lrsch-warmup-steps 100 --lrsch-update-lr-on-opt-step"
# ft_nnet_name=${nnet_name}.ft_${ft_min_chunk}_${ft_max_chunk}_sgdcos_lr${ft_lr}_b${ft_eff_batch_size}_amp.v2
# # ft_opt_opt="--opt-optimizer adam --opt-lr $ft_lr --opt-beta1 0.9 --opt-beta2 0.95 --opt-weight-decay 1e-5 --opt-amsgrad --use-amp"
# # ft_lrs_opt="--lrsch-lrsch-type exp_lr --lrsch-decay-rate 0.5 --lrsch-decay-steps 8000 --lrsch-hold-steps 40000 --lrsch-min-lr 1e-5 --lrsch-warmup-steps 1000 --lrsch-update-lr-on-opt-step"
# # ft_nnet_name=${nnet_name}.ft_${ft_min_chunk}_${ft_max_chunk}_adam_lr${ft_lr}_b${ft_eff_batch_size}_amp.v2
# ft_nnet_dir=exp/xvector_nnets/$ft_nnet_name
# ft_nnet=$ft_nnet_dir/model_ep0007.pth


#back-end
lda_dim=200
plda_y_dim=150
plda_z_dim=200

plda_data=voxceleb2cat_combined
plda_type=splda
