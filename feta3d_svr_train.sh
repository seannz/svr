#!/bin/bash -x
model="flow_SNet3d0_384" # "flow_UNet2d" #"flow_UNet2d"
loss="l22_loss_affine_invariant"
dataset="feta3d0_4_svr"
train_metrics="MeanL22LossInvariant MeanL21LossInvariant"
valid_metrics="MeanL22LossInvariant MeanL21LossInvariant"
tests_metrics="MeanL22LossInvariant MeanL21LossInvariant"
batch_size=1
mode="--train" # --load $load" # --resume --load $load"
seed=0

remarks="$dataset"_"$model"_"$loss"_bulktrans0_bulkrot180_trans10_rot20_250k_cvpr #unmasked_pe_trans_random_zoom #_no_skip
python train.py --trainee segment --dataset $dataset --batch_size $batch_size --valid_batch_size $batch_size --loss $loss --valid_metrics $valid_metrics --tests_metrics $tests_metrics --network $model --max_steps 250000 --limit_train_batches 20000 --optim adam --lr_start 1e-4 --momentum 0.90 --decay 0.0000 --schedule poly --val_check_interval 1.0 --monitor val_loss --monitor_mode min --seed $seed --remarks $remarks $mode | tee results/$remarks.txt
