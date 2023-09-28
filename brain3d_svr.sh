#/bin/bash -x
cat "$0" >&2 
model="flow_SNet3d1_256" #"flow_SNet3d1_4_noin" # "flow_UNet2d" #"flow_UNet2d"
loss="l2_loss"
tests_set="brain3d111_4_svr" #brain3d_reg" #"brain2d_reg" #"brainreg2d" #"brain2d_reg"  #"flyingchairs"
denoiser="nothing"
postfilt="nothing"
train_metrics="MSE"
valid_metrics="MSE"
tests_metrics="MSE"
batch_size=1
mode="--train"
seed=0
drop=0.000
alpha=1.000
positional=0

for dataset in $tests_set
do
    for mu in 0.000
    do
	gpu=0
	for theta in 0.000
	do
	    echo "(GPU $gpu) seed: $seed, launching batch_size: $batch_size, theta: $theta, mu: $mu, drop: $drop"
	    remarks="$dataset"_"$model"_"$loss"_bulktrans05_bulkrot45_trans10_rot20_250k_nobound_nosine_nomask_2stacks #unmasked_pe_trans_random_zoom #_no_skip
	    python train.py --trainee segment --dataset $dataset --denoiser $denoiser --postfilt $postfilt --mu $mu --theta $theta --alpha $alpha --batch_size $batch_size --valid_batch_size $batch_size --loss $loss --valid_metrics $valid_metrics --tests_metrics $tests_metrics --aux_loss $loss --network $model --padding 0 --max_steps 250000 --limit_train_batches 20000 --positional $positional --drop $drop --optim adam --lr_start 1e-4 --momentum 0.90 --decay 0.0000 --schedule poly --val_check_interval 1.0 --monitor val_loss --monitor_mode min --seed $seed --remarks $remarks $mode --direct > results/"$remarks".txt

	done

    done
done
