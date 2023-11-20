#/bin/bash -x
cat "$0" >&2 
model="flow_SNet3d1_32" #"flow_SNet3d1_4_noin" # "flow_UNet2d" #"flow_UNet2d"
loss="l2_loss"
denoiser="nothing"
postfilt="nothing"
train_metrics="MSE"
valid_metrics="MSE"
tests_metrics="MSE"
batch_size=1
mode="--train"
seed=0
log_every_n_steps=5
printf "hello"
max_epochs = 50 #issue with this, have to manually input number below instead of variable
dataset="feta3d1_4_svr_mf_clinical"
echo "(GPU $gpu) seed: $seed, launching batch_size: $batch_size, theta: $theta, mu: $mu, drop: $drop"
remarks="$dataset"_"$model"_"$loss"_bulktrans05_bulkrot45_trans10_rot20_250k_nobound_nosine_nomask_2stacks #unmasked_pe_trans_random_zoom #_no_skip
echo "hello1"
#python train.py --trainee segment --dataset $dataset  --batch_size $batch_size --valid_batch_size $batch_size --loss $loss --aux_loss $loss --network $model --max_steps 250000 --limit_train_batches 20000 --optim adam --lr_start 1e-4 --momentum 0.90 --decay 0.0000 --schedule poly --val_check_interval 1.0 --monitor val_loss --monitor_mode min --seed $seed --remarks $remarks $mode --direct --log_every_n_steps $log_every_n_steps --max_epochs 5
python model_test_fet_clin.py --trainee segment --dataset $dataset  --batch_size $batch_size --valid_batch_size $batch_size --loss $loss --aux_loss $loss --network $model --max_steps 250000 --limit_train_batches 20000 --optim adam --lr_start 1e-4 --momentum 0.90 --decay 0.0000 --schedule poly --val_check_interval 1.0 --monitor val_loss --monitor_mode min --seed $seed --remarks $remarks $mode --direct --log_every_n_steps $log_every_n_steps 
#> results/"$remarks".txt
# --batch_size $batch_size, --max_steps 250000 
echo "hello2!"


