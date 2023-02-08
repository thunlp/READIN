#!/bin/bash
#SBATCH -p rtx2080
#SBATCH -G 1
#SBATCH --job-name afqmc_mb


# Model
model_path="chinese-roberta-wwm-ext"
model_path="rbt3"
lr="5e-5"
# model_path="chinese-macbert-base"
# lr="2e-5"


# Experiment Setup
seed="1"
train_data_name="afqmc_balanced"
# train_data_dir="../data/afqmc_balanced_da_noise/phonetic_50"
# train_data_dir="../data/realtypo/${train_data_name}"
# test_data_dir="../data/realtypo/afqmc_unbalanced"
train_data_dir="/home/zhangzhengyan/noisy_data/readin/afqmc_balanced"
test_data_dir="/home/zhangzhengyan/noisy_data/readin/afqmc_balanced"
output_dir="results/${train_data_name}/${model_path}_lr${lr}_seed${seed}"
ckpt_dir="${output_dir}/checkpoint-280"

# Global args
cmd="python3 train_afqmc_bert.py"
cmd+=" --model_path hfl/$model_path"
cmd+=" --output_dir $output_dir"
cmd+=" --train_dir $train_data_dir"
cmd+=" --test_dir $test_data_dir"
# Train will only resume from last checkpoint, then stop training (because all
# epochs have passed)
cmd+=" --mode train_test"
cmd+=" --num_epochs 4"
cmd+=" --batch_size 8"
cmd+=" --grad_acc_steps 32"
cmd+=" --lr $lr"
cmd+=" --seed $seed"
cmd+=" --log_interval 40"
# cmd+=" --num_examples 10000"
# cmd+=" --resume_from_ckpt ${ckpt_dir}"

logfile="$output_dir/test.log"
mkdir -p $output_dir
echo $cmd
echo ''
$cmd | tee $logfile
