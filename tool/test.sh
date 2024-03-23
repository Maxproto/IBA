#!/bin/sh

## uncomment for slurm
##SBATCH -p q3090-4
##SBATCH --gres=gpu:rtx3090:4
##SBATCH --cpus-per-task=16
##SBATCH -t 48:00:00
##SBATCH --mail-type=ALL,TIME_LIMIT_80

export PYTHONPATH=./
eval "$(conda shell.bash hook)"
conda activate base
PYTHON=python

dataset=$1
exp_name=$2
exp_dir=exp/${dataset}/${exp_name}
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${result_dir}
cp tool/test.sh tool/test.py ${config} ${exp_dir}

export PYTHONPATH=./
$PYTHON -u ${exp_dir}/test.py \
  --config=${config} \
  2>&1 | tee ${result_dir}/test-$now.log
