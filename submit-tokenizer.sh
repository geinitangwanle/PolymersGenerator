#!/bin/bash
#SBATCH --job-name=BWZ-psmiles_tokenizer           # 作业名
#SBATCH --partition=A800                 # 队列名（分区）
#SBATCH -N 1                             # 节点数（机器数）
#SBATCH --ntasks-per-node=1              # 每个节点只跑1个任务（进程启动器）
#SBATCH --cpus-per-task=4                # 每个任务分配4个CPU核
#SBATCH --gres=gpu:a800:1                # 申请1张A800 GPU
#SBATCH --output=./OUT/psmiles_tokenizer.out        # 标准输出日志文件
#SBATCH --error=./OUT/psmiles_tokenizer.err         # 错误输出文件
#SBATCH -t 24:00:00                      # 最大运行时间

source /share/home/u23514/apps/miniconda3/etc/profile.d/conda.sh
conda activate pytorch
export PYTHONUNBUFFERED=1

python psmiles_tokenizer.py
