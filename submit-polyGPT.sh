#!/bin/bash
#SBATCH --job-name=BWZ-polyGPT           # 作业名
#SBATCH --partition=A800                 # 队列名（分区）
#SBATCH -N 1                             # 节点数（机器数）
#SBATCH --ntasks-per-node=1              # 每个节点只跑1个任务（进程启动器）
#SBATCH --cpus-per-task=8                # 每个任务分配8个CPU核
#SBATCH --gres=gpu:a800:2                # 申请2张A800 GPU
#SBATCH --output=./OUT/demo_polyGPT.out        # 标准输出日志文件
#SBATCH --error=./OUT/demo_polyGPT.err         # 错误输出文件
#SBATCH -t 24:00:00                      # 最大运行时间

# -------- 环境准备 --------
source /share/home/u23514/apps/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

# -------- 关键设置：输出 & NCCL --------
export PYTHONUNBUFFERED=1    # 实时刷新 stdout（确保 .out 文件能看到 loss）
export NCCL_DEBUG=WARN       # 若有通信问题可设为 INFO
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4

# -------- 启动训练 --------
# torchrun 是多 GPU 启动的正确方式
torchrun --standalone --nproc_per_node=2 ./demo_polyGPT.py
