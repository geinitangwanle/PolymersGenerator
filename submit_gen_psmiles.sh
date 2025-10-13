#!/bin/bash
#SBATCH --job-name=gen-psmiles          # 任务名
#SBATCH --partition=A800                # 队列名（A800分区）
#SBATCH -N 1                            # 使用1个节点
#SBATCH --ntasks-per-node=1             # 每节点一个任务
#SBATCH --cpus-per-task=4               # 每任务4个CPU核
#SBATCH --gres=gpu:a800:1               # 申请1张A800 GPU
#SBATCH --output=./OUT/gen_psmiles.out        # 标准输出日志文件
#SBATCH --error=./OUT/gen_psmiles.err         # 错误输出文件
#SBATCH -t 02:00:00                     # 最长运行2小时，可调整

# ==== 环境准备 ====
# 1. 激活conda环境
source /share/home/u23514/apps/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

# 2. 设置实时输出（让 .out 文件实时显示）
export PYTHONUNBUFFERED=1

# 3. 启动生成脚本
python gen_psmiles.py \
  --ckpt psmiles-gpt \
  --inst "target_Tg=400.0" \
  --n 16 \
  --no_rdkit \
  --out results/gen_psmiles-1.csv
