#!/bin/bash
#SBATCH -J train_dggt
#SBATCH -A ASC26009
#SBATCH -p gh
#SBATCH -N 16                # 节点数；按需改
#SBATCH --ntasks-per-node=1   # 每节点启动1个任务（每节点1 GPU）
#SBATCH -t 48:00:00
#SBATCH -o /scratch/10102/hh29499/train_dggt/%x_%j.out
#SBATCH -e /scratch/10102/hh29499/train_dggt/%x_%j.err

set -euo pipefail

srun --export=ALL \
  bash /scratch/10102/hh29499/longtail_train/dggt_tacc/train_dggt.sh
