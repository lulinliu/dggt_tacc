#!/usr/bin/env bash
set -euo pipefail

cd /scratch/10102/hh29499/longtail_train/dggt_tacc

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29500   # 任意空闲端口
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_NODEID
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS="${MAX_JOBS:-1}"
export CUDA_HOME=/opt/apps/cuda/12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="$CUDA_HOME/include:${CPATH:-}"

unset NVCC_PREPEND_FLAGS

export TORCH_EXTENSIONS_DIR=/scratch/10102/hh29499/torch_extensions_cache
mkdir -p "$TORCH_EXTENSIONS_DIR"

LOG_DIR="/scratch/10102/hh29499/longtail_train/dggt_tacc/logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/train_${SLURM_JOB_ID:-manual}_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"

torchrun \
  --nnodes=$NNODES \
  --nproc_per_node 1 \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train.py \
  --image_dir /scratch/10102/hh29499/longtail_train/train_data/train_data_dggt_v3_all/train_data_dggt_v3 \
  --log_dir /scratch/10102/hh29499/longtail_train/dggt_tacc/logs \
  --ckpt_path /scratch/10102/hh29499/longtail_train/dggt_tacc/pretrained/model.pt \
  --input_views 3 \
  --sequence_length 4 \
  --max_epoch 5000 \
  --save_ckpt 50 \
  --no_train_dynamic_head \
  2>&1 | tee "$LOG"

