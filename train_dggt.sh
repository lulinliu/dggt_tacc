#!/usr/bin/env bash
set -euo pipefail

cd /scratch/10102/hh29499/longtail_train/dggt_tacc


MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)}"
MASTER_PORT="${MASTER_PORT:-29501}"
NNODES="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-16}}"
NODE_RANK="${SLURM_NODEID:?SLURM_NODEID is not set}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.8}"

LOG_DIR="/scratch/10102/hh29499/longtail_train/dggt_tacc/logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/train_${SLURM_JOB_ID:-manual}_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"

echo "[torchrun] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} NNODES=${NNODES} NODE_RANK=${NODE_RANK}"

TORCHRUN_CMD=(
  torchrun
  --nnodes="${NNODES}"
  --nproc_per_node=1
  --node_rank="${NODE_RANK}"
  --master_addr="${MASTER_ADDR}"
  --master_port="${MASTER_PORT}"
  train.py
  --image_dir /scratch/10102/hh29499/longtail_train/train_data/train_data_dggt_v3_all/train_data_dggt_v3
  --log_dir /scratch/10102/hh29499/longtail_train/dggt_tacc/logs
  --ckpt_path /scratch/10102/hh29499/longtail_train/dggt_tacc/pretrained/model.pt
  --input_views 3
  --sequence_length 4
  --max_epoch 5000
  --save_ckpt 50
  --no_train_dynamic_head
)

