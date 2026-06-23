#!/bin/bash

# script to train LLM







export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/scratch/$USER/labs/trial-lab03/nanochat" # fix this
mkdir -p $NANOCHAT_BASE_DIR

export DEPTH=6
export DEVICE_BATCH_SIZE=8



# ---------------------------------------------
# python venv setup with uv

# install uv if not previously installed 
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# ---------------------------------------------
# wandb setup 
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# ---------------------------------------------

# start with empty report with header and timestamp 
python -m nanochat.report reset


# ---------------------------------------------

# Tokenizer
python -m nanochat.dataset -n -1 # full dataset

python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!

# train tokenizer 
python -m scripts.tok_train --vocab-size 8192

# evaluate tokenizer 
python -m scripts.tok_eval

# ---------------------------------------------

# base model (pretraining)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# d24 model (slightly undertrained to beat GPT-2 => decrease data:params ratio from compute optimal 10.5 (default) to 8)
python -m torch.distributed.run --standalone --nproc_per_node=1 -m scripts.base_train -- --depth=$DEPTH --target-param-data-ratio=8 --device-batch-size=$DEVICE_BATCH_SIZE --fp8 --run=$WANDB_RUN

python -m torch.distributed.run --standalone --nproc_per_node=1 -m scripts.base_eval -- --device-batch-size=$DEVICE_BATCH_SIZE
# ---------------------------------------------
# SFT 
curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run SFT and eval the model
python -m torch.distributed.run --standalone --nproc_per_node=1 -m scripts.chat_sft -- --device-batch-size=$DEVICE_BATCH_SIZE --run=$WANDB_RUN

python -m torch.distributed.run --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i sft
# ---------------------------------------------
# generate full report 
python -m nanochat.report generate
