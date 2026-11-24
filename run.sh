#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

dataset="dbp"       # dbp (DBP15K), maea (ICEWS-WIKI)
mode="eval"         # train, eval
device="auto"       # auto, cpu, cuda:0
train_epochs=20
train_batch_size=20
eval_workers=100

mkdir -p logs
timestamp=$(date +"%Y-%m-%d-%H-%M-%S")
log_file="logs/${timestamp}_${dataset}_${mode}.log"

printf 'Running %s %s (%s)\n' "$dataset" "$mode" "$log_file"
"${MAEC_PYTHON:-python}" -u -m src.run \
    --dataset "$dataset" \
    --run_mode "$mode" \
    --device "$device" \
    --train_epochs "$train_epochs" \
    --train_batch_size "$train_batch_size" \
    --eval_workers "$eval_workers" > "$log_file" 2>&1
