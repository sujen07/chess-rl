#!/usr/bin/env bash
# Architecture sweep: identical supervised runs (data, seed, schedule) that
# differ only in the network. Results land in experiments/arch_sweep/<name>/.
# Usage: run.sh [python]   then   python summarize.py
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_DIR="$(cd "$HERE/../.." && pwd)"
PY="${1:-python}"

COMMON=(
    --max-positions 4000000 --val-positions 100000
    --num-epochs 2 --train-batch-size 1024
    --lr 1e-3 --min-lr 1e-5 --warmup-steps 500 --weight-decay 1e-4
    --value-coef 1.0 --value-head-channels 32 --value-hidden 256
    --log-every 1000
)

run() {
    local name=$1 seed=$2; shift 2
    local out="$HERE/$name"
    if [[ -f "$out/model.pt" ]]; then echo "skip $name (done)"; return; fi
    mkdir -p "$out"
    echo "=== $name ($(date +%T)) ==="
    (cd "$RL_DIR" && "$PY" train_supervised.py "${COMMON[@]}" --seed "$seed" "$@" \
        --checkpoint-dir "$out" --model-out "$out/model.pt" \
        --plot-path "$out/loss_curve.png") 2>&1 | tee "$out/train.log"
}

run baseline         0 --channels 64 --num-blocks 6
run baseline_seed1   1 --channels 64 --num-blocks 6
run se               0 --channels 64 --num-blocks 6 --se-ratio 4
run transformer      0 --arch transformer --channels 128 --attention-layers 4 --attention-heads 4
run transformer_bias 0 --arch transformer --channels 128 --attention-layers 4 --attention-heads 4 --attention-bias
run hybrid           0 --channels 64 --num-blocks 4 --se-ratio 4 --attention-layers 2 --attention-heads 4
echo "=== sweep done ($(date +%T)) ==="
