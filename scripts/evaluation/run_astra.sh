#!/bin/bash
# Reproduce Astra experiments: RMSNorm, SiLU*Mul, MergeState
# Uses OpenAI o4-mini via Agents SDK, R=5 optimization rounds on H100
#
# Usage:
#   ./scripts/evaluation/run_astra.sh                    # run all 3 benchmarks
#   ./scripts/evaluation/run_astra.sh rmsnorm            # run only RMSNorm
#   ./scripts/evaluation/run_astra.sh silu               # run only SiLU*Mul
#   ./scripts/evaluation/run_astra.sh mergestate         # run only MergeState
#
# Model/parallelism via env vars:
#   CODE_GEN_MODEL="openrouter:minimax/MiniMax-M2.5" BEST_OF_N=5 ./scripts/evaluation/run_astra.sh rmsnorm
#
# Requires: leo-astra-nvidia:latest on gilgamesh, OPENAI_API_KEY in .env

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MACHINE="gilgamesh"
IMAGE="leo-astra-nvidia:latest"
MAX_ITER=5

# Load API key from .env
if [ -f "$LEO_ROOT/.env" ]; then
    source "$LEO_ROOT/.env"
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set. Add it to $LEO_ROOT/.env"
    exit 1
fi

# Results directory
RESULTS_DIR="$LEO_ROOT/results/astra"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)

FOCUS_SHAPE="${2:-}"  # Optional: e.g., "512x16384"

# Model and parallelism configuration (env vars, defaults to current behavior)
CODE_GEN_MODEL="${CODE_GEN_MODEL:-o4-mini}"
STRATEGIST_MODEL="${STRATEGIST_MODEL:-o4-mini}"
BEST_OF_N="${BEST_OF_N:-1}"
MAX_DEBUG_RETRIES="${MAX_DEBUG_RETRIES:-2}"
BEAM_WIDTH="${BEAM_WIDTH:-1}"

run_experiment() {
    local name=$1
    local kernel_path=$2
    local compare_kind=$3
    local baseline_func=$4
    local export_func=$5

    local focus_flag=""
    if [ -n "$FOCUS_SHAPE" ]; then
        focus_flag="--focus-shape $FOCUS_SHAPE"
        name="${name}-focus${FOCUS_SHAPE}"
    fi

    # Build model/parallelism flags
    local model_flags=""
    if [ "$CODE_GEN_MODEL" != "o4-mini" ]; then
        model_flags="$model_flags --code-gen-model $CODE_GEN_MODEL"
        name="${name}-$(echo $CODE_GEN_MODEL | tr '/:' '-')"
    fi
    if [ "$STRATEGIST_MODEL" != "o4-mini" ]; then
        model_flags="$model_flags --strategist-model $STRATEGIST_MODEL"
    fi
    if [ "$BEST_OF_N" -gt 1 ] 2>/dev/null; then
        model_flags="$model_flags --best-of-n $BEST_OF_N"
        name="${name}-bon${BEST_OF_N}"
    fi
    if [ "$MAX_DEBUG_RETRIES" -gt 0 ] 2>/dev/null; then
        model_flags="$model_flags --max-debug-retries $MAX_DEBUG_RETRIES"
    fi
    if [ "$BEAM_WIDTH" -gt 1 ] 2>/dev/null; then
        model_flags="$model_flags --beam-width $BEAM_WIDTH"
        name="${name}-beam${BEAM_WIDTH}"
    fi

    # Build Docker env flags
    local env_flags="-e OPENAI_API_KEY=$OPENAI_API_KEY -e CUDA_VISIBLE_DEVICES=0"
    if [ -n "$OPENROUTER_API_KEY" ]; then
        env_flags="$env_flags -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY"
    fi

    local log_file="$RESULTS_DIR/${name}-${TIMESTAMP}.log"

    echo "============================================"
    echo " Astra: $name (R=$MAX_ITER on $MACHINE)"
    [ -n "$FOCUS_SHAPE" ] && echo " Focus shape: $FOCUS_SHAPE"
    [ "$CODE_GEN_MODEL" != "o4-mini" ] && echo " CodeGen model: $CODE_GEN_MODEL"
    [ "$BEST_OF_N" -gt 1 ] 2>/dev/null && echo " Best-of-N: $BEST_OF_N"
    [ "$BEAM_WIDTH" -gt 1 ] 2>/dev/null && echo " Beam width: $BEAM_WIDTH"
    echo " Log: $log_file"
    echo "============================================"

    ssh "$MACHINE" "docker run --rm --gpus all \
        $env_flags \
        $IMAGE \
        -c 'python3.11 cuda_kernel_optimizer_multi.py \
            --api-key \$OPENAI_API_KEY \
            --initial-kernel-path $kernel_path \
            --compare-kind $compare_kind \
            --baseline-func $baseline_func \
            --generated-export-func $export_func \
            --max-iterations $MAX_ITER \
            $focus_flag \
            $model_flags'" 2>&1 | tee "$log_file"

    echo ""
    echo " $name done. Log saved to $log_file"
    echo ""
}

# Experiment definitions
run_rmsnorm() {
    run_experiment "rmsnorm" \
        "test/rms/rms_v1.cu" \
        "rmsnorm" \
        "fused_add_rmsnorm" \
        "sgl_fused_add_rmsnorm"
}

run_silu() {
    run_experiment "silu" \
        "test/silu/silu_mul_v1.cu" \
        "silu" \
        "silu_and_mul" \
        "sgl_silu_mul"
}

run_mergestate() {
    run_experiment "mergestate" \
        "test/merge/merge_v1.cu" \
        "mergestate" \
        "merge_state" \
        "merge_state"
}

# Run selected or all experiments
case "${1:-all}" in
    rmsnorm)    run_rmsnorm ;;
    silu)       run_silu ;;
    mergestate) run_mergestate ;;
    all)
        run_rmsnorm
        run_silu
        run_mergestate
        echo "============================================"
        echo " All experiments complete."
        echo " Results in: $RESULTS_DIR"
        echo "============================================"
        ;;
    *)
        echo "Usage: $0 [rmsnorm|silu|mergestate|all]"
        exit 1
        ;;
esac
