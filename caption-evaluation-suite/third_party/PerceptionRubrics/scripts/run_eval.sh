#!/usr/bin/env bash
# End-to-end evaluation: predict captions then judge against rubrics.
#
# All parameters can be overridden via environment variables, e.g.:
#   PREDICT_MODEL=gpt-4o WORKERS=16 bash scripts/run_eval.sh
#
# Usage:
#   bash scripts/run_eval.sh

set -euo pipefail

# ============================================================
# Parameters — all support env-var override via ${VAR:-default}
# ============================================================

# --- Predict model (the VLM being evaluated) ---
PREDICT_API_KEY="${PREDICT_API_KEY:-${OPENAI_API_KEY:-}}"
PREDICT_BASE_URL="${PREDICT_BASE_URL:-${OPENAI_BASE_URL:-}}"
PREDICT_MODEL="${PREDICT_MODEL:-}"
PREDICT_MAX_TOKENS="${PREDICT_MAX_TOKENS:-32768}"
PREDICT_TEMPERATURE="${PREDICT_TEMPERATURE:-1.0}"
PREDICT_TOP_P="${PREDICT_TOP_P:-1}"
PREDICT_TOP_K="${PREDICT_TOP_K:--1}"
PREDICT_REASONING_EFFORT="${PREDICT_REASONING_EFFORT:-high}"

# --- Judge model (recommended: gpt-oss-120b) ---
JUDGE_API_KEY="${JUDGE_API_KEY:-${OPENAI_API_KEY:-}}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-${OPENAI_BASE_URL:-}}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-oss-120b}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-32768}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0}"
JUDGE_REASONING_EFFORT="${JUDGE_REASONING_EFFORT:-medium}"

# --- Dataset ---
DATASET="${DATASET:-M1chaelPeng/PerceptionRubrics}"
SPLIT="${SPLIT:-test}"

# --- Output & Runtime ---
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
LIMIT="${LIMIT:-}"              # leave empty to eval all; set a number for quick test
WORKERS="${WORKERS:-64}"        # concurrent API requests (shared by predict & judge)

# ============================================================
# Validation
# ============================================================
missing=()
for var in PREDICT_MODEL JUDGE_MODEL DATASET; do
    if [ -z "${!var}" ]; then
        missing+=("$var")
    fi
done
if [ ${#missing[@]} -gt 0 ]; then
    echo "Error: missing parameters: ${missing[*]}" >&2
    exit 1
fi

# ============================================================
# Paths — model-specific output directory
# ============================================================
MODEL_DIR_NAME="${PREDICT_MODEL//\//_}"
MODEL_OUTPUT_DIR="$OUTPUT_DIR/$MODEL_DIR_NAME"
mkdir -p "$MODEL_OUTPUT_DIR"

PREDICTIONS="$MODEL_OUTPUT_DIR/predictions.jsonl"
RESULTS="$MODEL_OUTPUT_DIR/results.jsonl"

LIMIT_FLAG=""
if [ -n "$LIMIT" ]; then
    LIMIT_FLAG="--limit $LIMIT"
fi

# ============================================================
# Run
# ============================================================

echo "=== Step 1: Predict ==="
echo "  Model:            $PREDICT_MODEL"
echo "  Dataset:          $DATASET ($SPLIT)"
echo "  Max tokens:       $PREDICT_MAX_TOKENS"
echo "  Temperature:      $PREDICT_TEMPERATURE"
echo "  Top-p:            $PREDICT_TOP_P"
echo "  Top-k:            $PREDICT_TOP_K"
echo "  Reasoning effort: $PREDICT_REASONING_EFFORT"
echo "  Workers:          $WORKERS"
echo "  Output:           $PREDICTIONS"
OPENAI_API_KEY="$PREDICT_API_KEY" OPENAI_BASE_URL="$PREDICT_BASE_URL" \
python scripts/predict.py \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --model "$PREDICT_MODEL" \
    --output "$PREDICTIONS" \
    --max-tokens "$PREDICT_MAX_TOKENS" \
    --temperature "$PREDICT_TEMPERATURE" \
    --top-p "$PREDICT_TOP_P" \
    --top-k "$PREDICT_TOP_K" \
    --reasoning-effort "$PREDICT_REASONING_EFFORT" \
    --workers "$WORKERS" \
    $LIMIT_FLAG

echo ""
echo "=== Step 2: Judge ==="
echo "  Model:            $JUDGE_MODEL"
echo "  Max tokens:       $JUDGE_MAX_TOKENS"
echo "  Temperature:      $JUDGE_TEMPERATURE"
echo "  Reasoning effort: $JUDGE_REASONING_EFFORT"
echo "  Workers:          $WORKERS"
echo "  Output:           $RESULTS"
python scripts/judge.py \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --predictions "$PREDICTIONS" \
    --output "$RESULTS" \
    --predict-model "$PREDICT_MODEL" \
    --model "$JUDGE_MODEL" \
    --api-key "$JUDGE_API_KEY" \
    --base-url "$JUDGE_BASE_URL" \
    --max-tokens "$JUDGE_MAX_TOKENS" \
    --temperature "$JUDGE_TEMPERATURE" \
    --reasoning-effort "$JUDGE_REASONING_EFFORT" \
    --workers "$WORKERS" \
    $LIMIT_FLAG

echo ""
echo "=== Done ==="
echo "Output dir:  $MODEL_OUTPUT_DIR"
echo "Predictions: $PREDICTIONS"
echo "Results:     $RESULTS"
echo "Meta files:  $MODEL_OUTPUT_DIR/predict_meta.json, $MODEL_OUTPUT_DIR/judge_meta.json"
