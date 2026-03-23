#!/bin/bash
# Multi-stage training orchestration for Golden Snitch
set -e

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# Activate venv
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
    source "$ROOT/.venv/bin/activate"
fi

# ── Defaults ────────────────────────────────────────────────────────────────
TRACK="${TRACK:-simple_oval}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cpu}"
RESUME_MODEL=""      # name of model in saved_models/ to resume from
MODEL_NAME=""        # name to save the resulting model as
SKIP_EXPERT=false
SKIP_BC=false
NO_DOMAIN_RAND=false

# ── Flag parsing ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            RESUME_MODEL="$2"; shift 2 ;;
        --model-name)
            MODEL_NAME="$2"; shift 2 ;;
        --skip-expert)
            SKIP_EXPERT=true; shift ;;
        --skip-bc)
            SKIP_BC=true; shift ;;
        --no-domain-rand)
            NO_DOMAIN_RAND=true; shift ;;
        --track)
            TRACK="$2"; shift 2 ;;
        --seed)
            SEED="$2"; shift 2 ;;
        --device)
            DEVICE="$2"; shift 2 ;;
        --help|-h)
            cat <<'EOF'
Usage: bash scripts/train.sh [OPTIONS]

Options:
  --model <name>        Resume PPO from saved_models/<name>.zip (implies --skip-expert --skip-bc)
  --model-name <name>   Name to save the resulting model as (default: <track>_<timestamp>)
  --skip-expert         Skip Stage 1 (expert data generation), reuse latest existing data
  --skip-bc             Skip Stage 2 (behavior cloning), reuse latest existing BC model
  --no-domain-rand      Disable physics and sensor domain randomization during PPO training
  --track <name>        Track name (default: simple_oval)
  --seed <int>          Random seed (default: 42)
  --device <str>        Torch device: cpu | mps | cuda (default: cpu)

Environment variable overrides (legacy):
  TRACK, SEED, DEVICE, TIMESTEPS, N_ENVS

Examples:
  bash scripts/train.sh
  bash scripts/train.sh --model simple_oval_20260320_162152
  bash scripts/train.sh --skip-expert --skip-bc
  bash scripts/train.sh --model mymodel --model-name mymodel_v2
  TIMESTEPS=5000000 bash scripts/train.sh --track figure_eight
EOF
            exit 0 ;;
        *)
            echo "Unknown flag: $1. Run with --help for usage." >&2; exit 1 ;;
    esac
done

# Resuming from a saved model implies skipping upstream stages
if [[ -n "$RESUME_MODEL" ]]; then
    SKIP_EXPERT=true
    SKIP_BC=true
fi

echo "=== Golden Snitch Training Pipeline ==="
echo "Track: $TRACK | Seed: $SEED | Device: $DEVICE"
[[ -n "$RESUME_MODEL" ]]  && echo "Resuming model: $RESUME_MODEL"
[[ "$SKIP_EXPERT" == true ]] && echo "Skipping Stage 1 (expert generation)"
[[ "$SKIP_BC" == true ]]     && echo "Skipping Stage 2 (behavior cloning)"
[[ "$NO_DOMAIN_RAND" == true ]] && echo "Domain randomization: disabled"
echo ""

# ── Stage 1: Expert Data Generation ─────────────────────────────────────────
if [[ "$SKIP_EXPERT" == false ]]; then
    echo "--- Stage 1: Generating Expert Data ---"
    python -m planning.expert_policy \
        --track "$TRACK" \
        --episodes 500 \
        --speed 2.0 \
        --perturbation 0.3 \
        --seed "$SEED"
else
    echo "--- Stage 1: Skipped (reusing existing expert data) ---"
fi

EXPERT_DIR=$(ls -dt outputs/expert_data/expert_${TRACK}_* 2>/dev/null | head -1)
if [[ -z "$EXPERT_DIR" ]]; then
    echo "Error: No expert data found for track '$TRACK'. Remove --skip-expert to generate it." >&2
    exit 1
fi
echo "Expert data: $EXPERT_DIR"
echo ""

# ── Stage 2: Behavior Cloning ─────────────────────────────────────────────
if [[ "$SKIP_BC" == false ]]; then
    echo "--- Stage 2: Behavior Cloning ---"
    python -m learning.training.train_bc \
        --data "$EXPERT_DIR/expert_data.npz" \
        --track "$TRACK" \
        --epochs 50 \
        --batch-size 512 \
        --lr 0.001 \
        --seed "$SEED" \
        --device "$DEVICE"
else
    echo "--- Stage 2: Skipped (reusing existing BC model) ---"
fi

BC_DIR=$(ls -dt outputs/bc/bc_${TRACK}_* 2>/dev/null | head -1)
if [[ -z "$BC_DIR" ]]; then
    echo "Error: No BC model found for track '$TRACK'. Remove --skip-bc to train one." >&2
    exit 1
fi
echo "BC model: $BC_DIR"
echo ""

# ── Stage 3: PPO Fine-Tuning ─────────────────────────────────────────────
echo "--- Stage 3: PPO Training ---"

PPO_EXTRA_ARGS=""

# Resume from a saved model
if [[ -n "$RESUME_MODEL" ]]; then
    RESUME_PATH="$ROOT/saved_models/${RESUME_MODEL}.zip"
    if [[ ! -f "$RESUME_PATH" ]]; then
        echo "Error: Model '$RESUME_MODEL' not found at $RESUME_PATH" >&2
        echo "Available models:" >&2
        ls "$ROOT/saved_models/"*.zip 2>/dev/null | xargs -I{} basename {} .zip >&2 || echo "  (none)" >&2
        exit 1
    fi
    PPO_EXTRA_ARGS="$PPO_EXTRA_ARGS --resume $RESUME_PATH"

    CURRICULUM_STATE="$ROOT/saved_models/${RESUME_MODEL}_curriculum.json"
    if [[ -f "$CURRICULUM_STATE" ]]; then
        PPO_EXTRA_ARGS="$PPO_EXTRA_ARGS --curriculum-state $CURRICULUM_STATE"
    fi
fi

# Domain randomization flag
if [[ "$NO_DOMAIN_RAND" == true ]]; then
    PPO_EXTRA_ARGS="$PPO_EXTRA_ARGS --no-domain-rand"
fi

python -m learning.training.train_ppo \
    --track "$TRACK" \
    --bc-model "$BC_DIR/best_model.pt" \
    --timesteps "${TIMESTEPS:-2500000}" \
    --n-envs "${N_ENVS:-4}" \
    --seed "$SEED" \
    --device "$DEVICE" \
    $PPO_EXTRA_ARGS

PPO_DIR=$(ls -dt outputs/ppo/ppo_${TRACK}_* | head -1)
echo "PPO model: $PPO_DIR"
echo ""

# ── Stage 4: Evaluation ─────────────────────────────────────────────────────
echo "--- Stage 4: Evaluation ---"
python -m learning.evaluation.report eval \
    --model "$PPO_DIR/best_model.zip" \
    --track "$TRACK" \
    --episodes 100 \
    --output-dir "$PPO_DIR/eval"

# ── Save to saved_models/ ────────────────────────────────────────────────────
SAVED_DIR="$ROOT/saved_models"
mkdir -p "$SAVED_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_NAME="${MODEL_NAME:-${TRACK}_${TIMESTAMP}}"

cp "$PPO_DIR/best_model.zip" "$SAVED_DIR/${MODEL_NAME}.zip"
if [[ -f "$PPO_DIR/curriculum_history.json" ]]; then
    cp "$PPO_DIR/curriculum_history.json" "$SAVED_DIR/${MODEL_NAME}_curriculum.json"
fi

echo ""
echo "=== Training Complete ==="
echo "Results:     $PPO_DIR/eval/"
echo "Saved model: saved_models/${MODEL_NAME}.zip"
echo ""
echo "To resume this model later:"
echo "  bash scripts/train.sh --model ${MODEL_NAME}"
