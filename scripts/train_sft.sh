#!/usr/bin/env bash
# =============================================================================
#  Paso 4 — SFT completo de Audio8-TTS en euskera (1× RTX 5090, 24 GB)
#
#  Envuelve repos/Audio8_TTS/audio8_tts_sft.sh con valores pensados para
#  ADAPTAR UN IDIOMA NUEVO (el euskera no está entre los 23 de Realistic-Air).
#  Cualquier variable se puede sobreescribir:  LEARNING_RATE=1e-5 bash scripts/train_sft.sh
#  Argumentos extra se pasan al Trainer:       bash scripts/train_sft.sh --max_steps 20
#
#  Prueba rápida (≈2 min):  RUN_NAME=smoke bash scripts/train_sft.sh --max_steps 20 --save_steps 1000000
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh" >/dev/null

DATA="${DATA:-$ROOT/dataset_processed/openslr76/sft}"
RUN_NAME="${RUN_NAME:-audio8_eu_sft_v1}"

export MODEL="${MODEL:-Edge0/Audio8-TTS-Realistic-Air}"
export TRAIN_JSONL="${TRAIN_JSONL:-$DATA/train.jsonl}"
export EVAL_JSONL="${EVAL_JSONL:-$DATA/eval.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/checkpoints/$RUN_NAME}"
export EXPORT_DIR="${EXPORT_DIR:-$OUTPUT_DIR/export}"
export PYTHON="$ROOT/.venv/bin/python"
export NPROC_PER_NODE=1                          # sólo la 5090 (la 3070 de 8 GB no da para SFT completo)
export MASTER_PORT="${MASTER_PORT:-29517}"

# ── Hiperparámetros ──────────────────────────────────────────────────────────
# ~6.7k frases / lote efectivo 16 ≈ 420 pasos por época.
export BATCH_SIZE="${BATCH_SIZE:-4}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"   # efectivo = 16
export LEARNING_RATE="${LEARNING_RATE:-2e-5}"    # 2× el defecto de Audio8 (1e-5, pensado para la misma lengua)
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-6}" # ≈2.5k pasos; escucha checkpoints, suele bastar con 3–5
export WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
export MAX_LENGTH="${MAX_LENGTH:-2048}"          # debe coincidir con make_sft_splits.py --max-length
export LOGGING_STEPS="${LOGGING_STEPS:-10}"
export SAVE_STEPS="${SAVE_STEPS:-250}"
export EVAL_STEPS="${EVAL_STEPS:-250}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}" # cada checkpoint ≈ 10 GB (pesos + códec + estados Adam)
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
export BF16=true
export GRADIENT_CHECKPOINTING=false              # el modelo 0.6B no lo soporta
export RESUME_MODE="${RESUME_MODE:-auto}"        # reanuda del último checkpoint si existe
export REPORT_TO="${REPORT_TO:-tensorboard}"
# DeepSpeed ZeRO-2 se mantiene aunque haya 1 GPU: el modelo se carga en bf16 y ZeRO
# guarda pesos maestros fp32 en el optimizador; sin él, updates de ~1e-5 se pierden en bf16.

for f in "$TRAIN_JSONL" "$EVAL_JSONL"; do
  [[ -f "$f" ]] || { echo "Falta $f — ejecuta antes scripts/make_sft_splits.py" >&2; exit 2; }
done
mkdir -p "$OUTPUT_DIR"
{
  echo "# $(date -Iseconds)"
  env | grep -E '^(DYNAMIC_REF_RATIO|FREEZE_SLOW_AR|FREEZE_FAST_AR|MODEL|TRAIN_JSONL|EVAL_JSONL|OUTPUT_DIR|BATCH_SIZE|EVAL_BATCH_SIZE|GRADIENT_ACCUMULATION_STEPS|LEARNING_RATE|NUM_TRAIN_EPOCHS|WARMUP_RATIO|LR_SCHEDULER_TYPE|WEIGHT_DECAY|MAX_LENGTH|SAVE_STEPS|EVAL_STEPS|CUDA_VISIBLE_DEVICES)=' | sort
  echo "EXTRA_ARGS=$*"
} > "$OUTPUT_DIR/run_config.env"

# Dynamic references (scripts/sft_dynamic.py): DYNAMIC_REF_RATIO=0.75 → each row is drawn
# with/without a reference every time it is loaded. Empty = static references from train.jsonl.
export DYNAMIC_REF_RATIO="${DYNAMIC_REF_RATIO:-}"
export REF_POOL="${REF_POOL:-$DATA/ref_pool.json}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$ROOT/repos/Audio8_TTS/configs/deepspeed_zero2.json}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "[train] $RUN_NAME → $OUTPUT_DIR  (GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, PCI order)"
echo "[train] referencias dinámicas: ${DYNAMIC_REF_RATIO:-no (estáticas)}"
echo "[train] tensorboard --logdir $OUTPUT_DIR"
# Same arguments as repos/Audio8_TTS/audio8_tts_sft.sh, but runs scripts/sft_dynamic.py
# (identical to audio8_tts_sft.py when DYNAMIC_REF_RATIO is empty).
"$PYTHON" -m torch.distributed.run \
  --nnodes 1 --node_rank 0 --nproc_per_node "$NPROC_PER_NODE" \
  --master_addr 127.0.0.1 --master_port "$MASTER_PORT" \
  "$ROOT/scripts/sft_dynamic.py" \
  --model_name_or_path "$MODEL" \
  --train_jsonl "$TRAIN_JSONL" --eval_jsonl "$EVAL_JSONL" \
  --output_dir "$OUTPUT_DIR" --export_dir "$EXPORT_DIR" \
  --max_length "$MAX_LENGTH" \
  --per_device_train_batch_size "$BATCH_SIZE" --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning_rate "$LEARNING_RATE" --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --warmup_ratio "$WARMUP_RATIO" --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
  --weight_decay "$WEIGHT_DECAY" --max_grad_norm "$MAX_GRAD_NORM" \
  --logging_steps "$LOGGING_STEPS" \
  --save_strategy steps --save_steps "$SAVE_STEPS" --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --do_train true --do_eval true --eval_strategy steps --eval_steps "$EVAL_STEPS" \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --bf16 "$BF16" --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
  --freeze_slow_ar "${FREEZE_SLOW_AR:-false}" --freeze_fast_ar "${FREEZE_FAST_AR:-false}" \
  --resume_mode "$RESUME_MODE" --report_to "$REPORT_TO" \
  --remove_unused_columns false \
  --seed "${SEED:-42}" \
  --deepspeed "$DEEPSPEED_CONFIG" \
  "$@" 2>&1 | tee -a "$OUTPUT_DIR/train.log"
