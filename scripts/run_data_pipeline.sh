#!/usr/bin/env bash
# =============================================================================
#  Pasos 1–3: OpenSLR76 → WAV 44.1 kHz → códigos Audio8 → manifiestos SFT
#  Idempotente: audio8_tts_prepare.py reutiliza los .npy ya válidos.
#  Uso:  bash scripts/run_data_pipeline.sh  [--ref-ratio 0.6 …]   (args extra → make_sft_splits.py)
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh" >/dev/null
RAW="$ROOT/dataset_raw/openslr76"
OUT="$ROOT/dataset_processed/openslr76"

if ! find "$RAW" -name '*.wav' -print -quit | grep -q .; then
  echo "No hay WAVs en $RAW. Descarga y descomprime:" >&2
  echo "  cd $RAW && for f in eu_es_female.zip eu_es_male.zip; do curl -LO https://www.openslr.org/resources/76/\$f && unzip -q \$f -d audio; done" >&2
  exit 2
fi

echo "== 1/3 Preprocesado de audio y texto"
python -W ignore "$ROOT/scripts/prepare_openslr76.py" --raw-dir "$RAW" --out-dir "$OUT"

echo "== 2/3 Codificación con el códec de Audio8 (GPU)"
python -W ignore "$ROOT/repos/Audio8_TTS/audio8_tts_prepare.py" \
  --model "$MODEL" \
  --input-jsonl "$OUT/raw.jsonl" \
  --output-jsonl "$OUT/prepared/all.jsonl" \
  --batch-size "${PREPARE_BATCH_SIZE:-16}"

echo "== 3/3 Splits train/eval + parejas de referencia"
python -W ignore "$ROOT/scripts/make_sft_splits.py" "$@"

echo
echo "Datos listos en $OUT/sft"
echo "Siguiente:  python scripts/synth_checkpoints.py --checkpoint base   (línea base, antes de entrenar)"
echo "            bash scripts/train_sft.sh"
