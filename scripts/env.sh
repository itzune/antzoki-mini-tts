# Uso:  source scripts/env.sh
# Activa .venv y fija variables del proyecto. Pensado para hacer `source`, no para ejecutarse.

_TTS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TTS_ROOT="$_TTS_ROOT"
# shellcheck disable=SC1091
source "$TTS_ROOT/.venv/bin/activate"

# GPUs: por defecto CUDA ordena "la más rápida primero", que NO coincide con nvidia-smi.
# Con PCI_BUS_ID: 0 = RTX 3070 (8 GB), 1 = RTX 5090 Laptop (24 GB).
# Entrenamos sólo en la 5090 (mezclar 8 GB + 24 GB en DDP te limita a la pequeña).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

export PYTHONPATH="$TTS_ROOT/repos/Audio8_TTS${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DS_SKIP_CUDA_CHECK=1          # nvcc 13.x del sistema vs CUDA 12.8 de torch

# phonemizer: localizar libespeak-ng explícitamente evita el típico "espeak not installed"
for _lib in /usr/lib/x86_64-linux-gnu/libespeak-ng.so.1 /usr/lib/x86_64-linux-gnu/libespeak-ng.so; do
  if [[ -e "$_lib" ]]; then export PHONEMIZER_ESPEAK_LIBRARY="$_lib"; break; fi
done
unset _lib _TTS_ROOT

export MODEL="${MODEL:-Edge0/Audio8-TTS-Realistic-Air}"
echo "[env] .venv activo | CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (PCI order) | MODEL=$MODEL"
