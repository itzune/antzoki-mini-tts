#!/usr/bin/env bash
# =============================================================================
#  setup_env.sh — Entorno para fine-tuning de Audio8_TTS en euskera
#
#  Uso:
#    bash setup_env.sh                  # todo (pide sudo para apt)
#    bash setup_env.sh --skip-system    # sin apt (si ya instalaste los paquetes)
#    bash setup_env.sh --recreate       # borra y recrea .venv
#    bash setup_env.sh --skip-deepspeed # sin DeepSpeed (el launcher sft lo usa por defecto)
#
#  Variables opcionales: PYTHON_VERSION (3.11), TORCH_VERSION (2.8.0), CUDA_TAG (cu128)
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${ROOT}/.venv"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
CUDA_TAG="${CUDA_TAG:-cu128}"
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"

SKIP_SYSTEM=0; SKIP_CLONE=0; SKIP_DEEPSPEED=0; RECREATE=0
for arg in "$@"; do
  case "$arg" in
    --skip-system)    SKIP_SYSTEM=1 ;;
    --skip-clone)     SKIP_CLONE=1 ;;
    --skip-deepspeed) SKIP_DEEPSPEED=1 ;;
    --recreate)       RECREATE=1 ;;
    -h|--help) sed -n 2,13p "$0"; exit 0 ;;
    *) echo "Argumento desconocido: $arg" >&2; exit 2 ;;
  esac
done

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[!] %s\033[0m\n' "$*" >&2; }

cd "$ROOT"
mkdir -p repos dataset_raw dataset_processed checkpoints scripts

# ── 1. Paquetes de sistema ────────────────────────────────────────────────────
APT_PKGS=(ffmpeg sox libsox-fmt-all libsndfile1 espeak-ng libespeak-ng1 git git-lfs curl build-essential)
if [[ $SKIP_SYSTEM -eq 0 ]]; then
  missing=()
  for p in "${APT_PKGS[@]}"; do
    dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q "install ok installed" || missing+=("$p")
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    log "Instalando paquetes de sistema: ${missing[*]}"
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends "${missing[@]}"
  else
    log "Paquetes de sistema ya instalados"
  fi
  git lfs install --skip-repo >/dev/null 2>&1 || true
else
  log "Saltando paquetes de sistema (--skip-system)"
fi

# ── 2. Repositorios ───────────────────────────────────────────────────────────
if [[ $SKIP_CLONE -eq 0 && -f .gitmodules ]] && git rev-parse --git-dir >/dev/null 2>&1; then
  # Clon de este proyecto: submódulos fijados al commit con el que se entrenó.
  log "Submódulos en repos/ (commits fijados)"
  git submodule update --init --recursive
elif [[ $SKIP_CLONE -eq 0 ]]; then
  log "Repositorios en repos/"
  [[ -d repos/Audio8_TTS/.git ]] || git clone https://github.com/Edge0-AI/Audio8_TTS.git repos/Audio8_TTS
  if [[ ! -d repos/antzoki-tts/.git ]]; then
    git clone --recurse-submodules https://github.com/itzune/antzoki-tts.git repos/antzoki-tts
  else
    git -C repos/antzoki-tts submodule update --init --recursive
  fi
fi

# ── 3. uv (gestor de Python/venv) ────────────────────────────────────────────
export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
  log "Instalando uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version

# ── 4. Entorno virtual (.venv con Python gestionado por uv) ──────────────────
# El Python del sistema (3.14) es demasiado nuevo para deepspeed/numba/phonemizer.
if [[ $RECREATE -eq 1 && -d "$VENV" ]]; then
  log "Borrando .venv existente (--recreate)"
  rm -rf "$VENV"
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  log "Creando .venv con Python ${PYTHON_VERSION}"
  uv python install "$PYTHON_VERSION"
  uv venv "$VENV" --python "$PYTHON_VERSION" --seed
fi
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
PY="$VENV/bin/python"
"$PY" --version

# ── 5. PyTorch con CUDA ───────────────────────────────────────────────────────
log "Instalando torch==${TORCH_VERSION} / torchaudio==${TORCH_VERSION} (${CUDA_TAG})"
uv pip install --python "$PY" \
  "torch==${TORCH_VERSION}" "torchaudio==${TORCH_VERSION}" \
  --index-url "$TORCH_INDEX"

# ── 6. Resto de dependencias (torch bloqueado por constraints) ───────────────
log "Instalando dependencias Python"
uv pip install --python "$PY" -r requirements-env.txt -c constraints-env.txt

# ── 7. DeepSpeed (lo usa audio8_tts_sft.sh por defecto, ZeRO-2) ──────────────
# DS_BUILD_OPS=0: no compila kernels en la instalación. El nvcc del sistema (13.x) no
# coincide con el CUDA 12.8 de torch; con la config ZeRO-2 de Audio8 (optimizador de
# torch) no hace falta ningún op compilado.
if [[ $SKIP_DEEPSPEED -eq 0 ]]; then
  log "Instalando DeepSpeed (sin compilar ops)"
  uv pip install --python "$PY" setuptools wheel packaging ninja py-cpuinfo psutil -c constraints-env.txt
  DS_BUILD_OPS=0 DS_SKIP_CUDA_CHECK=1 \
    uv pip install --python "$PY" --no-build-isolation "deepspeed>=0.17,<0.19" -c constraints-env.txt \
    || warn "DeepSpeed no se pudo instalar. Entrena sin él: añade '--deepspeed \"\"' o usa scripts/env.sh + torchrun."
fi

# ── 8. Instantánea de versiones ───────────────────────────────────────────────
uv pip freeze --python "$PY" > requirements.lock.txt
log "Versiones congeladas en requirements.lock.txt"

cat <<EOF

Entorno listo.
  Activar:     source scripts/env.sh
  Verificar:   python verify_env.py            (rápido)
               python verify_env.py --load-model   (descarga ~2.5 GB y sintetiza una frase)
EOF
