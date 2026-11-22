#!/usr/bin/env python3
"""Verificación del entorno antzoki-mini-tts (Audio8_TTS + pipeline tipo antzoki-tts).

Uso (con el entorno activado: `source scripts/env.sh`):
    python verify_env.py                 # comprobaciones rápidas, sin descargas
    python verify_env.py --run-tests     # + tests unitarios de Audio8_TTS (pytest)
    python verify_env.py --load-model    # + descarga Audio8 (~2.5 GB), tokeniza euskera y sintetiza

Código de salida 0 si todas las comprobaciones obligatorias pasan.
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIO8_DIR = ROOT / "repos" / "Audio8_TTS"
BASQUE_TEXT = "Kaixo! Gaur eguraldi ona dago Donostian, eta hondartzara joango gara."

RESULTS: list[tuple[str, str, str]] = []  # (estado, nombre, detalle)


def check(name: str, required: bool = True):
    """Decorador: ejecuta la comprobación y registra OK / FAIL / WARN."""

    def wrap(fn):
        def run(*args, **kwargs):
            try:
                detail = fn(*args, **kwargs) or ""
                RESULTS.append(("OK", name, str(detail)))
                print(f"  \033[32m✔\033[0m {name}: {detail}")
            except Exception as exc:  # noqa: BLE001
                state = "FAIL" if required else "WARN"
                RESULTS.append((state, name, f"{type(exc).__name__}: {exc}"))
                color = "31" if required else "33"
                print(f"  \033[{color}m✘ [{state}]\033[0m {name}: {type(exc).__name__}: {exc}")
                if os.environ.get("VERIFY_DEBUG"):
                    traceback.print_exc()

        return run

    return wrap


def section(title: str) -> None:
    print(f"\n\033[1m── {title} \033[0m" + "─" * max(0, 60 - len(title)))


# ─────────────────────────────────────────────────────────────────────────────
#  1. Python / sistema
# ─────────────────────────────────────────────────────────────────────────────
@check("Python")
def check_python():
    v = sys.version_info
    if not (3, 10) <= (v.major, v.minor) < (3, 13):
        raise RuntimeError(f"Python {v.major}.{v.minor}; se espera 3.10–3.12 (recomendado 3.11)")
    in_venv = sys.prefix != sys.base_prefix
    return f"{sys.version.split()[0]} | venv={'sí' if in_venv else 'NO'} ({sys.prefix})"


@check("Binarios de sistema (ffmpeg, sox, espeak-ng)")
def check_binaries():
    missing = [b for b in ("ffmpeg", "sox", "espeak-ng") if shutil.which(b) is None]
    if missing:
        raise RuntimeError(f"faltan en PATH: {missing} → sudo apt install {' '.join(missing)}")
    ff = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split("\n")[0]
    return ff


@check("espeak-ng tiene voz euskera ('eu')")
def check_espeak_voice():
    out = subprocess.run(["espeak-ng", "--voices=eu"], capture_output=True, text=True).stdout
    if " eu " not in out and "\teu" not in out and "basque" not in out.lower():
        raise RuntimeError("espeak-ng no lista la voz 'eu'")
    return out.strip().splitlines()[-1].strip()


# ─────────────────────────────────────────────────────────────────────────────
#  2. PyTorch / CUDA
# ─────────────────────────────────────────────────────────────────────────────
@check("PyTorch + CUDA")
def check_torch():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"torch {torch.__version__}: CUDA NO disponible")
    archs = torch.cuda.get_arch_list()
    return f"torch {torch.__version__} | CUDA {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()} | arch={archs}"


@check("GPUs visibles (cómputo bf16 + SDPA en cada una)")
def check_gpus():
    import torch
    import torch.nn.functional as F

    lines = []
    archs = set(torch.cuda.get_arch_list())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        sm = f"sm_{p.major}{p.minor}"
        if sm not in archs and f"compute_{p.major}{p.minor}" not in archs:
            raise RuntimeError(f"cuda:{i} {p.name} ({sm}) no soportada por este build de torch: {archs}")
        dev = torch.device(f"cuda:{i}")
        try:
            free, total = torch.cuda.mem_get_info(dev)
        except Exception as exc:  # noqa: BLE001 - sin memoria ni para crear el contexto CUDA
            if "out of memory" not in str(exc):
                raise
            free, total = 0, p.total_memory
        if free < 1 * 2**30:
            raise RuntimeError(
                f"cuda:{i} {p.name}: sólo {free / 2**30:.2f}/{total / 2**30:.1f} GiB libres; otro proceso "
                "ocupa la GPU (mira `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`)"
            )
        a = torch.randn(1024, 1024, device=dev, dtype=torch.bfloat16)
        (a @ a).sum().item()
        q = torch.randn(1, 8, 128, 64, device=dev, dtype=torch.bfloat16)
        F.scaled_dot_product_attention(q, q, q, is_causal=True).sum().item()
        lines.append(f"cuda:{i}={p.name} [{sm}, {free / 2**30:.1f}/{total / 2**30:.1f} GiB libres, "
                     f"bf16={torch.cuda.is_bf16_supported()}]")
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "<todas>")
    order = os.environ.get("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    return f"CUDA_VISIBLE_DEVICES={vis} ({order}) → " + "; ".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  3. Audio
# ─────────────────────────────────────────────────────────────────────────────
def _tone(sr: int = 44100, seconds: float = 1.0):
    import numpy as np

    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32"), sr


@check("soundfile / libsndfile (WAV + FLAC)")
def check_soundfile():
    import numpy as np
    import soundfile as sf

    wav, sr = _tone()
    with tempfile.TemporaryDirectory() as d:
        for ext in ("wav", "flac"):
            path = Path(d) / f"t.{ext}"
            sf.write(path, wav, sr)
            back, sr2 = sf.read(path, dtype="float32")
            assert sr2 == sr and np.allclose(back, wav, atol=1e-3), ext
    return f"soundfile {sf.__version__} | {sf.__libsndfile_version__}"


@check("torchaudio (resample 48k→44.1k, load, MelSpectrogram en GPU)")
def check_torchaudio():
    import soundfile as sf
    import torch
    import torchaudio
    from torchaudio.functional import resample

    wav, _ = _tone(48000)
    x = torch.from_numpy(wav)
    y = resample(x, 48000, 44100)
    assert abs(y.numel() - 44100) <= 1
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.wav"
        sf.write(p, wav, 48000)
        loaded, sr = torchaudio.load(str(p))
        assert sr == 48000 and loaded.shape[-1] == wav.shape[0]
    mel = torchaudio.transforms.MelSpectrogram(44100, n_fft=2048, n_mels=128).cuda()
    m = mel(y.cuda())
    backends = getattr(torchaudio, "list_audio_backends", lambda: ["?"])()
    return f"torchaudio {torchaudio.__version__} | backends={backends} | mel={tuple(m.shape)}"


@check("librosa (+ numba)")
def check_librosa():
    import librosa
    import numba

    wav, sr = _tone(22050)
    f0 = librosa.yin(wav, fmin=80, fmax=800, sr=sr)
    S = librosa.feature.melspectrogram(y=wav, sr=sr)
    return f"librosa {librosa.__version__} | numba {numba.__version__} | f0≈{float(f0.mean()):.0f} Hz | mel={S.shape}"


@check("ffmpeg decodifica MP3 (formato típico de OpenSLR/Common Voice)")
def check_ffmpeg_mp3():
    import soundfile as sf

    wav, sr = _tone()
    with tempfile.TemporaryDirectory() as d:
        src, mp3, out = Path(d) / "a.wav", Path(d) / "a.mp3", Path(d) / "b.wav"
        sf.write(src, wav, sr)
        for cmd in (
            ["ffmpeg", "-loglevel", "error", "-y", "-i", src, mp3],
            ["ffmpeg", "-loglevel", "error", "-y", "-i", mp3, "-ar", "44100", "-ac", "1", out],
        ):
            subprocess.run([str(c) for c in cmd], check=True)
        info = sf.info(out)
    return f"mp3→wav OK ({info.samplerate} Hz, {info.duration:.2f}s)"


@check("pyloudnorm (normalización LUFS)", required=False)
def check_loudnorm():
    import pyloudnorm as pyln

    wav, sr = _tone()
    return f"{pyln.Meter(sr).integrated_loudness(wav):.1f} LUFS"


# ─────────────────────────────────────────────────────────────────────────────
#  4. Fonemizador
# ─────────────────────────────────────────────────────────────────────────────
@check("phonemizer + espeak-ng (euskera)", required=False)
def check_phonemizer():
    from phonemizer import phonemize
    from phonemizer.backend import EspeakBackend

    ver = ".".join(map(str, EspeakBackend.version()))
    ipa = phonemize(BASQUE_TEXT, language="eu", backend="espeak", strip=True,
                    preserve_punctuation=True, with_stress=True)
    if not ipa:
        raise RuntimeError("salida vacía")
    return f"espeak-ng {ver} | '{BASQUE_TEXT[:30]}…' → /{ipa[:60]}…/"


# ─────────────────────────────────────────────────────────────────────────────
#  5. Stack de ML
# ─────────────────────────────────────────────────────────────────────────────
@check("transformers / accelerate / safetensors / hub")
def check_hf():
    import accelerate
    import huggingface_hub
    import safetensors
    import transformers

    major_minor = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    if not (4, 57) <= major_minor < (5, 0):
        raise RuntimeError(f"transformers {transformers.__version__}; Audio8 exige >=4.57,<5")
    return (f"transformers {transformers.__version__} | accelerate {accelerate.__version__} | "
            f"safetensors {safetensors.__version__} | hub {huggingface_hub.__version__}")


@check("DeepSpeed (usado por audio8_tts_sft.sh)", required=False)
def check_deepspeed():
    os.environ.setdefault("DS_SKIP_CUDA_CHECK", "1")
    import deepspeed

    return f"deepspeed {deepspeed.__version__}"


@check("tensorboard / peft / jiwer / pandas", required=False)
def check_misc():
    vers = []
    for mod in ("tensorboard", "peft", "jiwer", "pandas", "yaml"):
        m = importlib.import_module(mod)
        vers.append(f"{mod} {getattr(m, '__version__', '?')}")
    return " | ".join(vers)


# ─────────────────────────────────────────────────────────────────────────────
#  6. Código de Audio8_TTS
# ─────────────────────────────────────────────────────────────────────────────
@check("Import de módulos Audio8_TTS")
def check_audio8_import():
    if not AUDIO8_DIR.is_dir():
        raise FileNotFoundError(f"{AUDIO8_DIR} no existe (clona el repo)")
    if str(AUDIO8_DIR) not in sys.path:
        sys.path.insert(0, str(AUDIO8_DIR))
    mods = ["audio8_tts_data", "audio8_tts_prepare", "audio8_tts_infer", "audio8_tts_sft"]
    for m in mods:
        importlib.import_module(m)
    from audio8_tts_data import NUM_CODEBOOKS, clean_text

    cleaned = clean_text("  Kaixo,\tmundua!\nZer moduz?​ ")
    return f"{', '.join(mods)} | NUM_CODEBOOKS={NUM_CODEBOOKS} | clean_text→'{cleaned}'"


@check("Tests unitarios Audio8_TTS (pytest)")
def check_audio8_tests():
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
                       cwd=AUDIO8_DIR, capture_output=True, text=True)
    tail = (r.stdout.strip().splitlines() or ["(sin salida)"])[-1]
    if r.returncode != 0:
        raise RuntimeError(tail + "\n" + r.stdout[-2000:] + r.stderr[-1000:])
    return tail


@check("Modelo Audio8 (descarga, tokenizador en euskera, síntesis)")
def check_audio8_model(model_id: str):
    import torch
    from transformers import AutoProcessor

    t0 = time.time()
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    tok = getattr(proc, "tokenizer", None)
    tok_info = ""
    if tok is not None:
        ids = tok(BASQUE_TEXT, add_special_tokens=False)["input_ids"]
        roundtrip = tok.decode(ids)
        assert roundtrip.strip() == BASQUE_TEXT, f"round-trip roto: {roundtrip!r}"
        tok_info = f"{len(ids)} tokens / {len(BASQUE_TEXT)} chars ({len(BASQUE_TEXT) / len(ids):.2f} c/t)"

    out_dir = ROOT / "checkpoints" / "verify"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "audio8_eu_test.wav"
    cmd = [sys.executable, str(AUDIO8_DIR / "audio8_tts_infer.py"), "--model", model_id,
           "--text", BASQUE_TEXT, "--output", str(out), "--overwrite"]
    r = subprocess.run(cmd, cwd=AUDIO8_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2500:])
    import soundfile as sf

    info = sf.info(out)
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
    return (f"{tok_info} | WAV {info.samplerate} Hz, {info.duration:.1f}s → {out.relative_to(ROOT)} "
            f"| {time.time() - t0:.0f}s (escúchalo: sin fine-tune sonará con acento no vasco)"
            + (f" | peak {peak:.1f} GiB" if peak else ""))


# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-tests", action="store_true", help="ejecutar tests de Audio8_TTS")
    ap.add_argument("--load-model", action="store_true", help="descargar modelo y sintetizar")
    ap.add_argument("--model", default=os.environ.get("MODEL", "Edge0/Audio8-TTS-Realistic-Air"))
    args = ap.parse_args()

    section("Sistema");        check_python(); check_binaries(); check_espeak_voice()
    section("PyTorch / GPU");  check_torch(); check_gpus()
    section("Audio");          check_soundfile(); check_torchaudio(); check_librosa(); check_ffmpeg_mp3(); check_loudnorm()
    section("Fonemizador");    check_phonemizer()
    section("Stack ML");       check_hf(); check_deepspeed(); check_misc()
    section("Audio8_TTS");     check_audio8_import()
    if args.run_tests:
        check_audio8_tests()
    if args.load_model:
        check_audio8_model(args.model)

    fails = [r for r in RESULTS if r[0] == "FAIL"]
    warns = [r for r in RESULTS if r[0] == "WARN"]
    oks = len(RESULTS) - len(fails) - len(warns)
    print(f"\n\033[1mResumen:\033[0m {oks} OK, {len(warns)} avisos, {len(fails)} fallos")
    for state, name, detail in fails + warns:
        print(f"  [{state}] {name}: {detail.splitlines()[0] if detail else ''}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
