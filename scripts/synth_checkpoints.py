#!/usr/bin/env python3
"""Paso 5 — Sintetiza los prompts de escucha con uno o varios checkpoints.

Los checkpoints intermedios del Trainer (checkpoint-N/) traen pesos, código remoto y
codec.pth, pero NO el tokenizer/procesador; este script los copia del modelo base
antes de inferir.

Uso:
  python scripts/synth_checkpoints.py --checkpoint base                       # modelo sin fine-tune (línea base)
  python scripts/synth_checkpoints.py --checkpoint checkpoints/audio8_eu_sft_v1/checkpoint-500
  python scripts/synth_checkpoints.py --run checkpoints/audio8_eu_sft_v1      # todos los checkpoint-* + export

Salida: <checkpoint>/listen/<id>.wav   (o checkpoints/baseline_listen/ para "base")
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INFER = ROOT / "repos" / "Audio8_TTS" / "audio8_tts_infer.py"
PROCESSOR_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "processor_config.json", "preprocessor_config.json", "processing_arktts.py",
    "configuration_arktts.py", "modeling_arktts.py", "modeling_arktts_codec.py",
    "generation_config.json",
)


def base_snapshot(model_id: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_id, allow_patterns=list(PROCESSOR_FILES)))


def complete_checkpoint(ckpt: Path, base: Path) -> None:
    for name in PROCESSOR_FILES:
        src, dst = base / name, ckpt / name
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)
    if not (ckpt / "codec.pth").is_file():
        raise FileNotFoundError(f"{ckpt} no contiene codec.pth")


def synth(model: str, out_dir: Path, prompts: Path, args) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(INFER), "--model", model, "--input-jsonl", str(prompts),
           "--output-dir", str(out_dir), "--batch-size", str(args.batch_size),
           "--seed", str(args.seed)]
    if args.overwrite:
        cmd.append("--overwrite")
    print(f"\n[synth] {model} → {out_dir}")
    return subprocess.run(cmd, cwd=INFER.parent).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", help="ruta a checkpoint-N/, export/, o 'base'")
    g.add_argument("--run", type=Path, help="directorio de un run: procesa todos sus checkpoints")
    ap.add_argument("--prompts", type=Path,
                    default=ROOT / "dataset_processed" / "openslr76" / "sft" / "val_prompts.jsonl")
    ap.add_argument("--model", default=os.environ.get("MODEL", "Edge0/Audio8-TTS-Realistic-Air"))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not args.prompts.is_file():
        sys.exit(f"No existe {args.prompts}; ejecuta scripts/make_sft_splits.py")

    if args.checkpoint == "base":
        return synth(args.model, ROOT / "checkpoints" / "baseline_listen", args.prompts, args)

    base = base_snapshot(args.model)
    if args.run:
        ckpts = sorted(args.run.glob("checkpoint-*"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0))
        if (args.run / "export" / "model.safetensors").is_file():
            ckpts.append(args.run / "export")
    else:
        ckpts = [Path(args.checkpoint)]
    rc = 0
    for ckpt in ckpts:
        ckpt = ckpt.resolve()
        if not (ckpt / "model.safetensors").is_file():
            print(f"[synth] saltando {ckpt} (sin model.safetensors)")
            continue
        complete_checkpoint(ckpt, base)
        rc |= synth(str(ckpt), ckpt / "listen", args.prompts, args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
