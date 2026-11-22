#!/usr/bin/env python3
"""Convert the Basque fine-tune to the Audio8 llama.cpp F32 bundle (then quantize with quantize_eu.sh).

The official converter (repos/Audio8_TTS/llama_cpp/converter) only accepts the pinned public
checkpoint: it checks every source file against model-source-lock.json. This wrapper runs it
unchanged, but first swaps that lock for one computed from our model:

  * model.safetensors and the codebook sidecar: hashes of the fine-tuned weights;
  * config.json: ours, plus "tie_word_embeddings": true (transformers drops the default when it
    saves; the architecture is identical and the converter requires the key);
  * LICENSE / NOTICE: the original Audio8 files (the bundle verifier requires them);
  * codec, tokenizer and processor are byte-identical to the base model, so the hashes the
    native runtime hard-codes still match.
The RoPE tables are recomputed and still checked against the official values.

Needs the pinned conversion env (.venv-gguf: torch 2.8.0 CPU, numpy 1.26.4, safetensors 0.5.3,
PyYAML 6.0.3) and a patched llama.cpp tree (vendor/llama.cpp):

  .venv-gguf/bin/python scripts/gguf/convert_eu.py --model hf_release/antzoki-mini-tts \
      --out gguf/antzoki-mini-tts-f32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONVERTER = ROOT / "repos" / "Audio8_TTS" / "llama_cpp" / "converter"
AUDIO8 = ROOT / "repos" / "Audio8_TTS"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="HF folder (export/ or hf_release/<name>)")
    ap.add_argument("--out", type=Path, required=True, help="new bundle directory (must not exist)")
    ap.add_argument("--llama-cpp", type=Path, default=ROOT / "vendor" / "llama.cpp")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    model = args.model.resolve()
    out = args.out.absolute()
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.with_name(f".{out.name}.src")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    sys.path.insert(0, str(CONVERTER))
    import convert_audio8_realistic as conv  # noqa: E402

    lock = conv.MODEL_LOCK
    for name in lock["files"]:
        dst = staging / name
        if name == "config.json":
            cfg = json.loads((model / name).read_text())
            cfg.setdefault("tie_word_embeddings", True)
            dst.write_text(json.dumps(cfg, indent=2) + "\n")
        elif name in ("LICENSE", "NOTICE"):
            shutil.copyfile(AUDIO8 / name, dst)
        else:
            if not (model / name).is_file():
                sys.exit(f"missing {model / name}")
            dst.symlink_to(model / name)

    # the native runtime hard-codes these: they must be the base model's files
    for name in ("codec.pth", "tokenizer.json", "processing_arktts.py"):
        if sha256(staging / name) != lock["files"][name]["sha256"]:
            sys.exit(f"{name} differs from the base model: the native runtime would reject it")
    for name in ("LICENSE", "NOTICE"):
        if sha256(staging / name) != lock["files"][name]["sha256"]:
            sys.exit(f"{name} differs from the pinned Audio8 file")

    print("[gguf] hashing source files…", flush=True)
    for name in lock["files"]:
        p = staging / name
        lock["files"][name] = {"bytes": p.stat().st_size, "sha256": sha256(p)}

    import numpy as np
    from safetensors import safe_open

    with safe_open(str(staging / "model.safetensors"), framework="pt", device="cpu") as f:
        cb = f.get_tensor("codebook_embeddings.weight")
    arr = np.asarray(cb.float().contiguous().numpy(), dtype="<f4")
    lock["sidecar_payload_sha256"] = hashlib.sha256(memoryview(arr)).hexdigest()
    del cb, arr

    sys.argv = ["convert_audio8_realistic.py", "--model-dir", str(staging), "--llama-cpp",
                str(args.llama_cpp.resolve()), "--output-dir", str(out), "--threads", str(args.threads)]
    conv.main()
    shutil.rmtree(staging)
    print(f"[gguf] F32 bundle: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
