#!/usr/bin/env bash
# =============================================================================
#  Quantize the slow/fast AR GGUFs of an Audio8 F32 bundle and assemble a new bundle.
#
#  bash scripts/gguf/quantize_eu.sh gguf/antzoki-mini-tts-f32 Q4_K_M [gguf/antzoki-mini-tts-q4_k_m]
#
#  * The RoPE table (audio8.rope_coeffs.weight) stays F32: the native runtime requires it.
#  * The codec stays F32 (the runtime only supports F32 codec weights) and the codebook
#    sidecar is a raw F32 file; both are hard-linked from the F32 bundle (no extra disk).
#  * audio8-runtime.json gets the new paths and sizes (the runtime checks sizes, not hashes).
#  Note: dim 896 is not a multiple of 256, so K-quants fall back to Q5_0/Q8_0 for many
#  tensors; Q4_0 is a "pure" 4-bit alternative.
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${1:?usage: quantize_eu.sh <f32-bundle> <TYPE> [out-dir]}"
TYPE="${2:?quant type, e.g. Q4_K_M or Q4_0}"
lower="$(echo "$TYPE" | tr 'A-Z' 'a-z')"
OUT="${3:-${SRC%-f32}-$lower}"
QUANT="${QUANT:-$ROOT/vendor/build-cpu/bin/llama-quantize}"
THREADS="${THREADS:-$(nproc)}"

[[ -f "$SRC/audio8-runtime.json" ]] || { echo "not an Audio8 bundle: $SRC" >&2; exit 2; }
[[ -e "$OUT" ]] && { echo "$OUT already exists" >&2; exit 2; }
mkdir -p "$OUT/models"

for b in slow fast; do
  "$QUANT" --tensor-type "rope_coeffs=f32" "$SRC/models/$b-f32.gguf" "$OUT/models/$b-$lower.gguf" "$TYPE" "$THREADS" \
    > "$OUT/models/.$b-quantize.log" 2>&1 || { cat "$OUT/models/.$b-quantize.log" >&2; exit 1; }
  grep -E "quant size" "$OUT/models/.$b-quantize.log" | sed "s/^/[$b] /"
done
rm -f "$OUT"/models/.*-quantize.log
ln "$SRC/models/audio8-realistic-codec-f32.gguf" "$OUT/models/" 2>/dev/null || cp "$SRC/models/audio8-realistic-codec-f32.gguf" "$OUT/models/"
ln "$SRC/models/codebook_embeddings.f32" "$OUT/models/" 2>/dev/null || cp "$SRC/models/codebook_embeddings.f32" "$OUT/models/"
cp "$SRC/LICENSE" "$SRC/NOTICE" "$OUT/"

python3 - "$SRC" "$OUT" "$lower" "$TYPE" <<'EOF'
import hashlib, json, sys
from pathlib import Path
src, out, lower, qtype = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        while c := f.read(8 << 20): h.update(c)
    return h.hexdigest()
rt = json.loads((src / "audio8-runtime.json").read_text())
paths = {"slow": f"models/slow-{lower}.gguf", "fast": f"models/fast-{lower}.gguf",
         "codebook_embeddings": "models/codebook_embeddings.f32", "codec": "models/audio8-realistic-codec-f32.gguf"}
for k, rel in paths.items():
    p = out / rel
    rt["assets"][k] = {"path": rel, "bytes": p.stat().st_size, "sha256": sha(p)}
rt["quantization"] = {"ar": qtype, "codec": "F32", "rope_tables": "F32",
                      "source_bundle": src.name, "note": "quantized after conversion with llama-quantize"}
(out / "audio8-runtime.json").write_text(json.dumps(rt, indent=2) + "\n")
(out / "README.md").write_text(
    f"# antzoki-mini-tts — llama.cpp bundle ({qtype} AR, F32 codec)\n\n"
    "Requires the Audio8-patched llama.cpp runtime (repos/Audio8_TTS/llama_cpp).\n"
    "Text must be normalized with eu_normalize.normalize() first.\n")
sums = "".join(f"{sha(p)}  {p.relative_to(out).as_posix()}\n"
               for p in sorted(out.rglob("*")) if p.is_file() and p.name != "SHA256SUMS")
(out / "SHA256SUMS").write_text(sums)
EOF
du -sh --apparent-size "$OUT"
echo "[gguf] $OUT"
