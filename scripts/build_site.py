#!/usr/bin/env python3
"""Copia y convierte a MP3 los audios que usa la web (docs/index.html).

    python scripts/build_site.py --model checkpoints/audio8_eu_sft_v4/export

Lee:   <model>/listen/*.wav          (scripts/synth_checkpoints.py --checkpoint <model>)
       checkpoints/baseline_listen/  (scripts/synth_checkpoints.py --checkpoint base)
       dataset_processed/openslr76/sft/val_prompts.jsonl  (audios de referencia)
Escribe: docs/audio/{ft,base,ref}/*.mp3   (mono, 96 kb/s)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "audio"

# Muestras que usa docs/index.html (si cambias esto, cambia también el HTML)
FEATURED = ["euf_07505_s04", "eum_01523_s05"]
COMPARE = ["euf_07505_s00", "eum_01523_s02", "euf_07505_s03", "eum_01523_s06"]
CLONING = {"euf_07505": ["euf_07505_s01", "euf_07505_s07"], "eum_01523": ["eum_01523_s01", "eum_01523_s03"]}
# v4 (released): eum_01523_s07 ran away and noref_s00 is too slow (7.8 chars/s); not shown,
# but the failure rate is reported on the page. (v2: noref_s00, noref_s01 and eum_01523_s07 failed.)
NOREF = ["noref_s01", "noref_s02"]
# "What can a reference fix?": the Antton test (checkpoints/my_voice/antton, v4, seed 7).
# Only take 2 with Antton in the reference came out right; take 0 without it is a typical miss.
EXTRA = {
    "antton/ref.mp3": "checkpoints/my_voice/antton/erref_antton_barne__REF.wav",
    "antton/ok.mp3": "checkpoints/my_voice/antton/erref_antton_barne__Antton__2.wav",
    "antton/bad.mp3": "checkpoints/my_voice/antton/erref_antton_gabe__Antton__0.wav",
    # replaced after listening: picked from 6 new takes (checkpoints/site_takes, seed 101)
    "ft/euf_07505_s00.mp3": "checkpoints/site_takes/euf_07505_s00__take3.wav",
    "ft/euf_07505_s01.mp3": "checkpoints/site_takes/euf_07505_s01__take1.wav",
}


def to_mp3(src: Path, dst: Path) -> None:
    if not src.is_file():
        sys.exit(f"falta {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1",
                    "-codec:a", "libmp3lame", "-b:a", "96k", str(dst)], check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="checkpoint con listen/ ya sintetizado")
    ap.add_argument("--baseline", type=Path, default=ROOT / "checkpoints" / "baseline_listen")
    ap.add_argument("--prompts", type=Path,
                    default=ROOT / "dataset_processed" / "openslr76" / "sft" / "val_prompts.jsonl")
    args = ap.parse_args()

    listen = args.model / "listen"
    prompts = {json.loads(l)["id"]: json.loads(l) for l in args.prompts.open(encoding="utf-8")}
    if DOCS.exists():
        shutil.rmtree(DOCS)

    ft_ids = set(FEATURED + COMPARE + NOREF + [i for v in CLONING.values() for i in v])
    for sid in sorted(ft_ids):
        to_mp3(listen / f"{sid}.wav", DOCS / "ft" / f"{sid}.mp3")
    for sid in COMPARE:
        to_mp3(args.baseline / f"{sid}.wav", DOCS / "base" / f"{sid}.mp3")
    for spk, ids in CLONING.items():
        ref = (args.prompts.parent / prompts[ids[0]]["reference_audio"]).resolve()
        to_mp3(ref, DOCS / "ref" / f"{spk}.mp3")

    for dst, src in EXTRA.items():
        to_mp3(ROOT / src, DOCS / dst)

    files = sorted(DOCS.rglob("*.mp3"))
    size = sum(f.stat().st_size for f in files) / 2**20
    print(f"[site] {len(files)} MP3 en {DOCS} ({size:.1f} MiB)")
    texts = {sid: prompts[sid]["text"] for sid in sorted(ft_ids)}
    texts.update({f"ref:{spk}": prompts[ids[0]]["reference_text"] for spk, ids in CLONING.items()})
    print(json.dumps(texts, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
