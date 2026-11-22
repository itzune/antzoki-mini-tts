#!/usr/bin/env python3
"""Basque TTS with pace control: best-of-N takes by speaking rate (+ optional time-stretch).

The model has no speed parameter, and without a reference its pace varies a lot from seed
to seed (the same sentence came out between 8 and 29 characters/s). So:

  1. generate N candidate takes per sentence (one batched call),
  2. drop the broken ones (no EOS, or below 8 / above 20 characters/s),
  3. keep the one whose pace is closest to --cps (default 11.5 = corpus median),
  4. optionally --fit: time-stretch the chosen take to exactly --cps (pitch kept),
     or --speed X to stretch it by a fixed factor.

With a reference (--ref + --ref-text) the model also copies the reference's pace, so a
reference read slowly gives slower output (measured: ~60–75 % of the reference pace carries over).

  python scripts/say.py "Kaixo, zer moduz zaude gaur?"                         # no reference
  python scripts/say.py "Kaixo!" --cps 10 --fit -o kaixo.wav                   # slower, exact
  python scripts/say.py --texts-file esaldiak.txt --out-dir irteera/ -n 6
  python scripts/say.py "Kaixo!" --ref nire_ahotsa.wav --ref-text "..."        # cloning

Slow / normal / fast ≈ 9 / 11.5 / 14 characters per second (spaces not counted).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from clone_my_voice import SR, load_model, prepare_reference, speaking_rate, synth_batch, tempo  # noqa: E402
from eu_normalize import normalize  # noqa: E402

MIN_CPS, MAX_CPS = 8.0, 20.0
DEFAULT_MODEL = ROOT / "checkpoints" / "audio8_eu_sft_v4" / "export"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="*", help="sentence(s) to say")
    ap.add_argument("--texts-file", type=Path, help="one sentence per line")
    ap.add_argument("-o", "--out", type=Path, help="output wav (single sentence)")
    ap.add_argument("--out-dir", type=Path, default=Path("say_out"), help="output folder (several sentences)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="'base' or an export/checkpoint folder")
    ap.add_argument("--ref", type=Path, help="reference recording for voice cloning")
    ap.add_argument("--ref-text", help="exact transcript of --ref")
    ap.add_argument("-n", "--candidates", type=int, default=4, help="takes generated per sentence")
    ap.add_argument("--cps", type=float, default=11.5, help="target pace in characters/s (9 slow, 14 fast)")
    ap.add_argument("--fit", action="store_true", help="time-stretch the chosen take to exactly --cps")
    ap.add_argument("--speed", type=float, default=1.0, help="extra fixed time-stretch factor (0.5–2.0)")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--keep-candidates", action="store_true", help="also save every take")
    ap.add_argument("--base", default=os.environ.get("MODEL", "Edge0/Audio8-TTS-Realistic-Air"))
    args = ap.parse_args()

    texts = list(args.text)
    if args.texts_file:
        texts += [l.strip() for l in args.texts_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not texts:
        ap.error("give a sentence or --texts-file")
    if bool(args.ref) != bool(args.ref_text):
        ap.error("--ref and --ref-text go together")
    if args.out and len(texts) > 1:
        ap.error("-o is for one sentence; use --out-dir for several")

    outs = [args.out] if args.out else [args.out_dir / f"s{i:02d}.wav" for i in range(len(texts))]
    for o in outs:
        o.parent.mkdir(parents=True, exist_ok=True)

    ref = ref_text = None
    if args.ref:
        ref = outs[0].parent / "_ref_prepared.wav"
        prepare_reference(args.ref, ref, 1.0)
        ref_text = normalize(args.ref_text)
        rcps, _ = speaking_rate(ref, ref_text)
        print(f"[ref] {rcps:.1f} kar/s — the output tends toward this pace")

    import soundfile as sf

    proc, model = load_model(args.model, args.base)
    norm = [normalize(t) for t in texts]
    per_call = max(1, 16 // args.candidates)          # sentences per batched call
    for start in range(0, len(texts), per_call):
        chunk = list(range(start, min(start + per_call, len(texts))))
        batch = [norm[i] for i in chunk for _ in range(args.candidates)]
        res = synth_batch(proc, model, batch, ref, ref_text, args.seed + start, args.temperature)
        for j, i in enumerate(chunk):
            takes = []
            for k in range(args.candidates):
                finished, wav = res[j * args.candidates + k]
                tmp = outs[i].with_name(f"{outs[i].stem}_take{k}.wav")
                sf.write(tmp, wav, SR)
                cps, dur = speaking_rate(tmp, norm[i])
                ok = finished and MIN_CPS <= cps <= MAX_CPS
                takes.append((not ok, abs(cps - args.cps), cps, dur, tmp))
            takes.sort(key=lambda t: (t[0], t[1]))
            bad, _, cps, dur, best = takes[0]
            factor = args.speed * (min(max(args.cps / cps, 0.75), 1.33) if args.fit and not bad else 1.0)
            if abs(factor - 1.0) > 0.01:
                tempo(best, outs[i], factor)
            else:
                best.replace(outs[i]) if not args.keep_candidates else sf.write(outs[i], sf.read(best)[0], SR)
            if not args.keep_candidates:
                for t in takes:
                    t[4].unlink(missing_ok=True)
            all_cps = " ".join(f"{t[2]:.1f}{'' if not t[0] else '✗'}" for t in sorted(takes, key=lambda t: t[4].name))
            final = cps * factor
            print(f"{outs[i]}  {final:4.1f} kar/s{'  [!] take guztiak txarrak' if bad else ''}"
                  f"   (hautagaiak: {all_cps})  «{texts[i]}»", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
