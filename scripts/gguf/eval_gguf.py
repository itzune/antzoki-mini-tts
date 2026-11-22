#!/usr/bin/env python3
"""Compare llama.cpp bundles (F32 vs quantized) on CPU: runaways, pace, speaker SIM, speed.

  python scripts/gguf/eval_gguf.py gguf/antzoki-mini-tts-f32 gguf/antzoki-mini-tts-q4_k_m gguf/antzoki-mini-tts-q4_0

Same 8 test sentences as eval_robustness.py, with the held-out reference voice euf_07505 and
without a reference, for each --seeds. Runs --jobs llama-tts processes in parallel
(--threads each). Output: gguf/eval/<bundle>/<mode>_<sid>_s<seed>.wav and gguf/eval/summary.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from eu_normalize import normalize  # noqa: E402
from make_sft_splits import LISTEN_SENTENCES  # noqa: E402

TTS = ROOT / "vendor" / "build-cpu" / "bin" / "llama-tts"
MIN_CPS = 8.0


def run(bundle: Path, out: Path, text: str, seed: int, threads: int, ref: tuple[str, str] | None) -> dict:
    wav, rep = out.with_suffix(".wav"), out.with_suffix(".json")
    if not rep.is_file():
        wav.unlink(missing_ok=True)
        cmd = [str(TTS), "--audio8-model-dir", str(bundle), "--text", text, "--output", str(wav),
               "--gpu-layers", "0", "--threads", str(threads), "--seed", str(seed),
               "--max-frames", "512", "--report", str(rep)]
        if ref:
            cmd += ["--reference-audio", ref[0], "--reference-text", ref[1]]
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            return {"error": r.stderr.strip().splitlines()[-1] if r.stderr.strip() else f"exit {r.returncode}"}
    d = json.loads(rep.read_text())
    return {k: d[k] for k in ("stop_reason", "samples", "total_ms", "ar_ms", "codec_decode_ms", "reference_encode_ms")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundles", nargs="+", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "gguf" / "eval")
    args = ap.parse_args()

    prompts = [json.loads(l) for l in (ROOT / "dataset_processed/openslr76/sft/val_prompts.jsonl").open()]
    p0 = next(p for p in prompts if p["id"].startswith("euf_07505"))
    ref = (str((ROOT / "dataset_processed/openslr76/sft" / p0["reference_audio"]).resolve()), p0["reference_text"])
    texts = [(f"s{i:02d}", normalize(s)) for i, s in enumerate(LISTEN_SENTENCES)]

    tasks = []
    for b in args.bundles:
        d = args.out_dir / b.name
        d.mkdir(parents=True, exist_ok=True)
        for mode in ("ref", "noref"):
            for sid, text in texts:
                for seed in args.seeds:
                    tasks.append((b, d / f"{mode}_{sid}_s{seed}", text, seed, mode))
    print(f"[eval] {len(tasks)} generations, {args.jobs} jobs × {args.threads} threads", flush=True)
    with ThreadPoolExecutor(args.jobs) as ex:
        results = list(ex.map(lambda t: (t, run(t[0], t[1], t[2], t[3], args.threads, ref if t[4] == "ref" else None)), tasks))

    import soundfile  # noqa: F401  (fail early if missing)
    import torch
    from eval_robustness import SpeakerEmbedder, speaking_rate

    emb = SpeakerEmbedder("cuda" if torch.cuda.is_available() else "cpu")
    summary = {}
    for b in args.bundles:
        res = {m: {"n": 0, "runaway": 0, "errors": 0, "sims": [], "cps": [], "rtf": []} for m in ("ref", "noref")}
        for (bb, out, text, seed, mode), r in results:
            if bb != b:
                continue
            s = res[mode]
            s["n"] += 1
            if "error" in r:
                s["errors"] += 1
                s["runaway"] += 1
                print(f"  ! {out.name}: {r['error']}")
                continue
            wav = out.with_suffix(".wav")
            cps = speaking_rate(wav, text)
            bad = r["stop_reason"] != "eos" or cps < MIN_CPS
            s["runaway"] += int(bad)
            if not bad:
                s["cps"].append(cps)
                s["rtf"].append(r["total_ms"] / 1000 / (r["samples"] / 44100))
                if mode == "ref":
                    s["sims"].append(emb.sim(wav, Path(ref[0])))
        for m, s in res.items():
            sims = s.pop("sims")
            s["sim_mean"] = round(st.mean(sims), 3) if sims else None
            c, t = s.pop("cps"), s.pop("rtf")
            s["cps_median"] = round(st.median(c), 1) if c else None
            s["rtf_median"] = round(st.median(t), 2) if t else None
        summary[b.name] = res
        r, n = res["ref"], res["noref"]
        print(f"{b.name:24s} SIM {r['sim_mean']} | runaway ref {r['runaway']}/{r['n']} noref {n['runaway']}/{n['n']} "
              f"| cps {r['cps_median']}/{n['cps_median']} | RTF {r['rtf_median']}/{n['rtf_median']}", flush=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
