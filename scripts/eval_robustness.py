#!/usr/bin/env python3
"""Cloning strength + robustness over several seeds, in both modes (with / without reference).

For each model and seed, synthesizes val_prompts.jsonl and measures:
  * runaway: generation without EOS, or a speaking rate < MIN_CPS characters/s (a short
    sentence lasting 15–20 s, with long silences or babbling)
  * SIM (with reference only): cosine similarity between speaker embeddings
    (microsoft/wavlm-base-plus-sv) of the generated audio and of the reference.
    Reference values: SIM between two real recordings of the same speaker (ceiling)
    and between different speakers (floor).

Usage (on a different GPU from the one training):
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_robustness.py base checkpoints/audio8_eu_sft_v2/export ...
Results: checkpoints/robustness/<model>/s<seed>/*.wav  and  checkpoints/robustness/summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
INFER = ROOT / "repos" / "Audio8_TTS" / "audio8_tts_infer.py"
MIN_CPS = 8.0


def label(model: str) -> str:
    if model == "base":
        return "base"
    p = Path(model).resolve()
    return f"{p.parent.name}__{p.name}"   # audio8_eu_sft_v2__export


def synth(model_path: str, out: Path, prompts: Path, seed: int, batch: int) -> None:
    if (out / "manifest.jsonl").is_file():
        return
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-W", "ignore", str(INFER), "--model", model_path,
                    "--input-jsonl", str(prompts), "--output-dir", str(out),
                    "--batch-size", str(batch), "--seed", str(seed)],
                   cwd=INFER.parent, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class SpeakerEmbedder:
    def __init__(self, device: str):
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.torch = torch
        self.fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        self.model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device).eval()
        self.device = device
        self.cache: dict[str, object] = {}

    def __call__(self, path: Path):
        key = str(path)
        if key not in self.cache:
            import librosa

            y, _ = librosa.load(path, sr=16000, mono=True)
            y, _ = librosa.effects.trim(y, top_db=40)
            inp = self.fe(y, sampling_rate=16000, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                e = self.model(**inp).embeddings[0]
            self.cache[key] = self.torch.nn.functional.normalize(e, dim=-1).cpu()
        return self.cache[key]

    def sim(self, a: Path, b: Path) -> float:
        return float((self(a) * self(b)).sum())


def speaking_rate(path: Path, text: str) -> float:
    import librosa
    import soundfile as sf

    y, sr = sf.read(path, dtype="float32")
    _, (s, e) = librosa.effects.trim(y, top_db=40)
    return len(text.replace(" ", "")) / max((e - s) / sr, 1e-3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="'base' or checkpoint paths")
    ap.add_argument("--prompts", type=Path, default=ROOT / "dataset_processed/openslr76/sft/val_prompts.jsonl")
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44])
    ap.add_argument("--out-dir", type=Path, default=ROOT / "checkpoints" / "robustness")
    ap.add_argument("--base", default=os.environ.get("MODEL", "Edge0/Audio8-TTS-Realistic-Air"))
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    import torch
    from synth_checkpoints import base_snapshot, complete_checkpoint

    prompts = [json.loads(l) for l in args.prompts.open(encoding="utf-8")]
    pdir = args.prompts.parent
    # absolute reference paths + every sentence also without a reference (more no-ref samples)
    for p in prompts:
        if p.get("reference_audio"):
            p["reference_audio"] = str((pdir / p["reference_audio"]).resolve())
    have_noref = {p["text"] for p in prompts if not p.get("reference_audio")}
    for i, text in enumerate(dict.fromkeys(p["text"] for p in prompts)):
        if text not in have_noref:
            prompts.append({"id": f"noref_x{i:02d}", "text": text})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    full_prompts = args.out_dir / "prompts.jsonl"
    full_prompts.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in prompts), encoding="utf-8")
    args.prompts = full_prompts
    emb = SpeakerEmbedder("cuda" if torch.cuda.is_available() else "cpu")

    # reference values from real recordings: same speaker (ceiling) / different speaker (floor)
    wavs = ROOT / "dataset_processed/openslr76/wavs"
    refs = {p["id"].rsplit("_s", 1)[0]: (pdir / p["reference_audio"]).resolve()
            for p in prompts if p.get("reference_audio")}
    ceiling, floor = [], []
    for spk, ref in refs.items():
        others = sorted(w for w in wavs.glob(f"{spk}_*.wav") if w.resolve() != ref)[:20]
        ceiling += [emb.sim(ref, w) for w in others]
        for spk2, ref2 in refs.items():
            if spk2 != spk:
                floor += [emb.sim(ref, w) for w in sorted(wavs.glob(f"{spk2}_*.wav"))[:20]]

    base_snap = None
    summary = {"seeds": args.seeds, "min_cps": MIN_CPS,
               "sim_real_same_speaker": round(st.mean(ceiling), 3),
               "sim_real_other_speaker": round(st.mean(floor), 3), "models": {}}
    print(f"[ref] SIM real recordings: same speaker {summary['sim_real_same_speaker']:.3f} | "
          f"different speaker {summary['sim_real_other_speaker']:.3f}")

    for model in args.models:
        path = args.base if model == "base" else str(Path(model).resolve())
        if model != "base":
            base_snap = base_snap or base_snapshot(args.base)
            complete_checkpoint(Path(path), base_snap)
        name = label(model)
        res = {"ref": {"n": 0, "runaway": 0, "sims": []}, "noref": {"n": 0, "runaway": 0}}
        for seed in args.seeds:
            out = args.out_dir / name / f"s{seed}"
            synth(path, out, args.prompts, seed, args.batch_size)
            status = {json.loads(l)["id"]: json.loads(l)["status"] for l in (out / "manifest.jsonl").open()}
            for p in prompts:
                wav = out / f"{p['id']}.wav"
                mode = "ref" if p.get("reference_audio") else "noref"
                bad = status.get(p["id"]) != "OK" or not wav.is_file() or speaking_rate(wav, p["text"]) < MIN_CPS
                res[mode]["n"] += 1
                res[mode]["runaway"] += int(bad)
                if mode == "ref" and not bad:
                    res["ref"]["sims"].append(emb.sim(wav, (pdir / p["reference_audio"]).resolve()))
        sims = res["ref"].pop("sims")
        res["ref"]["sim_mean"] = round(st.mean(sims), 3) if sims else None
        res["ref"]["sim_median"] = round(st.median(sims), 3) if sims else None
        summary["models"][name] = res
        r, n = res["ref"], res["noref"]
        print(f"{name:34s} SIM {r['sim_mean']}  | runaway with ref {r['runaway']}/{r['n']} "
              f"({100*r['runaway']/r['n']:.0f}%) | without ref {n['runaway']}/{n['n']} ({100*n['runaway']/n['n']:.0f}%)",
              flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prev = {}
    if (args.out_dir / "summary.json").is_file():
        prev = json.loads((args.out_dir / "summary.json").read_text()).get("models", {})
    summary["models"] = {**prev, **summary["models"]}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
