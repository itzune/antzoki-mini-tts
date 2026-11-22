#!/usr/bin/env python3
"""Build a Hugging Face–ready folder from a trained checkpoint.

    python scripts/export_hf.py --checkpoint checkpoints/audio8_eu_sft_v2/export \
        --repo-id <user>/audio8-tts-eu [--run checkpoints/audio8_eu_sft_v2]

Output: hf_release/<repo-name>/  (git-ignored)
  model weights + codec + remote code + tokenizer/processor   (loadable with AutoModel/AutoProcessor)
  eu_normalize.py      Basque text normalizer used in training (apply it at inference too)
  samples/*.wav        a few no-reference samples (no cloned volunteer voices)
  README.md            model card with metadata, metrics, usage, licenses
  LICENSE, NOTICE

Copies only an explicit allowlist: training_args.bin, optimizer states, etc. are never copied
(they contain local paths).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = (
    "model.safetensors", "config.json", "generation_config.json", "codec.pth",
    "configuration_arktts.py", "modeling_arktts.py", "modeling_arktts_codec.py",
    "processing_arktts.py", "processor_config.json", "preprocessor_config.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
)
REQUIRED = ("model.safetensors", "config.json", "codec.pth", "modeling_arktts.py", "tokenizer.json")


def git_commit(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def load_json(path: Path) -> dict | list | None:
    return json.loads(path.read_text()) if path.is_file() else None


def metrics_table(rows: list[dict] | None) -> str:
    if not rows:
        return "_Run `python scripts/eval_loss.py --run <run_dir> --out <run_dir>/eval_loss_compare.json` to fill this table._"
    lines = ["| Model | eval loss | slow loss | fast loss | slow acc |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        name = "base (Audio8-TTS-Realistic-Air)" if r["model"] == "base" else Path(r["model"]).name
        lines.append(f"| {name} | {r['eval_loss']:.3f} | {r['slow_loss']:.3f} | {r['fast_loss']:.3f} | {r['slow_acc']:.3f} |")
    return "\n".join(lines)


CARD = """---
license: apache-2.0
language:
  - eu
library_name: transformers
pipeline_tag: text-to-speech
base_model: Edge0/Audio8-TTS-Realistic-Air
base_model_relation: finetune
datasets:
  - openslr/76
tags:
  - text-to-speech
  - tts
  - basque
  - euskara
  - voice-cloning
  - audio8
---

# {title}

Basque (*euskara*) text-to-speech: a full supervised fine-tune of
[Edge0/Audio8-TTS-Realistic-Air](https://huggingface.co/Edge0/Audio8-TTS-Realistic-Air)
(0.6B, DualAR, 44.1 kHz codec, zero-shot voice cloning). Basque is not one of the base
model's 23 languages, so this release adapts it to a new language.

> **Status: experimental.** Trained on about 9 hours of read speech from one corpus. Listen to
> the samples before you use it, and expect errors on rare words, acronyms and code-switching.

## Usage

```bash
git clone https://github.com/Edge0-AI/Audio8_TTS && cd Audio8_TTS
pip install -r requirements.txt
```

```python
import sys, soundfile as sf, torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoProcessor

repo = "{repo_id}"
sys.path.insert(0, hf_hub_download(repo, "eu_normalize.py").rsplit("/", 1)[0])
from eu_normalize import normalize          # same normalizer as in training

processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
model = AutoModel.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()

text = normalize("Donostiako zinemaldia 1953an sortu zen.")   # -> "... mila bederatziehun eta berrogeita hamahiruan ..."
inputs = processor(text=[text], return_tensors="pt")
# voice cloning: processor(text=[text], reference_audio=["ref.wav"], reference_text=["exact transcript"], return_tensors="pt")
inputs = {{k: v.cuda() for k, v in inputs.items()}}
out = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.8, top_p=0.95,
                     top_k=50, return_dict_in_generate=True)
wav, lengths = model.decode_audio(out.codes)
sf.write("out.wav", wav[0, : int(lengths[0])].float().cpu().numpy(), model.config.codec_sample_rate)
```

Or with the CLI from the Audio8 repo:
`python audio8_tts_infer.py --model {repo_id} --text "..." [--reference-audio ref.wav --reference-text "..."] --output out.wav`

**Always normalize the text with `eu_normalize.normalize()`.** The model never saw digits
during training: numbers become Basque words that keep their case suffix
(`2008an` → `bi mila eta zortzian`, `1950eko` → `mila bederatziehun eta berrogeita hamarreko`).
Keep each input under about 150 characters.

## Training

| | |
|---|---|
| Data | [OpenSLR 76](https://www.openslr.org/76/): crowdsourced Basque read speech. After filtering: {train_rows} training utterances ({train_hours} h), {train_speakers} speakers |
| Held out | speakers {holdout}, used for evaluation and never trained on |
| Audio prep | silence trimming, resampled to 44.1 kHz, loudness set to −20 LUFS (−1 dBFS peak), speaking-rate outlier filter |
| Voice cloning | {ref_pct}% of training rows paired with another utterance of the same speaker as the reference prompt |
| Method | full SFT of the slow and fast AR branches, bf16, DeepSpeed ZeRO-2, 1× RTX 5090 (24 GB) |
| Hyper-params | {hparams} |
| Pipeline | {github} (Audio8_TTS @ `{audio8_commit}`) |

## Evaluation

Teacher-forced loss on {eval_rows} utterances from the held-out speakers (lower is better).

{metrics}

A first run (2e-5 learning rate, 6 epochs) reached its best held-out loss after about 250 steps
and then overfit: OpenSLR 76 has many speakers reading the same sentences. This release comes
from the shorter, early-stopped run.

## Samples

{samples}

## Limitations and responsible use

- Small single-domain corpus (read sentences): spontaneous or expressive speech, long inputs,
  dialects and code-switching are out of distribution.
- Acronyms and alphanumeric names (`U2`, `AP-68`) are read with Basque digit names.
- Voice cloning can imitate real people. Do not clone a voice without the speaker's consent,
  and do not use this model to impersonate anyone or to deceive.

## License and attribution

- Model weights and code: **Apache-2.0**, as for the base model (see `LICENSE` and `NOTICE`).
- Training data: **OpenSLR 76** by Google, CC BY-SA 4.0. *Kjartansson, O., Gutkin, A., Butryna, A.,
  Demirsahin, I., Rivera, C. (2020). Open-Source High Quality Speech Datasets for Basque, Catalan and
  Galician. SLTU-CCURL 2020.* Check whether the ShareAlike terms apply to your use.
- Base model: [Edge0/Audio8-TTS-Realistic-Air](https://huggingface.co/Edge0/Audio8-TTS-Realistic-Air) (Apache-2.0).
- Data recipe inspired by [itzune/antzoki-tts](https://github.com/itzune/antzoki-tts).
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True, help="checkpoint-N/ or export/ directory")
    ap.add_argument("--repo-id", required=True, help="e.g. user/audio8-tts-eu")
    ap.add_argument("--run", type=Path, help="run dir (reads run_config.env and eval_loss_compare.json)")
    ap.add_argument("--stats", type=Path, default=ROOT / "dataset_processed" / "openslr76" / "sft" / "stats.json")
    ap.add_argument("--github", default="", help="URL of this project's GitHub repo, for the card")
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    ckpt = args.checkpoint.resolve()
    missing = [f for f in REQUIRED if not (ckpt / f).is_file()]
    if missing:
        # intermediate checkpoints don't have the processor: complete them from the base model
        sys.path.insert(0, str(ROOT / "scripts"))
        from synth_checkpoints import base_snapshot, complete_checkpoint
        complete_checkpoint(ckpt, base_snapshot("Edge0/Audio8-TTS-Realistic-Air"))
        missing = [f for f in REQUIRED if not (ckpt / f).is_file()]
        if missing:
            sys.exit(f"{ckpt} is missing {missing}")

    name = args.repo_id.split("/")[-1]
    out = (args.out or ROOT / "hf_release" / name).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for f in MODEL_FILES:
        if (ckpt / f).is_file():
            shutil.copy2(ckpt / f, out / f)
    shutil.copy2(ROOT / "scripts" / "eu_normalize.py", out / "eu_normalize.py")
    shutil.copy2(ROOT / "repos" / "Audio8_TTS" / "LICENSE", out / "LICENSE")
    notice = (ROOT / "repos" / "Audio8_TTS" / "NOTICE").read_text()
    (out / "NOTICE").write_text(
        notice.rstrip() + "\n\nThis is a fine-tuned derivative of Edge0/Audio8-TTS-Realistic-Air.\n"
        "Fine-tuned on OpenSLR 76 (Google, CC BY-SA 4.0): https://www.openslr.org/76/\n"
    )

    # samples: no-reference outputs only (a random voice, not a cloned volunteer)
    listen = ckpt / "listen"
    samples_md = "_No samples: run `python scripts/synth_checkpoints.py --checkpoint <ckpt>` before exporting._"
    prompts = {}
    pfile = ROOT / "dataset_processed" / "openslr76" / "sft" / "val_prompts.jsonl"
    if pfile.is_file():
        prompts = {json.loads(l)["id"]: json.loads(l)["text"] for l in pfile.open(encoding="utf-8")}
    wavs = sorted(listen.glob("noref_*.wav"))[: args.n_samples] if listen.is_dir() else []
    if wavs:
        (out / "samples").mkdir()
        rows = []
        for w in wavs:
            shutil.copy2(w, out / "samples" / w.name)
            rows.append(f"- **{prompts.get(w.stem, w.stem)}**\n  <audio controls src=\"samples/{w.name}\"></audio>")
        samples_md = "No reference audio (the model picks a voice):\n\n" + "\n".join(rows)

    stats = load_json(args.stats) or {}
    env = {}
    if args.run and (args.run / "run_config.env").is_file():
        for line in (args.run / "run_config.env").read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k] = v
    hp = ", ".join(f"{k.lower()}={env[k]}" for k in (
        "LEARNING_RATE", "NUM_TRAIN_EPOCHS", "BATCH_SIZE", "GRADIENT_ACCUMULATION_STEPS",
        "WARMUP_RATIO", "LR_SCHEDULER_TYPE", "WEIGHT_DECAY") if k in env)
    if env.get("EXTRA_ARGS"):
        hp += f", extra: `{env['EXTRA_ARGS']}`"
    metrics = load_json(args.run / "eval_loss_compare.json") if args.run else None
    tr = stats.get("train", {})

    card = CARD.format(
        title=name, repo_id=args.repo_id,
        train_rows=tr.get("rows", "?"), train_hours=tr.get("hours", "?"),
        train_speakers=stats.get("train_speakers", "?"),
        holdout=", ".join(f"`{s}`" for s in stats.get("holdout_speakers", [])) or "?",
        ref_pct=round(100 * tr["with_ref"] / tr["rows"]) if tr.get("rows") else "?",
        eval_rows=stats.get("eval", {}).get("rows", "?"),
        hparams=hp or "see run_config.env",
        github=args.github or "_(add the GitHub URL with --github)_",
        audio8_commit=git_commit(ROOT / "repos" / "Audio8_TTS"),
        metrics=metrics_table(metrics), samples=samples_md,
    )
    (out / "README.md").write_text(card, encoding="utf-8")

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 2**30
    print(f"[export] {out}  ({size:.2f} GiB)")
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print(f"   {p.relative_to(out)}  {p.stat().st_size / 2**20:.1f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
