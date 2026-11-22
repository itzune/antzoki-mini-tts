# TTS_basque — fine-tuning Audio8-TTS for Basque (euskara)

This project fine-tunes [Audio8-TTS-Realistic-Air](https://huggingface.co/Edge0/Audio8-TTS-Realistic-Air)
(0.6B, zero-shot voice cloning, 44.1 kHz) for Basque, a language the base model does not support.
It trains on the [OpenSLR 76](https://www.openslr.org/76/) corpus with a data recipe adapted from
[itzune/antzoki-tts](https://github.com/itzune/antzoki-tts).

Everything runs on a single 24 GB GPU (RTX 5090). Preparing the data takes about 15 minutes and a
training run takes 15–35 minutes.

## Layout

```
setup_env.sh            system packages + uv + .venv (Python 3.11, torch 2.8.0+cu128) + deps
verify_env.py           environment checks (GPU, audio libs, espeak-ng, Audio8 import/tests, synthesis)
requirements-env.txt    Python deps · constraints-env.txt pins torch/transformers · requirements.lock.txt exact versions
repos/                  git submodules, pinned: Audio8_TTS (trainer and inference) and antzoki-tts (reference)
scripts/
  env.sh                  `source` it: activates .venv, picks the GPU, sets env vars
  eu_normalize.py         Basque text normalizer: numbers become words and keep their case suffix
  prepare_openslr76.py    step 1: trim, resample to 44.1 kHz, set −20 LUFS, normalize text, filter
  make_sft_splits.py      step 3: held-out speakers, voice-cloning reference pairs, length checks
  run_data_pipeline.sh    steps 1–3 in one go (step 2 = Audio8 codec encoding)
  train_sft.sh            step 4: full SFT (DeepSpeed ZeRO-2, bf16)
  eval_loss.py            held-out loss: base model vs checkpoints
  synth_checkpoints.py    listening tests on new sentences, for each checkpoint
  eval_robustness.py      cloning similarity (SIM) and runaway rate over several seeds
  clone_my_voice.py       clone your own voice and compare models on a blind A/B listening page
  say.py                  synthesis with pace control (best-of-N takes, optional time-stretch)
  export_hf.py            build a Hugging Face model folder with a model card
  upload_hf.sh            upload it (private by default)
dataset_raw/ dataset_processed/ checkpoints/   git-ignored; rebuilt by the scripts
```

## Quick start

```bash
git clone --recurse-submodules <this-repo> && cd TTS_basque
bash setup_env.sh                       # asks for sudo to install ffmpeg, sox, espeak-ng, libsndfile
source scripts/env.sh
python verify_env.py --run-tests

# data (~3 GB download)
mkdir -p dataset_raw/openslr76 && cd dataset_raw/openslr76
for f in line_index_female.tsv line_index_male.tsv eu_es_female.zip eu_es_male.zip; do
  curl -LO https://www.openslr.org/resources/76/$f; done
unzip -q eu_es_female.zip -d audio && unzip -q eu_es_male.zip -d audio && cd ../..
bash scripts/run_data_pipeline.sh

# baseline, training, evaluation
python scripts/synth_checkpoints.py --checkpoint base
RUN_NAME=audio8_eu_sft_v2 LEARNING_RATE=1e-5 NUM_TRAIN_EPOCHS=2 SAVE_STEPS=100 EVAL_STEPS=100 SAVE_TOTAL_LIMIT=3 \
  bash scripts/train_sft.sh --load_best_model_at_end true --metric_for_best_model eval_loss --greater_is_better false
python scripts/eval_loss.py --run checkpoints/audio8_eu_sft_v2 --out checkpoints/audio8_eu_sft_v2/eval_loss_compare.json
python scripts/synth_checkpoints.py --run checkpoints/audio8_eu_sft_v2
```

## Environment notes

- **Python 3.11** comes from uv. A newer system Python (3.14) breaks deepspeed, numba and phonemizer.
- **torch 2.8.0 + CUDA 12.8** is the oldest build that supports Blackwell (sm_120), and it is the
  version DramaBox pins. torchaudio 2.8 still has the classic `load()`.
- **transformers 4.57.3** is the version the checkpoint was exported with (Audio8 needs `<5`).
- **DeepSpeed stays on even with one GPU.** The model loads in bf16, and ZeRO-2 keeps fp32 master
  weights. Without them, updates of about 1e-5 are lost. No CUDA ops are compiled, so a
  system `nvcc` 13.x is fine.
- **Multi-GPU machines.** `scripts/env.sh` sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`, because CUDA's
  default order ("fastest first") does not match `nvidia-smi`.
- **Phonemes are not used.** Audio8 reads BPE text. espeak-ng/phonemizer are installed for
  analysis only. Text normalization is what matters, so apply `eu_normalize.normalize()` at
  inference too.

## Data

| | |
|---|---|
| Corpus | OpenSLR 76: 7,136 utterances, 52 speakers, 48 kHz |
| After filtering | 7,092 utterances, **9.4 h of speech** (the raw clips carry about 2.2 s of silence each) |
| Train | 6,811 utterances, 50 speakers, 61 % with a same-speaker reference prompt |
| Eval | 200 utterances from 2 held-out speakers (1 F, 1 M) |

The same sentences are read by many speakers, so 193 of the 200 eval sentences also appear in
train, read by other speakers. The eval loss therefore mostly measures generalization to unseen
**voices**. The listening prompts (`val_prompts.jsonl`) use sentences that are not in the corpus.

## Results

Teacher-forced loss on the held-out speakers (lower is better):

| Model | eval loss | slow | fast |
|---|---:|---:|---:|
| Base (no fine-tune) | 12.73 | 5.42 | 7.31 |
| v1, step 250 (best) | **9.17** | 3.84 | 5.34 |
| v1, step 2556 (final; lr 2e-5, 6 epochs) | 10.70 | 5.17 | 5.53 |
| **v2, step 800 (best; lr 1e-5, 2 epochs)** | **9.09** | 3.80 | 5.29 |

v1 improved quickly and then overfit after about one epoch: train loss kept falling while the
held-out "slow" (semantic) loss went back up. v2 is the shorter, early-stopped run. Its held-out
loss improved at every evaluation.

**Cloning and robustness** (`scripts/eval_robustness.py`): 8 new sentences × (2 held-out reference
voices + no reference) × 4 seeds. SIM is the cosine similarity of WavLM speaker embeddings between
the output and the reference; between real recordings it is 0.970 for the same speaker and 0.673
for different speakers. A generation counts as a runaway when it has no EOS or a speaking rate
below 8 characters/s (a short sentence that lasts 15–20 s).

| Model | seeds | SIM ↑ | runaway, with ref | runaway, no ref |
|---|---:|---:|---:|---:|
| Base | 8 | **0.968** | 8/128 (6%) | 5/64 (8%) |
| v1 final | 4 | 0.948 | 2/64 (3%) | 2/32 (6%) |
| v2 best | 8 | 0.956 | 7/128 (5%) | 8/64 (12%) |
| v3 (dynamic refs, ratio 0.75) | 4 | 0.954 | 8/64 (12%) | 7/32 (22%) |
| v4 (fast AR frozen) | 8 | 0.957 | 4/128 (3%) | 8/64 (12%) |

- The base model already clones almost perfectly but has a non-Basque accent. Fine-tuning trades
  about 0.012 SIM for Basque pronunciation.
- Runaways already happen with the base model, and fine-tuning keeps a similar rate. A
  duration-based retry at inference is the practical fix.
- v3 drew references at random each time a row was loaded (`DYNAMIC_REF_RATIO=0.75`), and v4 froze
  the fast AR (`FREEZE_FAST_AR=true`). Neither improved SIM or robustness beyond noise, and v4's
  eval loss is slightly worse (9.19 vs 9.09). The SIM drop therefore does not come from the fast
  (acoustic) branch.
- **v4 is the released model** (demo page). Its loss is slightly higher than v2's, but it sounds and measures about the same, it has fewer runaways with a reference (3% vs 5%), and it keeps the original acoustic decoder.

## Try it: your own voice, no reference, speed

Put the sentences in a text file, one per line (`esaldiak.txt` has examples). Digits are fine:
the scripts run `eu_normalize` (`2024an 15.000` → "bi mila eta hogeita lauan hamabost mila").

```bash
source scripts/env.sh
```

### Your own voice

```bash
# record 8 s from the default microphone (the sentence to read is shown on screen)
python scripts/clone_my_voice.py --record 8 --name mikel --texts-file esaldiak.txt
# or use an existing recording (wav/mp3/m4a/mp4...) with its EXACT transcript
python scripts/clone_my_voice.py --ref my_recording.wav --ref-text "what I say in it" \
  --name mikel --texts-file esaldiak.txt
xdg-open checkpoints/my_voice/mikel/index.html
```

- The script trims and loudness-normalizes the reference into `checkpoints/my_voice/<name>/ref.wav`.
  Do not name your own file `ref.wav` in that folder, or the script will overwrite it.
  A phone recording in `.mp4` converts with
  `ffmpeg -i ref.mp4 -map 0:a:0 -ac 1 -ar 48000 -c:a pcm_s16le ref_original.wav`.
- A good reference is 5–10 s long, recorded in a quiet room with the mic 15–20 cm away, and read
  naturally **at the pace you want to hear back**. Give the exact words: the transcript is
  aligned with the audio.
- It compares `base` and v2 by default. Add v4 with
  `--models base checkpoints/audio8_eu_sft_v2/export checkpoints/audio8_eu_sft_v4/export`.
- Runaway takes (no EOS or below 8 characters/s) are regenerated with a new seed (`--tries 3`).
- `index.html` labels the players A/B in random order. Rate each one from 1 to 5 for voice
  similarity (A) and naturalness and pronunciation (N), then click *Erakutsi ereduak* to see
  which model is which, with per-model averages and SIM. *Deskargatu balorazioak* saves the
  ratings as JSON. `report.tsv` has duration, pace, SIM and number of tries for each clip.
- Everything stays in `checkpoints/`, which is git-ignored.

### What a reference can fix

A reference carries pace and pronunciation as well as the voice. Some effects are measured;
the others are hypotheses that have not been tested yet.

| Error | Can a reference help? |
|---|---|
| Runaway generations | **Yes** (measured): 3% with a reference vs 12% without (v4, 8 seeds). A continuous reading without long pauses helps. |
| Speed | **Yes** (measured): the model copies 60–75% of the reference's pace (see [Speed](#speed-cps)). |
| Accent and pronunciation | **Partly** (hypothesis): sounds it hears in the reference are easier for it to repeat. |
| A specific word (foreign names, *Antton*…) | **Sometimes** (small test, v4): with *Antton* in the reference, 1 take out of 3 came out right; without it, 0 of 3. Respelling it (*Antyon*, *Antxon*) did not help (0 of 12). |
| Acronyms and numbers (`U2`, `AP-68`) | **No**: this is a text problem. Spell them out in the input text. |
| Long or expressive sentences | **No**: this comes from missing data (9 h of neutral read speech). Split them into shorter sentences. |

A good reference:

1. is one sentence of 6–9 s, read continuously, with commas but no long pauses. This matches
   the OpenSLR read sentences the model learned to clone from;
2. contains the hard Basque sounds: sibilants (z / s / x), affricates (tz / ts / tx), r / rr,
   and j. The original model mixes these up most;
3. uses only plain words, with no digits, acronyms or foreign names. The transcript must match
   the audio exactly, so write "zazpiak eta erdietan", not "7:30";
4. is read at the pace and in the register you want in the output.

Suggested sentences:

- Main one: *Gaur goizean, zazpiak eta erdietan, txakurra hartu eta itsasertzera joan naiz, eta
  haizeak gogor jotzen zuen.* (about 8.5 s at normal pace)
- With x, ñ and an exclamation: *Kaixo, andereño! Xabierrek esan dit bihar ez dela eskolara
  etorriko, hotzeria duelako.*
- To fix a specific word, put it in the reference and generate several takes (it does not always work): *Antton eta Itziar atzo etorri ziren gurera,
  eta arratsalde osoa elkarrekin pasatu genuen.*

Why *Antton* fails: the tokenizer splits it as `Ant` + `ton`, and most *tt* words in OpenSLR 76 are English
(*twitter*, *Pitt*, *Ottawa*, where tt = t). *Antton* appears in only 5 utterances, all the same
sentence. The palatal *tt* / *dd* sounds need more Basque data to be learned properly.

Only one reference is passed per run, so it applies to every sentence. To compare references,
record each one and run `clone_my_voice.py` once per reference with a different `--name`.

### Without a reference

```bash
python scripts/say.py --texts-file esaldiak.txt --model checkpoints/audio8_eu_sft_v4/export \
  --out-dir checkpoints/my_voice/noref -n 6 --keep-candidates
```

Each sentence gets `-n` takes. Broken ones are dropped (no EOS, below 8 or above 20 characters/s),
and the one closest to the target pace is saved as `s00.wav`, `s01.wav`… Without a reference
every take has a different voice, so `--keep-candidates` also keeps all of them
(`s00_take0.wav`…) to pick from by ear.

### Speed (cps)

**cps = characters per second**: the sentence's letters (spaces not counted) divided by the
speech duration (leading and trailing silence trimmed). For "Kaixo, zer moduz zaude gaur?
Aspaldi ez zaitudala ikusi." (48 characters):

| | cps | duration |
|---|---:|---:|
| slow | 9 | ≈ 5.3 s |
| normal (OpenSLR 76 median) | 11.5 | ≈ 4.2 s |
| fast | 14 | ≈ 3.4 s |

Takes below 8 or above 20 cps are almost always broken (long silences, babbling, or cut short).

The model has no speed parameter. The pace can be steered in these ways, from most to least
natural:

1. **The reference's pace.** With a reference, the model copies roughly 60–75 % of its pace. In
   a test with a held-out speaker, stretching the reference to 0.8×, 1× and 1.25× gave output at
   10.6, 12.2 and 14.2 cps, and SIM stayed at 0.92–0.95. Record the reference at the pace you
   want, or stretch it before cloning with `clone_my_voice.py --ref-speed 0.85` (slower) or
   `1.2` (faster).
2. **`say.py --cps X`** generates `-n` takes and keeps the one closest to X. It selects a take
   and does not change the audio. This matters most without a reference: the same sentence came
   out at anywhere from 8 to 29 cps depending on the seed.
3. **`say.py --fit`** time-stretches the chosen take to exactly X cps, keeping the pitch (limited
   to 0.75–1.33×). Large changes can add small artifacts. This option has not been tested end
   to end yet.
4. **`--speed 0.9`** (both scripts) applies a fixed time-stretch to the output.
5. **Punctuation.** Commas and full stops add pauses and slow the overall pace.

Your own voice at a chosen pace:

```bash
python scripts/say.py --texts-file esaldiak.txt --cps 10 \
  --ref checkpoints/my_voice/mikel/ref.wav --ref-text "exact words of the reference" \
  --out-dir checkpoints/my_voice/mikel_slow
```

## Demo page

`docs/index.html` (EU/EN) is a GitHub Pages site: Settings → Pages → *Deploy from branch*,
`main` / `/docs`. To refresh its audio after a new run:
`python scripts/synth_checkpoints.py --checkpoint <ckpt>` and then
`python scripts/build_site.py --model <ckpt>` (currently `checkpoints/audio8_eu_sft_v4/export`).

## Licenses

- Code in this repo: Apache-2.0 (`LICENSE`).
- Audio8-TTS code and weights: Apache-2.0. antzoki-tts: Apache-2.0.
- OpenSLR 76: CC BY-SA 4.0 (Kjartansson et al., 2020). Credit it when you share models trained on it.
