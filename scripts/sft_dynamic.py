#!/usr/bin/env python3
"""audio8_tts_sft.py with dynamic voice-cloning references (train set only).

Each time a training row is loaded:
  * with probability DYNAMIC_REF_RATIO a reference is drawn at random from the same speaker's
    pool (another sentence, 2–10 s, different text), otherwise the row goes without a reference;
  * so every sentence is seen in both modes, with a different reference each epoch.

The eval set is untouched (fixed references), so eval loss stays comparable across runs.
Without DYNAMIC_REF_RATIO it behaves exactly like audio8_tts_sft.py.

Env:  DYNAMIC_REF_RATIO=0.75  REF_POOL=.../sft/ref_pool.json  TRAIN_JSONL=.../sft/train.jsonl
Takes the same command-line arguments as audio8_tts_sft.py (launched by scripts/train_sft.sh).
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "repos" / "Audio8_TTS"))

import audio8_tts_sft as sft  # noqa: E402
from audio8_tts_data import build_sft_example, load_codes  # noqa: E402

LOGGER = logging.getLogger("sft_dynamic")
RATIO = os.environ.get("DYNAMIC_REF_RATIO", "").strip()
POOL = os.environ.get("REF_POOL", "").strip()
TRAIN = Path(os.environ["TRAIN_JSONL"]).resolve() if os.environ.get("TRAIN_JSONL") else None

_OriginalDataset = sft.Audio8TTSDataset


def speaker_of(sample_id: str) -> str:
    return "_".join(sample_id.split("_")[:2])  # euf_07973_02093565348 -> euf_07973


class DynamicRefDataset(_OriginalDataset):
    def __init__(self, manifest: Path, **kwargs):
        super().__init__(manifest, **kwargs)
        self.ratio = float(RATIO)
        base = Path(POOL).resolve().parent
        raw = json.loads(Path(POOL).read_text(encoding="utf-8"))
        self.pool = {spk: [((base / c["codes"]).resolve(), c["text"]) for c in items]
                     for spk, items in raw.items()}
        missing = {speaker_of(r.sample_id) for r in self.rows} - set(self.pool)
        if missing:
            raise ValueError(f"speakers missing from {POOL}: {sorted(missing)[:5]}")
        LOGGER.info("Dynamic references: ratio=%.2f, pool=%d speakers / %d clips",
                    self.ratio, len(self.pool), sum(map(len, self.pool.values())))

    def _pick(self, row) -> tuple[Path, str] | None:
        cands = self.pool[speaker_of(row.sample_id)]
        for _ in range(8):
            path, text = random.choice(cands)
            if path != row.target_codes and text != row.text:
                return path, text
        return None

    def __getitem__(self, index: int):
        row = self.rows[index]
        nq = int(self.config.num_codebooks)
        target = load_codes(row.target_codes, num_codebooks=nq)
        ref = self._pick(row) if random.random() < self.ratio else None
        kwargs = dict(tokenizer=self.tokenizer, config=self.config, text=row.text,
                      target_codes=target, max_length=self.max_length)
        if ref is not None:
            try:
                return build_sft_example(**kwargs, reference_codes=load_codes(ref[0], num_codebooks=nq),
                                         reference_text=ref[1])
            except ValueError:
                pass  # too long with this reference: fall back to no reference
        return build_sft_example(**kwargs)


def dataset_factory(manifest, **kwargs):
    manifest = Path(manifest)
    if RATIO and TRAIN is not None and Path(manifest).resolve() == TRAIN:
        return DynamicRefDataset(manifest, **kwargs)
    return _OriginalDataset(manifest, **kwargs)


if __name__ == "__main__":
    if RATIO:
        if not POOL or not Path(POOL).is_file():
            sys.exit(f"DYNAMIC_REF_RATIO is set but REF_POOL does not exist: {POOL!r}")
        sft.Audio8TTSDataset = dataset_factory
    sft.main()
