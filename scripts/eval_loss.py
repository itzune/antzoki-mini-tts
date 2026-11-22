#!/usr/bin/env python3
"""Pérdida de eval (slow/fast) de varios modelos sobre el mismo eval.jsonl, sin DeepSpeed.

Usa exactamente el Dataset, el collator y compute_loss de audio8_tts_sft.py.

Uso:
  python scripts/eval_loss.py base checkpoints/audio8_eu_sft_v1/checkpoint-2000 ...
  python scripts/eval_loss.py --run checkpoints/audio8_eu_sft_v1      # base + todos los checkpoint-* + export
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "repos" / "Audio8_TTS"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help="'base', rutas a checkpoints o IDs de HF")
    ap.add_argument("--run", type=Path, help="evalúa base + todos los checkpoints del run")
    ap.add_argument("--eval-jsonl", type=Path,
                    default=ROOT / "dataset_processed" / "openslr76" / "sft" / "eval.jsonl")
    ap.add_argument("--base", default=os.environ.get("MODEL", "Edge0/Audio8-TTS-Realistic-Air"))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--out", type=Path, help="guardar resultados en JSON")
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoProcessor, TrainingArguments

    from audio8_tts_data import collate_sft_examples
    from audio8_tts_sft import Audio8TTSDataset, Audio8TTSTrainer

    models = list(args.models)
    if args.run:
        models = ["base"] + [str(p) for p in sorted(args.run.glob("checkpoint-*"),
                                                    key=lambda p: int(re.sub(r"\D", "", p.name) or 0))]
        if (args.run / "export" / "model.safetensors").is_file():
            models.append(str(args.run / "export"))
    if not models:
        ap.error("indica modelos o --run")

    processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    results = []
    for name in models:
        path = args.base if name == "base" else name
        model = AutoModel.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16).cuda()
        ds = Audio8TTSDataset(args.eval_jsonl, tokenizer=processor.tokenizer, config=model.config,
                              max_length=args.max_length, max_samples=None)
        with tempfile.TemporaryDirectory() as tmp:
            targs = TrainingArguments(output_dir=tmp, per_device_eval_batch_size=args.batch_size,
                                      bf16=True, report_to=[], remove_unused_columns=False,
                                      dataloader_num_workers=2, disable_tqdm=True)
            trainer = Audio8TTSTrainer(
                model=model, args=targs, eval_dataset=ds,
                data_collator=partial(collate_sft_examples, pad_token_id=int(model.config.pad_token_id)),
                slow_loss_weight=1.0, fast_loss_weight=1.0,
            )
            m = trainer.evaluate()
            # compute_loss acumula métricas propias; evaluate() las vuelca en el último log
            extra = trainer.state.log_history[-1] if trainer.state.log_history else {}
        row = {"model": name, "eval_loss": m["eval_loss"],
               "slow_loss": extra.get("slow_loss"), "fast_loss": extra.get("fast_loss"),
               "slow_acc": extra.get("slow_accuracy"), "fast_acc": extra.get("fast_accuracy")}
        results.append(row)
        print(f"{Path(name).name if name != 'base' else 'base':>18}  eval_loss {row['eval_loss']:.4f}  "
              f"slow {row['slow_loss'] or float('nan'):.3f}  fast {row['fast_loss'] or float('nan'):.3f}  "
              f"slow_acc {row['slow_acc'] or float('nan'):.3f}", flush=True)
        del trainer, model
        torch.cuda.empty_cache()
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
