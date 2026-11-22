#!/usr/bin/env python3
"""Paso 3 — Manifiestos SFT (train / eval / prompts de escucha) a partir de los códigos.

Requiere haber ejecutado antes:
  1. scripts/prepare_openslr76.py   → metadata.jsonl, raw.jsonl, wavs/
  2. audio8_tts_prepare.py          → prepared/all.jsonl + prepared/codes/<id>.target.npy

Qué hace:
  * Reserva hablantes completos (1 F + 1 M por defecto, como antzoki-tts) para eval:
    miden generalización a voces no vistas y sirven de referencia en las pruebas de escucha.
  * Empareja cada frase, con probabilidad --ref-ratio, con OTRA frase del mismo hablante
    (texto distinto, 2–10 s) como audio de referencia → entrena la clonación de voz
    zero-shot igual que en inferencia. Reutiliza los .npy ya codificados (no recodifica).
  * Valida cada fila con build_sft_example (el mismo código que el entrenador) para que
    ninguna supere --max-length a mitad de entrenamiento.
  * Escribe val_prompts.jsonl con frases NUEVAS (fuera del corpus) para escuchar checkpoints.

Salida en dataset_processed/openslr76/sft/:
  train.jsonl  eval.jsonl  val_prompts.jsonl  stats.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "repos" / "Audio8_TTS"))
sys.path.insert(0, str(ROOT / "scripts"))

from eu_normalize import normalize  # noqa: E402

# Frases de escucha que NO están en OpenSLR76 (variedad: pregunta, exclamación,
# números con sufijo, frase larga, nombres propios, registro coloquial).
LISTEN_SENTENCES = [
    "Kaixo, zer moduz zaude gaur? Aspaldi ez zaitudala ikusi.",
    "Bihar goizean, zortzietan, Bilboko geltokian elkartuko gara.",
    "Euskara Europako hizkuntzarik zaharrenetakoa da, eta oraindik bizirik dirau.",
    "Ez dut uste hori egia denik; zergatik esan didazu halakorik?",
    "Donostiako zinemaldia 1953an sortu zen, eta urtero milaka ikusle erakartzen ditu.",
    "Mendira joan ginen, baina euria hasi zuen eta etxera itzuli behar izan genuen.",
    "Liburutegiak ehun eta berrogei liburu berri jaso ditu aste honetan.",
    "Ene bada! Hau bai sorpresa polita!",
]


def parse_args() -> argparse.Namespace:
    d = ROOT / "dataset_processed" / "openslr76"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", type=Path, default=d / "metadata.jsonl")
    p.add_argument("--prepared", type=Path, default=d / "prepared" / "all.jsonl")
    p.add_argument("--out-dir", type=Path, default=d / "sft")
    p.add_argument("--model", default=os.environ.get("MODEL", "Edge0/Audio8-TTS-Realistic-Air"))
    p.add_argument("--holdout-speakers", nargs="*", default=None,
                   help="IDs de hablante para eval (defecto: 1 F + 1 M elegidos con --seed)")
    p.add_argument("--ref-ratio", type=float, default=0.6, help="fracción de filas con referencia")
    p.add_argument("--ref-min", type=float, default=2.0)
    p.add_argument("--ref-max", type=float, default=10.0)
    p.add_argument("--max-length", type=int, default=2048, help="igual que MAX_LENGTH del SFT")
    p.add_argument("--eval-max", type=int, default=200, help="máx. filas de eval (velocidad)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def rel(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    import torch
    from transformers import AutoConfig, AutoProcessor

    from audio8_tts_data import build_sft_example, load_codes

    meta = {}
    with args.metadata.open(encoding="utf-8") as fh:
        for line in fh:
            m = json.loads(line)
            meta[m["id"]] = m
    rows = []
    with args.prepared.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            m = meta[r["id"]]
            rows.append({
                "id": r["id"], "text": r["text"], "speaker": m["speaker"], "gender": m["gender"],
                "dur": m["dur"], "codes": (args.prepared.parent / r["target_codes"]).resolve(),
                "wav": (args.metadata.parent / "wavs" / f"{r['id']}.wav").resolve(),
            })
    if not rows:
        sys.exit("prepared/all.jsonl está vacío")

    by_spk: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_spk[r["speaker"]].append(r)

    # ── hablantes reservados ────────────────────────────────────────────────
    if args.holdout_speakers:
        holdout = list(args.holdout_speakers)
        unknown = [s for s in holdout if s not in by_spk]
        if unknown:
            sys.exit(f"hablantes desconocidos: {unknown}")
    else:
        holdout = []
        for prefix in ("euf", "eum"):
            cands = sorted(s for s in by_spk if s.startswith(prefix) and len(by_spk[s]) >= 50)
            holdout.append(rng.choice(cands))
    holdout_set = set(holdout)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoProcessor.from_pretrained(args.model, trust_remote_code=True).tokenizer
    nq = int(config.num_codebooks)
    code_cache: dict[Path, torch.Tensor] = {}

    def codes(path: Path) -> torch.Tensor:
        if path not in code_cache:
            code_cache[path] = load_codes(path, num_codebooks=nq)
        return code_cache[path]

    def fits(target: dict, ref: dict | None) -> int | None:
        try:
            ex = build_sft_example(
                tokenizer=tokenizer, config=config, text=target["text"],
                target_codes=codes(target["codes"]),
                reference_codes=codes(ref["codes"]) if ref else None,
                reference_text=ref["text"] if ref else None,
                max_length=args.max_length,
            )
            return int(ex["input_ids"].shape[1])
        except ValueError:
            return None

    def build(split_rows: list[dict], out_path: Path) -> dict:
        base = out_path.parent
        stats = {"rows": 0, "with_ref": 0, "dropped_too_long": 0, "tokens_max": 0, "hours": 0.0}
        with out_path.open("w", encoding="utf-8") as fh:
            for t in split_rows:
                ref = None
                if rng.random() < args.ref_ratio:
                    pool = [c for c in by_spk[t["speaker"]]
                            if c["id"] != t["id"] and c["text"] != t["text"]
                            and args.ref_min <= c["dur"] <= args.ref_max]
                    rng.shuffle(pool)
                    for cand in pool[:5]:
                        if fits(t, cand) is not None:
                            ref = cand
                            break
                n_tok = fits(t, ref)
                if n_tok is None:
                    stats["dropped_too_long"] += 1
                    continue
                row = {"id": t["id"], "text": t["text"], "target_codes": rel(t["codes"], base)}
                if ref:
                    row.update(reference_codes=rel(ref["codes"], base), reference_text=ref["text"],
                               reference_audio=rel(ref["wav"], base))
                    stats["with_ref"] += 1
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["rows"] += 1
                stats["tokens_max"] = max(stats["tokens_max"], n_tok)
                stats["hours"] += t["dur"] / 3600
        stats["hours"] = round(stats["hours"], 2)
        return stats

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = [r for r in rows if r["speaker"] not in holdout_set]
    eval_rows = [r for r in rows if r["speaker"] in holdout_set]
    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)
    eval_rows = eval_rows[: args.eval_max]

    print(f"[splits] hablantes reservados para eval: {holdout}")
    print(f"[splits] validando {len(train_rows)} filas de train con build_sft_example…")
    st_train = build(train_rows, args.out_dir / "train.jsonl")
    st_eval = build(eval_rows, args.out_dir / "eval.jsonl")

    # ── prompts de escucha: referencia de cada hablante reservado + sin referencia ──
    base = args.out_dir
    prompts = []
    for spk in holdout:
        cands = sorted((r for r in by_spk[spk] if 4.0 <= r["dur"] <= 8.0), key=lambda r: r["id"])
        ref = rng.choice(cands or by_spk[spk])
        for i, s in enumerate(LISTEN_SENTENCES):
            prompts.append({"id": f"{spk}_s{i:02d}", "text": normalize(s),
                            "reference_audio": rel(ref["wav"], base), "reference_text": ref["text"]})
    for i, s in enumerate(LISTEN_SENTENCES[:3]):
        prompts.append({"id": f"noref_s{i:02d}", "text": normalize(s)})
    with (args.out_dir / "val_prompts.jsonl").open("w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in prompts)

    # ── reference pool for dynamic sampling (scripts/sft_dynamic.py) ──
    # Uses no RNG: train/eval stay identical for the same seed.
    pool: dict[str, list[dict]] = defaultdict(list)
    for spk, items in sorted(by_spk.items()):
        for c in sorted(items, key=lambda r: r["id"]):
            if args.ref_min <= c["dur"] <= args.ref_max:
                pool[spk].append({"codes": rel(c["codes"], base), "text": c["text"]})
    (args.out_dir / "ref_pool.json").write_text(json.dumps(pool, ensure_ascii=False))

    train_texts = {r["text"] for r in train_rows}
    overlap = sum(r["text"] in train_texts for r in eval_rows)
    stats = {
        "model": args.model, "seed": args.seed, "holdout_speakers": holdout,
        "ref_ratio_target": args.ref_ratio, "max_length": args.max_length,
        "train": st_train, "eval": st_eval,
        "train_speakers": len({r["speaker"] for r in train_rows}),
        "eval_texts_also_in_train": overlap,
        "val_prompts": len(prompts),
    }
    (args.out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"[splits] salida: {args.out_dir}")


if __name__ == "__main__":
    main()
