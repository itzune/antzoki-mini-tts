#!/usr/bin/env python3
"""Paso 1 — OpenSLR76 (euskera) → WAVs limpios a 44.1 kHz + manifiesto crudo para Audio8.

Por cada frase:
  1. lee el WAV (48 kHz), pasa a mono,
  2. recorta silencios inicial/final (librosa.effects.trim) dejando un margen,
  3. remuestrea a 44.1 kHz (frecuencia del códec de Audio8) con soxr,
  4. normaliza sonoridad a -20 LUFS con techo de pico a -1 dBFS,
  5. normaliza el texto con scripts/eu_normalize.py,
  6. filtra por duración, clipping y velocidad de habla anómala (texto ≠ audio).

Salida en dataset_processed/openslr76/:
  wavs/<id>.wav               audio procesado (PCM 16 bit, mono, 44.1 kHz)
  metadata.jsonl              TODAS las frases con métricas y estado (keep / motivo de descarte)
  raw.jsonl                   sólo las válidas → entrada de audio8_tts_prepare.py
  normalization_review.tsv    frases cuyo texto cambió al normalizar (revísalas)

Uso:  python scripts/prepare_openslr76.py [--workers 8] [--min-dur 1.0 --max-dur 20.0]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eu_normalize import normalize  # noqa: E402

TARGET_SR = 44100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=ROOT / "dataset_raw" / "openslr76")
    p.add_argument("--out-dir", type=Path, default=ROOT / "dataset_processed" / "openslr76")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--min-dur", type=float, default=1.0, help="segundos tras recortar")
    p.add_argument("--max-dur", type=float, default=20.0, help="segundos tras recortar")
    p.add_argument("--top-db", type=float, default=40.0, help="umbral de silencio para trim")
    p.add_argument("--pad", type=float, default=0.15, help="margen de silencio a conservar (s)")
    p.add_argument("--lufs", type=float, default=-20.0)
    p.add_argument("--peak-db", type=float, default=-1.0)
    p.add_argument("--max-clip-frac", type=float, default=1e-3)
    p.add_argument("--cps-mad", type=float, default=4.0,
                   help="descarta si chars/s se aleja más de N MAD de la mediana del hablante")
    p.add_argument("--limit", type=int, default=None, help="procesar sólo N frases (pruebas)")
    return p.parse_args()


def read_index(raw_dir: Path) -> list[dict]:
    rows = []
    for tsv in sorted(raw_dir.glob("line_index_*.tsv")):
        with tsv.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                file_id, text = line.rstrip("\n").split("\t", 1)
                file_id = file_id.strip()
                speaker = "_".join(file_id.split("_")[:2])  # euf_01208
                rows.append({
                    "id": file_id,
                    "speaker": speaker,
                    "gender": "f" if speaker.startswith("euf") else "m",
                    "text_raw": text.strip(),
                })
    if not rows:
        raise FileNotFoundError(f"no hay line_index_*.tsv en {raw_dir}")
    return rows


def process_one(job: dict) -> dict:
    # import local: cada worker carga lo suyo y con 1 hilo (evita sobre-suscripción)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import numpy as np
    import librosa
    import pyloudnorm as pyln
    import soundfile as sf

    out = {"id": job["id"]}
    try:
        y, sr = sf.read(job["src"], dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        out["sr_raw"] = int(sr)
        out["dur_raw"] = round(len(y) / sr, 3)
        clip_frac = float(np.mean(np.abs(y) >= 0.999)) if len(y) else 0.0
        out["clip_frac"] = round(clip_frac, 6)

        _, (start, end) = librosa.effects.trim(y, top_db=job["top_db"], frame_length=2048, hop_length=256)
        pad = int(job["pad"] * sr)
        y = y[max(0, start - pad): min(len(y), end + pad)]
        if sr != TARGET_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR, res_type="soxr_hq")
        dur = len(y) / TARGET_SR
        out["dur"] = round(dur, 3)
        if dur < 0.5:
            out["status"] = "drop_too_short"
            return out

        meter = pyln.Meter(TARGET_SR)
        loud = meter.integrated_loudness(y)
        out["lufs_in"] = round(float(loud), 2) if np.isfinite(loud) else None
        if np.isfinite(loud):
            y = y * (10 ** ((job["lufs"] - loud) / 20))
        peak = float(np.max(np.abs(y))) or 1.0
        ceiling = 10 ** (job["peak_db"] / 20)
        if peak > ceiling:
            y = y * (ceiling / peak)

        Path(job["dst"]).parent.mkdir(parents=True, exist_ok=True)
        sf.write(job["dst"], y.astype(np.float32), TARGET_SR, subtype="PCM_16")
        out["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        out["status"] = "drop_error"
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def main() -> None:
    args = parse_args()
    rows = read_index(args.raw_dir)
    wav_index = {p.stem: p for p in args.raw_dir.rglob("*.wav")}
    if not wav_index:
        sys.exit(f"No hay WAVs bajo {args.raw_dir}. ¿Has descomprimido eu_es_female.zip / eu_es_male.zip?")
    if args.limit:
        rows = rows[: args.limit]

    wav_dir = args.out_dir / "wavs"
    jobs, results = [], {}
    for r in rows:
        r["text"] = normalize(r["text_raw"])
        src = wav_index.get(r["id"])
        if src is None:
            results[r["id"]] = {"id": r["id"], "status": "drop_missing_audio"}
            continue
        jobs.append({
            "id": r["id"], "src": str(src), "dst": str(wav_dir / f"{r['id']}.wav"),
            "top_db": args.top_db, "pad": args.pad, "lufs": args.lufs, "peak_db": args.peak_db,
        })

    from tqdm import tqdm

    print(f"[prepare] {len(rows)} frases, {len(jobs)} con audio, {args.workers} procesos")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_one, j) for j in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="audio"):
            res = fut.result()
            results[res["id"]] = res

    # ── filtros que necesitan estadísticas globales / por hablante ──────────
    merged = [{**r, **results.get(r["id"], {"status": "drop_missing_audio"})} for r in rows]
    for m in merged:
        if m["status"] != "ok":
            continue
        if not args.min_dur <= m["dur"] <= args.max_dur:
            m["status"] = "drop_duration"
        elif m["clip_frac"] > args.max_clip_frac:
            m["status"] = "drop_clipping"
        elif not m["text"]:
            m["status"] = "drop_empty_text"
        m["cps"] = round(len(m["text"].replace(" ", "")) / m["dur"], 2) if m.get("dur") else None

    by_spk: dict[str, list[float]] = {}
    for m in merged:
        if m["status"] == "ok":
            by_spk.setdefault(m["speaker"], []).append(m["cps"])
    for m in merged:
        if m["status"] != "ok":
            continue
        vals = by_spk[m["speaker"]]
        med = statistics.median(vals)
        mad = statistics.median(abs(v - med) for v in vals) or 1e-6
        m["cps_z"] = round((m["cps"] - med) / (1.4826 * mad), 2)
        if abs(m["cps_z"]) > args.cps_mad:
            m["status"] = "drop_speaking_rate"

    # ── escritura ────────────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "metadata.jsonl").open("w", encoding="utf-8") as fh:
        for m in merged:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    kept = [m for m in merged if m["status"] == "ok"]
    with (args.out_dir / "raw.jsonl").open("w", encoding="utf-8") as fh:
        for m in kept:
            fh.write(json.dumps({"id": m["id"], "text": m["text"], "audio": f"wavs/{m['id']}.wav"},
                                ensure_ascii=False) + "\n")
    with (args.out_dir / "normalization_review.tsv").open("w", encoding="utf-8") as fh:
        fh.write("id\ttext_raw\ttext\n")
        for m in merged:
            if m["text"] != m["text_raw"]:
                fh.write(f"{m['id']}\t{m['text_raw']}\t{m['text']}\n")
    # borra WAVs de frases descartadas para no confundir
    for m in merged:
        if m["status"] != "ok":
            (wav_dir / f"{m['id']}.wav").unlink(missing_ok=True)

    # ── resumen ──────────────────────────────────────────────────────────────
    counts: dict[str, int] = {}
    for m in merged:
        counts[m["status"]] = counts.get(m["status"], 0) + 1
    hours = sum(m["dur"] for m in kept) / 3600
    spk = {m["speaker"] for m in kept}
    print("\n[prepare] estados:", json.dumps(counts, ensure_ascii=False))
    print(f"[prepare] válidas: {len(kept)} frases | {hours:.2f} h | {len(spk)} hablantes "
          f"({sum(s.startswith('euf') for s in spk)} F / {sum(s.startswith('eum') for s in spk)} M)")
    if kept:
        durs = sorted(m["dur"] for m in kept)
        print(f"[prepare] duración: min {durs[0]:.1f}s | mediana {durs[len(durs)//2]:.1f}s | max {durs[-1]:.1f}s")
    print(f"[prepare] salida: {args.out_dir}")


if __name__ == "__main__":
    main()
