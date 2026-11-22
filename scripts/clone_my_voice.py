#!/usr/bin/env python3
"""Clone your own voice and compare models side by side (blind listening page).

  # 1) record yourself (reads a sentence on screen, 8 s from the default microphone)
  python scripts/clone_my_voice.py --record 8 --name mikel

  # 2) or use an existing recording + its exact transcript
  python scripts/clone_my_voice.py --ref nire_ahotsa.wav --ref-text "Gaur goizean ..." --name mikel

  # options
  --text "Esaldi bat." --text "Beste bat."   own sentences (default: 8 test sentences)
  --models base checkpoints/audio8_eu_sft_v2/export   models to compare (default)
  --speed 0.9        slow down / speed up the OUTPUT (time-stretch, pitch kept; 0.5–2.0)
  --ref-speed 0.85   stretch the REFERENCE before cloning: the model copies its pace
  --tries 3          re-generate (new seed) when a take runs away (no EOS or < 8 chars/s)

Output: checkpoints/my_voice/<name>/  ref.wav, <model>/*.wav, report.tsv, index.html
Open index.html in a browser: players are labelled A/B in random order, rate them 1–5,
then "Erakutsi ereduak" reveals which model is which.

Reference tips: 5–10 s, quiet room, phone/headset mic 15–20 cm away, read naturally at the
pace you want to hear back, and give the EXACT transcript (it is aligned with the audio).
"""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eu_normalize import normalize  # noqa: E402

SR = 44100
MIN_CPS = 8.0
RECORD_PROMPT = ("Gaur goizean kalera atera naiz, eta lagun zahar batekin topo egin dut; "
                 "kafe bat hartu dugu elkarrekin.")
SENTENCES = [
    "Kaixo, zer moduz zaude gaur? Aspaldi ez zaitudala ikusi.",
    "Bihar goizean, zortzietan, Bilboko geltokian elkartuko gara.",
    "Euskara Europako hizkuntzarik zaharrenetakoa da, eta oraindik bizirik dirau.",
    "Ez dut uste hori egia denik; zergatik esan didazu halakorik?",
    "Donostiako zinemaldia 1953an sortu zen, eta urtero milaka ikusle erakartzen ditu.",
    "Mendira joan ginen, baina euria hasi zuen eta etxera itzuli behar izan genuen.",
    "Liburutegiak ehun eta berrogei liburu berri jaso ditu aste honetan.",
    "Ene bada! Hau bai sorpresa polita!",
]


# ── audio helpers ────────────────────────────────────────────────────────────
def record(path: Path, seconds: float) -> None:
    print(f"\nIrakurri ozen, lasai eta argi ({seconds:.0f} s):\n\n    «{RECORD_PROMPT}»\n")
    input("Sakatu Enter prest zaudenean... ")
    if shutil.which("pw-record") or shutil.which("parecord"):
        src = ["-f", "pulse", "-i", "default"]
    else:
        src = ["-f", "alsa", "-i", "default"]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *src, "-t", str(seconds),
                    "-ac", "1", "-ar", "48000", str(path)], check=True)
    print("Grabatuta.")


def tempo(src: Path, dst: Path, factor: float) -> None:
    """Time-stretch keeping pitch (ffmpeg atempo, 0.5–2.0)."""
    if not 0.5 <= factor <= 2.0:
        raise ValueError("speed must be within 0.5–2.0")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                    "-filter:a", f"atempo={factor}", str(dst)], check=True)


def prepare_reference(src: Path, dst: Path, ref_speed: float) -> float:
    import librosa
    import numpy as np
    import pyloudnorm as pyln
    import soundfile as sf

    y, _ = librosa.load(src, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=35, frame_length=1024, hop_length=256)
    y = np.concatenate([np.zeros(int(0.1 * SR), y.dtype), y, np.zeros(int(0.15 * SR), y.dtype)])
    loud = pyln.Meter(SR).integrated_loudness(y)
    if np.isfinite(loud):
        y = pyln.normalize.loudness(y, loud, -20.0)
    peak = float(np.abs(y).max() or 1.0)
    if peak > 0.95:
        y = y * (0.95 / peak)
    sf.write(dst, y.astype("float32"), SR)
    if ref_speed != 1.0:
        tmp = dst.with_name("ref_raw_tempo.wav")
        tempo(dst, tmp, ref_speed)
        tmp.replace(dst)
    dur = sf.info(dst).duration
    if dur < 3:
        print(f"[!] erreferentzia laburregia ({dur:.1f} s): 5–10 s gomendatzen da")
    elif dur > 12:
        print(f"[!] erreferentzia luzeegia ({dur:.1f} s): entrenamenduan 2–10 s erabili dira")
    return dur


def speaking_rate(path: Path, text: str) -> tuple[float, float]:
    import librosa
    import soundfile as sf

    y, sr = sf.read(path, dtype="float32")
    _, (s, e) = librosa.effects.trim(y, top_db=40)
    dur = max((e - s) / sr, 1e-3)
    return len(text.replace(" ", "")) / dur, dur


# ── synthesis ────────────────────────────────────────────────────────────────
def load_model(spec: str, base_id: str):
    import torch
    from transformers import AutoModel, AutoProcessor
    from synth_checkpoints import base_snapshot, complete_checkpoint

    path = base_id if spec == "base" else str(Path(spec).resolve())
    if spec != "base":
        complete_checkpoint(Path(path), base_snapshot(base_id))
    proc = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    model = AutoModel.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16).eval().to("cuda")
    return proc, model


def synth_batch(proc, model, texts, ref: Path | None, ref_text: str | None, seed: int,
                temperature: float = 0.8):
    import torch

    kw = {}
    if ref is not None:
        kw = dict(reference_audio=[str(ref)] * len(texts), reference_text=[ref_text] * len(texts))
    inputs = proc(text=texts, return_tensors="pt", **kw)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    gen = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1024, temperature=temperature, top_p=0.95, top_k=50,
                             do_sample=True, generator=gen, return_dict_in_generate=True)
        wavs, lens = model.decode_audio(out.codes)
    return [(bool(out.finished[i]), wavs[i, : int(lens[i])].float().cpu().numpy()) for i in range(len(texts))]


def label(spec: str) -> str:
    if spec == "base":
        return "base"
    p = Path(spec).resolve()
    return p.parent.name.replace("audio8_eu_sft_", "") if p.name == "export" else f"{p.parent.name}_{p.name}"


# ── listening page ───────────────────────────────────────────────────────────
PAGE = """<!doctype html><html lang="eu"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Nire ahotsa</title>
<style>
:root{--bg:#fafaf7;--fg:#1d1d1b;--mut:#6b6b66;--line:#e3e2dc;--card:#fff;--acc:#2f6f4f}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9a9a93;--line:#2c2c2a;--card:#1f1f1d;--acc:#7cc4a0}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:860px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.5rem;margin:0 0 4px}p.m{color:var(--mut);margin:0 0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
.t{font-weight:600;margin-bottom:8px}.row{display:flex;flex-wrap:wrap;align-items:center;gap:8px 14px;margin:6px 0}
.lab{min-width:2.2em;font-weight:700}.who{color:var(--acc);font-size:.9rem;display:none}
body.rev .who{display:inline}audio{height:36px;max-width:100%}
.st button{border:1px solid var(--line);background:transparent;color:var(--fg);border-radius:6px;width:30px;height:30px;cursor:pointer}
.st button.on{background:var(--acc);color:var(--bg);border-color:var(--acc)}
.bar{position:sticky;top:0;background:var(--bg);padding:10px 0;border-bottom:1px solid var(--line);display:flex;gap:10px;flex-wrap:wrap;z-index:1}
.bar button{padding:7px 12px;border-radius:6px;border:1px solid var(--acc);background:var(--acc);color:var(--bg);cursor:pointer}
#sum{margin-top:16px}table{border-collapse:collapse}td,th{padding:4px 12px;border-bottom:1px solid var(--line);text-align:left}
.bad{color:#c0392b;font-size:.85rem}
</style></head><body><main>
<h1>Nire ahotsaren klonazioa</h1>
<p class="m">Erreferentzia: «__REFTEXT__» (__REFDUR__ s). Entzun, eman 1–5 puntu ahotsaren antzari (A) eta naturaltasunari/ahoskerari (N), eta gero erakutsi ereduak.</p>
<div class="card"><div class="t">Zure grabazioa (erreferentzia)</div><audio controls preload="none" src="ref.wav"></audio></div>
<div class="bar"><button id="rev">Erakutsi ereduak</button><button id="dl">Deskargatu balorazioak</button></div>
__ROWS__
<div id="sum"></div>
</main><script>
const KEY="clone-ratings-__NAME__";let R={};try{R=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
function save(){try{localStorage.setItem(KEY,JSON.stringify(R))}catch(e){}}
document.querySelectorAll(".st").forEach(st=>{const k=st.dataset.k;for(let i=1;i<=5;i++){const b=document.createElement("button");b.textContent=i;
 if(R[k]==i)b.classList.add("on");b.onclick=()=>{R[k]=i;save();st.querySelectorAll("button").forEach(x=>x.classList.toggle("on",x===b));summary()};st.appendChild(b)}});
function summary(){const s={};for(const[k,v]of Object.entries(R)){const[clip,dim]=k.split("|");const m=document.querySelector(`[data-clip="${clip}"]`);if(!m)continue;
 const mod=m.dataset.model;s[mod]=s[mod]||{A:[],N:[]};s[mod][dim].push(v)}
 const avg=a=>a.length?(a.reduce((x,y)=>x+y,0)/a.length).toFixed(2)+" ("+a.length+")":"–";
 if(!document.body.classList.contains("rev")){document.getElementById("sum").innerHTML="";return}
 let h="<table><tr><th>Eredua</th><th>Antza</th><th>Naturaltasuna</th><th>SIM (WavLM)</th></tr>";
 for(const[m,v]of Object.entries(s))h+=`<tr><td>${m}</td><td>${avg(v.A)}</td><td>${avg(v.N)}</td><td>${(__SIMS__)[m]??"–"}</td></tr>`;
 document.getElementById("sum").innerHTML=h+"</table>"}
document.getElementById("rev").onclick=()=>{document.body.classList.toggle("rev");summary()};
document.getElementById("dl").onclick=()=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(R,null,1)],{type:"application/json"}));a.download="balorazioak___NAME__.json";a.click()};
</script></body></html>"""


def write_page(out: Path, name: str, ref_text: str, ref_dur: float, rows: list[dict], sims: dict) -> None:
    by_text: dict[str, list[dict]] = {}
    for r in rows:
        by_text.setdefault(r["id"], []).append(r)
    rng = random.Random(name)
    cards = []
    for sid, clips in by_text.items():
        clips = clips[:]
        rng.shuffle(clips)
        lines = []
        for letter, c in zip("ABCDEFGH", clips):
            key = f"{c['model']}/{sid}"
            warn = '<span class="bad">kontroletik kanpo</span>' if not c["ok"] else ""
            lines.append(
                f'<div class="row" data-clip="{html.escape(key)}" data-model="{html.escape(c["model"])}">'
                f'<span class="lab">{letter}</span><audio controls preload="none" src="{html.escape(c["file"])}"></audio>'
                f'<span>A</span><span class="st" data-k="{html.escape(key)}|A"></span>'
                f'<span>N</span><span class="st" data-k="{html.escape(key)}|N"></span>'
                f'<span class="who">{html.escape(c["model"])} · SIM {c["sim"]:.3f}</span>{warn}</div>')
        cards.append(f'<div class="card"><div class="t">{html.escape(clips[0]["text"])}</div>{"".join(lines)}</div>')
    page = (PAGE.replace("__ROWS__", "\n".join(cards)).replace("__REFTEXT__", html.escape(ref_text))
            .replace("__REFDUR__", f"{ref_dur:.1f}").replace("__NAME__", html.escape(name))
            .replace("__SIMS__", json.dumps(sims)))
    (out / "index.html").write_text(page, encoding="utf-8")


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ref", type=Path, help="your recording (wav/mp3/m4a/ogg...)")
    src.add_argument("--record", type=float, metavar="SECONDS", help="record from the microphone")
    ap.add_argument("--ref-text", help="exact transcript of --ref")
    ap.add_argument("--name", default="nire_ahotsa")
    ap.add_argument("--text", action="append", help="sentence to synthesize (repeatable)")
    ap.add_argument("--texts-file", type=Path, help="one sentence per line")
    ap.add_argument("--models", nargs="+", default=["base", "checkpoints/audio8_eu_sft_v2/export"])
    ap.add_argument("--speed", type=float, default=1.0, help="output time-stretch (0.5–2.0)")
    ap.add_argument("--ref-speed", type=float, default=1.0, help="reference time-stretch (0.5–2.0)")
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "checkpoints" / "my_voice")
    ap.add_argument("--base", default=os.environ.get("MODEL", "Edge0/Audio8-TTS-Realistic-Air"))
    args = ap.parse_args()

    out = args.out_dir / args.name
    out.mkdir(parents=True, exist_ok=True)
    if args.record:
        raw = out / "ref_recorded.wav"
        record(raw, args.record)
        ref_text = RECORD_PROMPT
    else:
        if not args.ref_text:
            ap.error("--ref needs --ref-text (the exact words you say in the recording)")
        raw, ref_text = args.ref, args.ref_text
    ref_text = normalize(ref_text)
    ref = out / "ref.wav"
    ref_dur = prepare_reference(raw, ref, args.ref_speed)
    ref_cps, _ = speaking_rate(ref, ref_text)
    print(f"[ref] {ref_dur:.1f} s, {ref_cps:.1f} karaktere/s (corpusaren mediana ≈ 11.4)  «{ref_text}»")

    texts = list(args.text or [])
    if args.texts_file:
        texts += [l.strip() for l in args.texts_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    texts = texts or SENTENCES
    items = [(f"s{i:02d}", t, normalize(t)) for i, t in enumerate(texts)]

    import soundfile as sf
    import torch
    from eval_robustness import SpeakerEmbedder

    rows: list[dict] = []
    for spec in args.models:
        name = label(spec)
        mdir = out / name
        mdir.mkdir(exist_ok=True)
        print(f"[{name}] kargatzen…", flush=True)
        proc, model = load_model(spec, args.base)
        todo = list(items)
        best: dict[str, tuple] = {}
        for t in range(args.tries):
            if not todo:
                break
            res = synth_batch(proc, model, [n for _, _, n in todo], ref, ref_text, args.seed + 1000 * t)
            still = []
            for (sid, orig, norm), (finished, wav) in zip(todo, res):
                path = mdir / f"{sid}.wav"
                sf.write(path, wav, SR)
                cps, dur = speaking_rate(path, norm)
                ok = finished and cps >= MIN_CPS
                prev = best.get(sid)
                if prev is None or ok or (not prev[2] and cps > prev[3]):
                    best[sid] = (orig, wav, ok, cps, dur, t + 1)
                if not ok:
                    still.append((sid, orig, norm))
            todo = still
        for sid, (orig, wav, ok, cps, dur, tries) in best.items():
            path = mdir / f"{sid}.wav"
            sf.write(path, wav, SR)
            if args.speed != 1.0:
                tmp = mdir / f"{sid}_tmp.wav"
                tempo(path, tmp, args.speed)
                tmp.replace(path)
            rows.append({"model": name, "id": sid, "text": orig, "file": f"{name}/{sid}.wav",
                         "ok": ok, "cps": round(cps, 1), "dur": round(dur, 2), "tries": tries})
        del model, proc
        torch.cuda.empty_cache()

    emb = SpeakerEmbedder("cuda")
    for r in rows:
        r["sim"] = emb.sim(out / r["file"], ref)
    sims = {}
    for m in dict.fromkeys(r["model"] for r in rows):
        v = [r["sim"] for r in rows if r["model"] == m and r["ok"]]
        sims[m] = round(sum(v) / len(v), 3) if v else None

    with (out / "report.tsv").open("w", encoding="utf-8") as fh:
        fh.write("model\tid\tok\ttries\tdur_s\tchars_per_s\tsim\ttext\n")
        for r in rows:
            fh.write(f"{r['model']}\t{r['id']}\t{int(r['ok'])}\t{r['tries']}\t{r['dur']}\t{r['cps']}\t{r['sim']:.3f}\t{r['text']}\n")
    write_page(out, args.name, ref_text, ref_dur, rows, sims)

    print()
    for m, s in sims.items():
        mr = [r for r in rows if r["model"] == m]
        bad = sum(not r["ok"] for r in mr)
        cps = sum(r["cps"] for r in mr) / len(mr)
        print(f"  {m:10s} SIM {s}  | abiadura {cps:.1f} kar/s | kontroletik kanpo {bad}/{len(mr)}")
    print(f"\nEntzuteko:  xdg-open {out / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
