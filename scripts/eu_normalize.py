#!/usr/bin/env python3
"""Normalización de texto en euskera para TTS (entrenamiento E inferencia).

Audio8 recibe el texto tal cual (BPE de Qwen, sin fonemas), así que el modelo debe
ver en entrenamiento lo mismo que verá en inferencia. Aplica SIEMPRE esta misma
función a los textos que sintetices con el modelo fine-tuneado.

Qué hace:
  * Números → palabras con el sistema vigesimal vasco, respetando el sufijo de
    declinación pegado a la cifra:  2008an → "bi mila eta zortzian",
    1950eko → "mila bederatziehun eta berrogeita hamarreko", 2001ean → "bi mila eta batean".
  * Miles con punto (1.500) y decimales con coma (3,5 → "hiru koma bost").
  * Porcentajes (%20 / 20% → "ehuneko hogei") y euros (20 € → "hogei euro").
  * Cifras pegadas a letras (TV3 → "TV hiru").
  * Unifica comillas tipográficas y espacios.
  * Excepciones manuales en OVERRIDES (marcas que se leen en otro idioma).

CLI:  python scripts/eu_normalize.py "2008an 150 urte bete zituen"
"""

from __future__ import annotations

import re
import sys
import unicodedata

UNITS = [
    "zero", "bat", "bi", "hiru", "lau", "bost", "sei", "zazpi", "zortzi", "bederatzi",
    "hamar", "hamaika", "hamabi", "hamahiru", "hamalau", "hamabost", "hamasei",
    "hamazazpi", "hemezortzi", "hemeretzi",
]
SCORES = {1: "hogei", 2: "berrogei", 3: "hirurogei", 4: "laurogei"}
HUNDREDS = {
    1: "ehun", 2: "berrehun", 3: "hirurehun", 4: "laurehun", 5: "bostehun",
    6: "seiehun", 7: "zazpiehun", 8: "zortziehun", 9: "bederatziehun",
}

# Excepciones literales (se aplican antes que nada, sin distinguir mayúsculas).
OVERRIDES: dict[str, str] = {
    "7-eleven": "seven eleven",
}

# Sufijos que se aceptan tras guion ("2008-an"). Sin guion se acepta cualquier sufijo.
HYPHEN_SUFFIXES = {
    "a", "ak", "an", "ean", "en", "ko", "eko", "ra", "era", "tik", "etik", "rekin",
    "garren", "garrena", "rena", "ek", "etan", "tan", "era", "rantz",
}


def _below_100(n: int) -> str:
    if n < 20:
        return UNITS[n]
    score, rest = divmod(n, 20)
    return SCORES[score] if rest == 0 else f"{SCORES[score]}ta {UNITS[rest]}"


def _below_1000_parts(n: int) -> list[str]:
    hundreds, rest = divmod(n, 100)
    parts = [HUNDREDS[hundreds]] if hundreds else []
    if rest:
        parts.append(_below_100(rest))
    return parts


def cardinal(n: int) -> str:
    """Cardinal vasco. 'eta' va una sola vez, antes del último componente."""
    if n < 0:
        return "minus " + cardinal(-n)
    if n == 0:
        return "zero"
    if n >= 10**12:
        return " ".join(UNITS[int(d)] for d in str(n))  # cifra a cifra
    parts: list[str] = []
    millions, rest = divmod(n, 10**6)
    if millions:
        parts.append("milioi bat" if millions == 1 else f"{cardinal(millions)} milioi")
    thousands, rest = divmod(rest, 1000)
    if thousands:
        parts.append("mila" if thousands == 1 else f"{cardinal(thousands)} mila")
    parts.extend(_below_1000_parts(rest))
    if len(parts) > 1:
        parts[-1] = "eta " + parts[-1]
    return " ".join(parts)


def _attach_suffix(words: str, suffix: str) -> str:
    """Une el sufijo escrito tras la cifra a la última palabra (hamar+eko → hamarreko)."""
    if not suffix:
        return words
    if words.endswith("r") and suffix[0] in "aeiou":
        return words + "r" + suffix
    return words + suffix


_NUMBER_RE = re.compile(
    r"(?P<int>\d{1,3}(?:\.\d{3})+|\d+)"      # entero, con miles opcionales "1.500"
    r"(?:,(?P<dec>\d+))?"                     # decimal con coma "3,5"
    r"(?P<suf>-?[a-zñ]+)?"                    # sufijo de declinación "an", "-eko"
)


def _expand_number(match: re.Match[str], text: str) -> str:
    integer = int(match.group("int").replace(".", ""))
    words = cardinal(integer)
    if match.group("dec"):
        words += " koma " + " ".join(
            cardinal(int(match.group("dec"))).split()
            if not match.group("dec").startswith("0")
            else [UNITS[int(d)] for d in match.group("dec")]
        )
    suffix = match.group("suf") or ""
    keep_hyphen_tail = ""
    if suffix.startswith("-"):
        if suffix[1:] in HYPHEN_SUFFIXES:
            suffix = suffix[1:]
        else:
            keep_hyphen_tail, suffix = suffix, ""
    out = _attach_suffix(words, suffix) + keep_hyphen_tail
    start = match.start()
    if start > 0 and text[start - 1].isalpha():  # "TV3" → "TV hiru"
        out = " " + out
    return out


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    for src, dst in OVERRIDES.items():
        text = re.sub(re.escape(src), dst, text, flags=re.IGNORECASE)
    text = (text.replace("’", "'").replace("‘", "'").replace("´", "'")
                .replace("“", '"').replace("”", '"').replace("«", '"').replace("»", '"')
                .replace("–", "-").replace("—", "-").replace(" ", " "))
    # porcentajes: "%20", "20%", "20 %" → "ehuneko 20" (en euskera va delante)
    text = re.sub(r"%\s?(\d+(?:,\d+)?)", r"ehuneko \1", text)
    text = re.sub(r"(\d+(?:,\d+)?)\s?%", r"ehuneko \1", text)
    text = re.sub(r"(\d+(?:,\d+)?)\s?€", r"\1 euro", text)
    text = _NUMBER_RE.sub(lambda m: _expand_number(m, m.string), text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


if __name__ == "__main__":
    samples = sys.argv[1:] or [
        "Heath Ledger aktore australiarra istripuz hil zen 2008an",
        "Asunción Villamil telebistan hasi zen lanean 1950eko hamarkadan",
        "Amir Khan aktoreak ekoiztetxea zabaldu zuen 2001ean",
        "Banco Bilbao Vizcaya Argentariak 150 urtetik gorako historia du",
        "Raquel Sans TV3 telebistako albistegiaren aurkezlea da",
        "7-eleven asko daude Estatu Batuetan",
        "Langabezia %12,5 jaitsi da; 1.500.000 pertsona eta 21 €",
    ]
    for s in samples:
        print(f"{s}\n  → {normalize(s)}")
