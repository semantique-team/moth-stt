"""E-015 addendum — generate the term-rich gold sentences for S1 and S2.

44 terms that appear in no training text (measured on 0.7.0, 17% accuracy). Fifty sentences per
speaker, distinct wording, every term once per speaker plus six two-term sentences. The script
computes ``en_spans`` from the terms themselves and refuses anything that is not canonical.

    uv run python runs/E-015/make_term_script.py --check
    uv run python runs/E-015/make_term_script.py --write
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recordings.generate_scripts import validate_candidate
from stt.text import normalize

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/E-015"

#: (text, terms, category). ``terms`` are exact phrases to annotate; every term must appear.
S1: list[tuple[str, list[str], str]] = [
    ("gli airpods si sono scaricati durante la call", ["airpods"], "brand"),
    ("abbiamo preso un gelato da baskin robbins", ["baskin robbins"], "brand"),
    ("il codice del progetto è su bitbucket", ["bitbucket"], "brand"),
    ("la cioccolata cadbury è finita", ["cadbury"], "brand"),
    ("ho lasciato il charger in ufficio", ["charger"], "single_word"),
    ("il checkout non accetta la mia carta", ["checkout"], "single_word"),
    ("il chromecast non trova la rete", ["chromecast"], "brand"),
    ("il delivery è arrivato in ritardo", ["delivery"], "single_word"),
    ("il domain del sito scade a dicembre", ["domain"], "single_word"),
    ("l'aspirapolvere dyson è potentissimo", ["dyson"], "brand"),
    ("il mio exam di inglese è andato bene", ["exam"], "single_word"),
    ("salva i file nella folder condivisa", ["folder"], "single_word"),
    ("il piano di hosting scade oggi", ["hosting"], "single_word"),
    ("il design è su invision", ["invision"], "brand"),
    ("il mio ipad non si carica", ["ipad"], "brand"),
    ("il kfc vicino alla stazione è pieno", ["kfc"], "brand"),
    ("fai il logout prima di chiudere", ["logout"], "single_word"),
    ("i mondays in ufficio sono pesanti", ["mondays"], "single_word"),
    ("domani ho un oral exam di inglese", ["oral exam"], "phrase"),
    ("chiudi quel popup e riprova", ["popup"], "single_word"),
    ("il prototype è pronto per la demo", ["prototype"], "single_word"),
    ("devo fare un reset della password", ["reset"], "single_word"),
    ("fai un restart del server", ["restart"], "single_word"),
    ("il roomba è bloccato sotto il divano", ["roomba"], "brand"),
    ("la scaleup ha assunto tre sviluppatori", ["scaleup"], "single_word"),
    ("mandami uno screenshot dell'errore", ["screenshot"], "single_word"),
    ("ci siamo fatti un selfie in ufficio", ["selfie"], "single_word"),
    ("lo screen sharing non funziona", ["screen sharing"], "phrase"),
    ("lo shutdown è previsto per le sei", ["shutdown"], "single_word"),
    ("le foto sono prese da shutterstock", ["shutterstock"], "brand"),
    ("ho aggiunto le skills nel profilo", ["skills"], "single_word"),
    ("ho cercato il volo su skyscanner", ["skyscanner"], "brand"),
    ("lo smartwatch mi conta i passi", ["smartwatch"], "single_word"),
    ("facciamo una sessione di problem solving", ["problem solving"], "phrase"),
    ("ho preso un caffè da starbucks", ["starbucks"], "brand"),
    ("la startup ha chiuso un round", ["startup"], "single_word"),
    ("ho mangiato un panino da subway", ["subway"], "brand"),
    ("fai swipe a destra per accettare", ["swipe"], "single_word"),
    ("ho letto le recensioni su tripadvisor", ["tripadvisor"], "brand"),
    ("il tè twinings è nella credenza", ["twinings"], "brand"),
    ("ho fatto unfollow a quel profilo", ["unfollow"], "single_word"),
    ("il file è in upload da stamattina", ["upload"], "single_word"),
    ("ho cambiato il wallpaper del desktop", ["wallpaper"], "single_word"),
    ("il website aziendale è lento", ["website"], "single_word"),
    ("il checkout del website è lento", ["checkout", "website"], "phrase"),
    ("prima un reset poi un restart del router", ["reset", "restart"], "phrase"),
    ("metti lo screenshot nella folder del progetto", ["screenshot", "folder"], "phrase"),
    ("il file è in upload e lo screen sharing non parte", ["upload", "screen sharing"], "phrase"),
    ("abbiamo fatto un selfie da starbucks", ["selfie", "starbucks"], "phrase"),
    ("il mio ipad è scarico e il charger è a casa", ["ipad", "charger"], "phrase"),
]

S2: list[tuple[str, list[str], str]] = [
    ("gli airpods nuovi hanno una buona durata", ["airpods"], "brand"),
    ("il gelato di baskin robbins è troppo dolce", ["baskin robbins"], "brand"),
    ("abbiamo spostato i repository su bitbucket", ["bitbucket"], "brand"),
    ("ho comprato una tavoletta cadbury", ["cadbury"], "brand"),
    ("il charger del telefono è nel cassetto", ["charger"], "single_word"),
    ("il checkout online richiede la verifica", ["checkout"], "single_word"),
    ("ho collegato il chromecast alla televisione", ["chromecast"], "brand"),
    ("il delivery costa cinque euro", ["delivery"], "single_word"),
    ("abbiamo comprato un domain nuovo", ["domain"], "single_word"),
    ("il dyson fa più rumore del previsto", ["dyson"], "brand"),
    ("il mio exam di domani è stato spostato", ["exam"], "single_word"),
    ("crea una folder per ogni cliente", ["folder"], "single_word"),
    ("abbiamo cambiato hosting per risparmiare", ["hosting"], "single_word"),
    ("il prototipo è caricato su invision", ["invision"], "brand"),
    ("il mio ipad serve per prendere appunti", ["ipad"], "brand"),
    ("abbiamo ordinato il pollo da kfc", ["kfc"], "brand"),
    ("dopo il logout la sessione scade", ["logout"], "single_word"),
    ("i mondays dopo le ferie sono difficili", ["mondays"], "single_word"),
    ("un oral exam è fissato per lunedì", ["oral exam"], "phrase"),
    ("il popup della privacy blocca il sito", ["popup"], "single_word"),
    ("il prototype ha superato il test", ["prototype"], "single_word"),
    ("un reset del modem risolve spesso", ["reset"], "single_word"),
    ("serve un restart dell'applicazione", ["restart"], "single_word"),
    ("il roomba pulisce il soggiorno ogni mattina", ["roomba"], "brand"),
    ("la scaleup cerca nuovi investitori", ["scaleup"], "single_word"),
    ("allega uno screenshot alla segnalazione", ["screenshot"], "single_word"),
    ("lo selfie con il team è venuto bene", ["selfie"], "single_word"),
    ("lo screen sharing consuma molta banda", ["screen sharing"], "phrase"),
    ("lo shutdown del sistema è automatico", ["shutdown"], "single_word"),
    ("abbiamo comprato le immagini su shutterstock", ["shutterstock"], "brand"),
    ("le skills tecniche si imparano sul campo", ["skills"], "single_word"),
    ("il volo più economico era su skyscanner", ["skyscanner"], "brand"),
    ("lo smartwatch ha la batteria scarica", ["smartwatch"], "single_word"),
    ("il problem solving è la competenza più richiesta", ["problem solving"], "phrase"),
    ("il caffè di starbucks costa troppo", ["starbucks"], "brand"),
    ("la startup ha aperto una sede a milano", ["startup"], "single_word"),
    ("il subway della stazione è chiuso", ["subway"], "brand"),
    ("lo swipe verso sinistra cancella il messaggio", ["swipe"], "single_word"),
    ("le recensioni su tripadvisor erano ottime", ["tripadvisor"], "brand"),
    ("ho comprato il tè twinings al supermercato", ["twinings"], "brand"),
    ("ho fatto unfollow dopo quel post", ["unfollow"], "single_word"),
    ("il servizio di upload richiede tempo", ["upload"], "single_word"),
    ("il wallpaper nuovo è troppo scuro", ["wallpaper"], "single_word"),
    ("il website non si apre da mobile", ["website"], "single_word"),
    ("il checkout sul domain nuovo non funziona", ["checkout", "domain"], "phrase"),
    ("un reset e un restart risolvono il problema", ["reset", "restart"], "phrase"),
    ("lo screenshot è nella folder dei bug", ["screenshot", "folder"], "phrase"),
    (
        "il servizio di upload è lento e lo screen sharing si blocca",
        ["upload", "screen sharing"],
        "phrase",
    ),
    ("un selfie da starbucks è quasi obbligatorio", ["selfie", "starbucks"], "phrase"),
    ("il mio ipad e il charger sono nello zaino", ["ipad", "charger"], "phrase"),
]


def spans_for(text: str, terms: list[str]) -> list[list[int]]:
    words = text.split()
    spans: list[list[int]] = []
    for term in terms:
        term_words = term.split()
        found = [
            index
            for index in range(len(words) - len(term_words) + 1)
            if words[index : index + len(term_words)] == term_words
        ]
        if len(found) != 1:
            raise SystemExit(f"{text!r}: term {term!r} found {len(found)} times")
        start = found[0]
        end = start + len(term_words)
        if any(start < other[1] and other[0] < end for other in spans):
            raise SystemExit(f"{text!r}: overlapping spans for {term!r}")
        spans.append([start, end])
    return spans


def rows(pairs: list[tuple[str, list[str], str]]) -> list[dict]:
    out = []
    for text, terms, category in pairs:
        if normalize(text) != text:
            raise SystemExit(f"not canonical: {normalize(text)!r} != {text!r}")
        out.append(validate_candidate(
            {"text": text, "category": category, "en_spans": spans_for(text, terms),
             "generator": "opencode/deepseek-v4.1-flash", "date": "2026-11-09"}
        ))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(prog="make_term_script.py")
    parser.add_argument("--check", action="store_true", help="validate only (the default)")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    for speaker, pairs in (("S1", S1), ("S2", S2)):
        data = rows(pairs)
        print(f"{speaker}: {len(data)} frasi, {len({t for _, ts, _ in pairs for t in ts})} termini")
        if args.write:
            path = OUT / f"candidates-{speaker}.jsonl"
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data),
                encoding="utf-8",
            )
            print(f"  scritto {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
