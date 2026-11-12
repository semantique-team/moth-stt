"""E-032 — candidate rare-term addendum for S1 (50 sentences, 62 term occurrences).

Third gold addendum: new **contexts**, not new terms. All 44 never-seen terms appear once, and
the six terms that E-026 found acoustically pruned (`startup`, `prototype`, `chromecast`,
`charger`, `exam`, `mondays`) appear four times each. Nothing here is recorded yet and the rows are
`reviewed: false`; the author must read them before a take.

Run:

    uv run python runs/E-032/make_terms3.py            # validate + write candidates and sheet
    uv run python runs/E-032/make_terms3.py --append   # after review: append to R-S1.jsonl
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from recordings.generate_scripts import _DedupeIndex, load_script, validate_candidate
from stt.text import normalize

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SCRIPTS = REPO / "recordings/scripts"
TERMS = (
    "airpods", "baskin", "bitbucket", "cadbury", "charger", "checkout", "chromecast",
    "delivery", "domain", "dyson", "exam", "folder", "hosting", "invision", "ipad", "kfc",
    "logout", "mondays", "oral", "popup", "prototype", "reset", "restart", "roomba",
    "scaleup", "screenshot", "selfie", "sharing", "shutdown", "shutterstock", "skills",
    "skyscanner", "smartwatch", "solving", "starbucks", "startup", "subway", "swipe",
    "tripadvisor", "twinings", "unfollow", "upload", "wallpaper", "website",
)

#: (text, category, term strings to annotate). Terms may be multi-word; spans are computed.
ROWS = [
    # six acoustically pruned terms, four new contexts each
    ("la startup assume un nuovo designer", "single_word", ["startup"]),
    ("la startup cerca un ufficio più grande", "single_word", ["startup"]),
    ("ogni startup vuole crescere in fretta", "single_word", ["startup"]),
    ("la startup ha vinto un bando regionale", "single_word", ["startup"]),
    ("abbiamo testato il prototype con tre utenti", "single_word", ["prototype"]),
    ("il prototype costa troppo per ora", "single_word", ["prototype"]),
    ("il nuovo prototype ha uno schermo più grande", "single_word", ["prototype"]),
    ("manca ancora un prototype completo", "single_word", ["prototype"]),
    ("il chromecast si è spento da solo", "brand", ["chromecast"]),
    ("presta il chromecast per la presentazione", "brand", ["chromecast"]),
    ("il chromecast supporta solo due servizi", "brand", ["chromecast"]),
    ("ho perso il telecomando del chromecast", "brand", ["chromecast"]),
    ("mi serve un charger più potente", "single_word", ["charger"]),
    ("il charger nuovo scalda troppo", "single_word", ["charger"]),
    ("porta il charger per il portatile", "single_word", ["charger"]),
    ("il charger è rimasto nella sala riunioni", "single_word", ["charger"]),
    ("un exam di programmazione è previsto a giugno", "single_word", ["exam"]),
    ("il mio exam di tedesco è andato male", "single_word", ["exam"]),
    ("un exam scritto vale più di un orale", "single_word", ["exam"]),
    ("dopo un exam ci si rilassa", "single_word", ["exam"]),
    ("i mondays sono i giorni peggiori", "single_word", ["mondays"]),
    ("odio i mondays pieni di riunioni", "single_word", ["mondays"]),
    ("i mondays iniziano sempre con una sorpresa", "single_word", ["mondays"]),
    ("i mondays mi mettono di cattivo umore", "single_word", ["mondays"]),
    # remaining terms, one new context each; twelve rows carry two terms
    ("gli airpods e il mio ipad si caricano insieme", "phrase", ["airpods", "ipad"]),
    ("baskin robbins e kfc sono entrambi affollati", "phrase", ["baskin robbins", "kfc"]),
    ("la folder del progetto è su bitbucket", "phrase", ["folder", "bitbucket"]),
    ("ho preso una cadbury da starbucks", "phrase", ["cadbury", "starbucks"]),
    ("il checkout del website è lento anche oggi", "phrase", ["checkout", "website"]),
    ("il domain e il piano di hosting scadono insieme", "phrase", ["domain", "hosting"]),
    ("il dyson e il roomba sono in carica", "phrase", ["dyson", "roomba"]),
    ("carica lo screenshot su invision", "phrase", ["screenshot", "invision"]),
    ("dopo il logout compare un popup", "phrase", ["logout", "popup"]),
    ("un reset seguito da un restart risolve tutto", "phrase", ["reset", "restart"]),
    ("facciamo uno selfie poi riprendiamo lo screen sharing", "phrase", ["selfie", "screen sharing"]),
    ("un upload del wallpaper dura dieci minuti", "phrase", ["upload", "wallpaper"]),
    ("il delivery arriva entro le sette", "single_word", ["delivery"]),
    ("un oral dura venti minuti", "single_word", ["oral"]),
    ("la scaleup ha raddoppiato i dipendenti", "single_word", ["scaleup"]),
    ("lo shutdown del server è pianificato", "single_word", ["shutdown"]),
    ("le foto arrivano da shutterstock", "brand", ["shutterstock"]),
    ("le skills richieste sono diverse", "single_word", ["skills"]),
    ("confronto i prezzi su skyscanner", "brand", ["skyscanner"]),
    ("lo smartwatch misura anche il sonno", "single_word", ["smartwatch"]),
    ("il problem solving richiede calma", "phrase", ["problem solving"]),
    ("la metro è più veloce del subway", "brand", ["subway"]),
    ("lo swipe a destra apre il menu", "single_word", ["swipe"]),
    ("tripadvisor ha cambiato la classifica", "brand", ["tripadvisor"]),
    ("il twinings è finito di nuovo", "brand", ["twinings"]),
    ("ho fatto unfollow senza pensarci", "single_word", ["unfollow"]),
]


def find_span(words: list[str], term: str) -> list[int]:
    target = term.split()
    for start in range(len(words) - len(target) + 1):
        if words[start : start + len(target)] == target:
            return [start, start + len(target)]
    raise SystemExit(f"term {term!r} not found in {words}")


def build_rows() -> list[dict]:
    existing = _DedupeIndex()
    for name in ("T.jsonl", "A.jsonl", "R-S1.jsonl", "R-S2.jsonl", "pool.jsonl"):
        for row in load_script(SCRIPTS / name):
            if name == "R-S1.jsonl" and int(row["id"].rsplit("-", 1)[1]) >= 692:
                continue
            existing.add(row["text"])

    rows = []
    covered: dict[str, int] = {term: 0 for term in TERMS}
    for offset, (text, category, terms) in enumerate(ROWS, start=692):
        if normalize(text) != text:
            raise SystemExit(f"not normalized: {text!r} -> {normalize(text)!r}")
        words = text.split()
        spans = sorted(find_span(words, term) for term in terms)
        for term in terms:
            for token in term.split():
                if token in covered:
                    covered[token] += 1
        candidate = validate_candidate(
            {
                "text": text,
                "category": category,
                "en_spans": spans,
                "generator": "opencode/deepseek-v4.1-flash",
                "date": date.today().isoformat(),
            }
        )
        if existing.is_duplicate(text):
            raise SystemExit(f"duplicate or near-duplicate of an existing script: {text!r}")
        existing.add(text)
        rows.append({"id": f"R-S1-{offset:04d}", "reviewed": False, **candidate})

    missing = [term for term, count in covered.items() if count == 0]
    if missing:
        raise SystemExit(f"terms not covered: {missing}")
    if len(rows) != 50:
        raise SystemExit(f"expected 50 rows, got {len(rows)}")
    return rows


def write_sheet(rows: list[dict]) -> None:
    lines = [
        "# E-032 — recording sheet, S1 rare-term addendum",
        "",
        f"{len(rows)} sentences, ids `{rows[0]['id']}`–`{rows[-1]['id']}`. All 44 never-seen terms",
        "appear once; `startup`, `prototype`, `chromecast`, `charger`, `exam` and `mondays`",
        "appear four times each because E-026 found them acoustically pruned.",
        "",
        "Read them once before recording. If a sentence sounds unnatural, say so: replace it",
        "before spending a take. Do not edit the text while recording.",
        "",
        "## Commands (after the rows are appended to `R-S1.jsonl`)",
        "",
        "```bash",
        "uv run python recordings/record.py --speaker S1 --kind R --start 692",
        "uv run python recordings/build_manifest.py build --kind R --speaker S1 --allow-unreviewed",
        "```",
        "",
        "`record.py` skips files that already exist; delete one wav and re-run to redo a take.",
        "",
        "| # | id | category | sentence |",
        "|---|---|---|---|",
    ]
    for index, row in enumerate(rows, start=692):
        lines.append(f"| {index} | `{row['id']}` | {row['category']} | {row['text']} |")
    (OUT / "RECORDING.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--append", action="store_true", help="append the reviewed rows to R-S1.jsonl")
    parser.add_argument(
        "--reviewed-by",
        default=None,
        help="reviewer name; required with --append and recorded in the rows",
    )
    args = parser.parse_args()

    rows = build_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    candidate = OUT / "candidates.jsonl"
    candidate.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    write_sheet(rows)

    if args.append:
        if not args.reviewed_by:
            raise SystemExit("--append requires --reviewed-by")
        target = SCRIPTS / "R-S1.jsonl"
        existing = load_script(target)
        if len(existing) != 691:
            raise SystemExit(f"R-S1.jsonl has {len(existing)} rows, expected 691")
        reviewed = [
            {
                **row,
                "reviewed": True,
                "reviewer": args.reviewed_by,
                "review_date": date.today().isoformat(),
            }
            for row in rows
        ]
        candidate.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reviewed),
            encoding="utf-8",
        )
        target.write_text(
            target.read_text(encoding="utf-8")
            + "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reviewed),
            encoding="utf-8",
        )
        print(f"appended {len(reviewed)} rows to {target.relative_to(REPO)}")
    else:
        print(f"wrote {candidate.relative_to(REPO)} and {OUT / 'RECORDING.md'} (review first)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
