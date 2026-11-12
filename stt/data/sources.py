"""Dataset registry, download and manifest planning (design: ``docs/03-data.md`` §2).

Only open licenses (CC0 / CC-BY / own recordings). Evaluation sources are marked ``eval_only``
and ``plan_records`` refuses them for training — leakage is prevented by construction, not by
memory. The registry records where each source comes from and what its license is, so that
``NOTICE`` and the data card can be written from code rather than recollection.

Downloads are checksum-verified and append to ``<raw>/SHA256SUMS``; existing files with a
matching digest are not downloaded again (resumable).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf

from stt.text import normalize

__all__ = [
    "SOURCES",
    "Source",
    "download",
    "plan_records",
    "probe_duration",
    "read_cv_tsv",
    "read_fleurs_tsv",
    "read_librispeech_transcripts",
    "read_mls_transcripts",
    "sample_by_duration",
    "sha256_file",
    "write_records",
]


@dataclass(frozen=True)
class Source:
    """One dataset: identity, license, splits and the reader used to build manifests."""

    name: str
    license: str
    url: str
    splits: tuple[str, ...]
    lang: str
    reader: str
    eval_only: bool = False
    notes: str = ""
    citation: str = ""
    extra: dict = field(default_factory=dict)


SOURCES: dict[str, Source] = {
    "cv_it_27": Source(
        name="Common Voice IT 27.0",
        license="CC0",
        url="https://commonvoice.mozilla.org/en/datasets",
        splits=("train", "dev", "test"),
        lang="it",
        reader="cv_tsv",
        notes="Many speakers; short read prompts; official splits only; drop digit transcripts.",
        citation="Mozilla Common Voice 27.0, Italian (CC0)",
    ),
    "mls_it": Source(
        name="Multilingual LibriSpeech Italian",
        license="CC-BY 4.0",
        url="https://www.openslr.org/94/",
        splits=("train", "dev", "test"),
        lang="it",
        reader="mls",
        notes="Audiobook read speech, 10-20 s clips, lowercase transcripts.",
        citation="MLS Italian, OpenSLR 94 (CC-BY 4.0)",
    ),
    "librispeech": Source(
        name="LibriSpeech train-clean-100",
        license="CC-BY 4.0",
        url="https://www.openslr.org/12",
        splits=("train", "dev", "test"),
        lang="en",
        reader="librispeech",
        notes="English read speech; or use a Common Voice EN slice (CC0) if an Italian-accented "
        "slice exists — decision recorded at M2.",
        citation="LibriSpeech, OpenSLR 12 (CC-BY 4.0)",
    ),
    "fleurs_it": Source(
        name="FLEURS Italian",
        license="CC-BY 4.0",
        url="https://huggingface.co/datasets/google/fleurs",
        splits=("dev", "test"),
        lang="it",
        reader="fleurs",
        eval_only=True,
        notes="Evaluation only — never train on it, not even the FLEURS train split.",
        citation="FLEURS (CC-BY 4.0)",
    ),
    "fleurs_en": Source(
        name="FLEURS English",
        license="CC-BY 4.0",
        url="https://huggingface.co/datasets/google/fleurs",
        splits=("dev", "test"),
        lang="en",
        reader="fleurs",
        eval_only=True,
        notes="Evaluation only.",
        citation="FLEURS (CC-BY 4.0)",
    ),
    "noise_musan": Source(
        name="MUSAN (music, speech and noise)",
        license="CC-BY-4.0",
        url="https://www.openslr.org/17/",
        splits=("noise", "music"),
        lang="it",
        reader="noise",
        notes="Only the noise and music subsets are used; own room tone and digital silence are ours.",
        citation="MUSAN (CC-BY 4.0), OpenSLR 17",
    ),
}


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path | str, sha256: str | None = None) -> Path:
    """Download ``url`` to ``dest`` unless a file with the expected digest already exists.

    Verifies ``sha256`` when given and appends the digest to ``<dest.parent>/SHA256SUMS``.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if sha256 is None or sha256_file(dest) == sha256:
            return dest
        dest.unlink()

    if url.startswith("file://"):
        shutil.copyfile(urllib.request.url2pathname(url[7:]), dest)
    else:
        with urllib.request.urlopen(url) as response, dest.open("wb") as handle:
            shutil.copyfileobj(response, handle)

    if sha256 is not None:
        actual = sha256_file(dest)
        if actual != sha256:
            dest.unlink()
            raise ValueError(f"checksum mismatch for {url}: expected {sha256}, got {actual}")

    sums = dest.parent / "SHA256SUMS"
    with sums.open("a", encoding="utf-8") as handle:
        handle.write(f"{sha256_file(dest)}  {dest.name}\n")
    return dest


def probe_duration(path: Path | str) -> float:
    """Duration in seconds of an audio file (soundfile; mp3 via libsndfile ≥ 1.1)."""
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


# ---------------------------------------------------------------------------------------------
# readers: each turns a source tree into manifest rows with ``audio``, ``text``, ``speaker``


def read_cv_tsv(path: Path | str, clips_dir: str = "clips") -> list[dict]:
    """Common Voice ``*.tsv``: id, path, sentence; rows without a sentence are skipped."""
    rows: list[dict] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    index = {name: position for position, name in enumerate(header)}
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) < len(header):
            continue
        sentence = fields[index["sentence"]].strip()
        if not sentence:
            continue
        clip = fields[index["path"]].strip()
        rows.append(
            {
                # One speaker records many clips: the clip name is unique, the client id is not.
                "id": f"cv-{Path(clip).stem}",
                "audio": f"{clips_dir}/{clip}",
                "text": sentence,
                "speaker": fields[index["client_id"]],
            }
        )
    return rows


def read_librispeech_transcripts(root: Path | str) -> list[dict]:
    """LibriSpeech ``*.trans.txt`` files under ``root`` (e.g. ``LibriSpeech/train-clean-100``)."""
    root = Path(root)
    rows: list[dict] = []
    for trans in sorted(root.rglob("*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            uid, _, text = line.partition(" ")
            speaker = uid.split("-")[0]
            audio = trans.parent.relative_to(root) / f"{uid}.flac"
            rows.append(
                {
                    "id": f"ls-{uid}",
                    "audio": audio.as_posix(),
                    "text": text.strip(),
                    "speaker": speaker,
                }
            )
    return rows


def read_mls_transcripts(root: Path | str) -> list[dict]:
    """MLS ``transcripts.txt`` (``<uid>\\t<text>``) with audio under ``audio/<speaker>/<book>/``."""
    root = Path(root)
    rows: list[dict] = []
    for line in (root / "transcripts.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        uid, _, text = line.partition("\t")
        speaker = uid.split("_")[0]
        book = uid.split("_")[1] if "_" in uid else "0"
        rows.append(
            {
                "id": f"mls-{uid}",
                "audio": f"audio/{speaker}/{book}/{uid}.flac",
                "text": text.strip(),
                "speaker": speaker,
            }
        )
    return rows


def read_fleurs_tsv(root: Path | str, split: str) -> list[dict]:
    """FLEURS ``<split>.tsv``: **headerless** official layout ``id, filename, transcription,
    raw_transcription, chars, num_samples, gender``; a headered variant is also accepted.

    The ``id`` column is *not* unique across rows (it identifies the sentence, not the
    recording), so the id used here is the filename stem, which is unique.
    """
    root = Path(root)
    lines = (root / f"{split}.tsv").read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].strip():
        return []
    first = lines[0].split("\t")
    if "transcription" in first:
        index = {name: position for position, name in enumerate(first)}
        data_lines = lines[1:]
        id_pos, file_pos, text_pos = index["id"], index["filename"], index["transcription"]
    else:
        data_lines = lines
        id_pos, file_pos, text_pos = 0, 1, 2
    rows: list[dict] = []
    for line in data_lines:
        fields = line.split("\t")
        if len(fields) <= max(id_pos, file_pos, text_pos):
            continue
        text = fields[text_pos].strip()
        if not text:
            continue
        filename = fields[file_pos].strip()
        rows.append(
            {
                "id": f"fleurs-{Path(filename).stem}",
                "audio": f"audio/{split}/{filename}",
                "text": text,
                "speaker": None,
            }
        )
    return rows


_READERS = {
    "cv_tsv": read_cv_tsv,
    "librispeech": read_librispeech_transcripts,
    "mls": read_mls_transcripts,
    "fleurs": read_fleurs_tsv,
}


def plan_records(
    source_key: str, root: Path | str, split: str, source: str | None = None
) -> list[dict]:
    """Turn one downloaded source split into manifest rows (without durations yet)."""
    if source_key not in SOURCES:
        raise ValueError(f"unknown source {source_key!r}; known: {sorted(SOURCES)}")
    meta = SOURCES[source_key]
    if meta.eval_only and split == "train":
        raise ValueError(f"{source_key} is evaluation-only: training on it is forbidden")
    if split not in meta.splits:
        raise ValueError(f"{source_key} has no split {split!r} (known: {meta.splits})")
    root = Path(root)
    reader = _READERS.get(meta.reader)
    if reader is None:
        raise ValueError(f"source {source_key} has no transcript reader ({meta.reader})")
    if meta.reader == "fleurs":
        rows = reader(root, split)
    elif meta.reader == "cv_tsv":
        rows = reader(root / f"{split}.tsv")
    else:
        rows = reader(root)
    planned = []
    for row in rows:
        text = row["text"]
        planned.append(
            {
                "id": row["id"],
                "audio": row["audio"],
                "text": text,
                "norm": normalize(text, meta.lang),
                "duration": probe_duration(root / row["audio"]),
                "source": source or source_key,
                "split": split,
                "speaker": row.get("speaker"),
                "lang": meta.lang,
                "category": None,
                "en_spans": [],
            }
        )
    return planned


def sample_by_duration(
    records: list[dict], target_seconds: float, seed: int = 0
) -> dict:
    """Deterministic duration-capped sample (``docs/03-data.md`` §2 proposes ≈ 170 h each)."""
    import random

    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    selected: list[dict] = []
    total = 0.0
    for position in order:
        row = records[position]
        if total >= target_seconds:
            break
        selected.append(row)
        total += float(row.get("duration") or 0.0)
    return {
        "selected": selected,
        "selected_seconds": total,
        "total_seconds": sum(float(row.get("duration") or 0.0) for row in records),
        "short": total < target_seconds,
    }


def write_records(rows: list[dict], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
