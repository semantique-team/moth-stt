#!/usr/bin/env python3
"""Generate the vector figures for the Moth paper.

All numbers are read from retained artifacts under ``runs/`` where a machine-readable
file exists; the remaining values are transcribed from the result reports named in the
comments.  Run with::

    uv run --with matplotlib --with numpy python paper/figures/make_figures.py

The script writes PDFs into ``paper/figures/``.  Nothing outside that directory is
written and no model, audio or checkpoint is touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------------------
# Shared style
# --------------------------------------------------------------------------------------
NAVY = "#1f3a5f"
VERMILION = "#c8553d"
TEAL = "#2a9d8f"
GOLD = "#d9a441"
GRAY = "#6b6b6b"
LIGHT = "#b9c6d6"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.6,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

TEXT_WIDTH = 6.5  # inches, matches arxiv.sty


def save(fig: plt.Figure, name: str) -> None:
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def load_json(rel: str):
    with open(ROOT / rel, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------------------
# Figure: accuracy versus required artifact bytes
# --------------------------------------------------------------------------------------
def fig_size_wer() -> None:
    # Package/artifact bytes and WER transcribed from docs/13-current-model-and-paper-readiness.md
    # (tables "Baseline Matrix" and "Identity and Architecture") and runs/E-031/release-0.8.1.json.
    systems = [
        # name, bytes, T-600 WER, FLEURS-IT WER, marker, colour
        ("Moth 0.8.1", 19_706_289, 24.08, 20.16, "o", VERMILION),
        ("Whistle", 16_919_407, 29.42, 11.85, "s", TEAL),
        ("Whisper-tiny", 148_738_560, 50.72, 27.42, "^", NAVY),
        ("Parakeet", 670_478_772, 10.98, 3.65, "D", GOLD),
    ]
    # Label anchors chosen by hand: the 17-20 MB cluster is too tight for automatic text.
    anchors = {
        "T": {
            "Moth 0.8.1": (30, 19.2),
            "Whistle": (13.5, 34.5),
            "Whisper-tiny": (115, 55.5),
            "Parakeet": (600, 14.0),
        },
        "F": {
            "Moth 0.8.1": (27, 17.2),
            "Whistle": (13.5, 9.0),
            "Whisper-tiny": (115, 30.0),
            "Parakeet": (600, 5.4),
        },
    }
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.8), sharey=False)
    panels = [
        (axes[0], "T", "Mixed test T-600 ($n=600$)", "WER (%)", (5, 59)),
        (axes[1], "F", "FLEURS-IT test ($n=697$)", "WER (%)", (0, 33)),
    ]
    for ax, key, title, ylabel, ylim in panels:
        for name, nbytes, t600, fleurs, marker, colour in systems:
            wer = t600 if key == "T" else fleurs
            mb = nbytes / 1e6
            ax.scatter(
                mb,
                wer,
                marker=marker,
                s=46,
                color=colour,
                zorder=3,
                edgecolor="white",
                linewidth=0.6,
            )
            tx, ty = anchors[key][name]
            ax.annotate(
                f"{name}  {wer:.1f}",
                (mb, wer),
                xytext=(tx, ty),
                textcoords="data",
                fontsize=7.2,
                color=colour,
                ha="left",
                va="center",
                arrowprops={"arrowstyle": "-", "color": colour, "linewidth": 0.5,
                                "shrinkA": 1, "shrinkB": 3},
            )
        ax.axvline(20, color=GRAY, linestyle=(0, (3, 2)), linewidth=0.8)
        ax.text(20.6, ylim[1] - 0.5, "20 MB", rotation=90, va="top", ha="left", fontsize=7,
                color=GRAY)
        ax.set_xscale("log")
        ax.set_xlim(12, 1100)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Declared artifact size (MB, log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig_size_wer.pdf")


# --------------------------------------------------------------------------------------
# Figure: decoder recalibration (E-022)
# --------------------------------------------------------------------------------------
def fig_decoder() -> None:
    # runs/E-022/RESULT.md, table "Measured vs predicted".
    rows = [
        ("CV-IT dev", 24.91, 19.90),
        ("FLEURS-EN", 55.48, 49.63),
        ("FLEURS-IT", 26.52, 20.46),
        ("T-S3", 35.43, 26.66),
        ("T-S1+S2", 31.46, 25.54),
        ("T-600", 32.77, 25.92),
    ]
    eta_rows = [
        ("T-S3", 35.21, 47.32),
        ("T-S1+S2", 46.66, 55.38),
        ("T-600", 42.86, 52.64),
    ]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.55), gridspec_kw={"wspace": 0.42})

    y = np.arange(len(rows))
    for i, (name, old, new) in enumerate(rows):
        ax.plot([old, new], [i, i], color=LIGHT, linewidth=2.2, solid_capstyle="round", zorder=1)
        ax.scatter(old, i, s=34, color=NAVY, zorder=3)
        ax.scatter(new, i, s=34, color=VERMILION, zorder=3)
        ax.annotate(f"{new - old:+.1f}", ((old + new) / 2, i), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7.2, color=VERMILION)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("WER (%)")
    ax.set_xlim(15, 60)
    ax.grid(axis="x", color="#ececec", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("(a) Word error rate", loc="left")
    ax.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color=NAVY, label="shipped decode"),
            Line2D([], [], marker="o", linestyle="", color=VERMILION, label="recalibrated"),
        ],
        loc="lower right",
    )

    y2 = np.arange(len(eta_rows))
    for i, (name, old, new) in enumerate(eta_rows):
        ax2.plot([old, new], [i, i], color=LIGHT, linewidth=2.2, solid_capstyle="round", zorder=1)
        ax2.scatter(old, i, s=34, color=NAVY, zorder=3)
        ax2.scatter(new, i, s=34, color=TEAL, zorder=3)
        ax2.annotate(f"{new - old:+.1f}", ((old + new) / 2, i), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=7.2, color=TEAL)
    ax2.set_yticks(y2, [r[0] for r in eta_rows])
    ax2.set_xlabel("ETA (%)")
    ax2.set_xlim(30, 60)
    ax2.grid(axis="x", color="#ececec", linewidth=0.6)
    ax2.set_axisbelow(True)
    ax2.set_title("(b) English-term accuracy", loc="left")
    fig.tight_layout()
    save(fig, "fig_decoder.pdf")


# --------------------------------------------------------------------------------------
# Figure: noise curve (E-024 shared pool, E-025 held-out room)
# --------------------------------------------------------------------------------------
def fig_noise() -> None:
    comp = load_json("runs/E-024/comparison.json")
    conds = ["clean", "30 dB", "20 dB", "15 dB", "10 dB", "5 dB"]
    base = [c["base_wer"] * 100 for c in comp["conditions"]]
    noisy = [c["noisy_wer"] * 100 for c in comp["conditions"]]
    ties = [c["wer_tie"] for c in comp["conditions"]]

    # runs/E-025/RESULT.md / docs/13 table "Noise Evidence".
    room_base = [22.24, 28.52, 38.71, 47.71, 63.33, 81.32]
    room_noisy = [23.94, 24.62, 31.92, 36.50, 44.99, 66.72]
    room_ties = [True, False, False, False, False, False]

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.7), sharey=True)
    x = np.arange(len(conds))
    for ax, b, n, t, title in [
        (axes[0], base, noisy, ties, "(a) Shared noise pool"),
        (axes[1], room_base, room_noisy, room_ties, "(b) Held-out room"),
    ]:
        ax.plot(x, b, marker="o", color=NAVY, linewidth=1.4, markersize=4, label="Moth 0.6.0")
        ax.plot(x, n, marker="o", color=VERMILION, linewidth=1.4, markersize=4, label="Moth 0.7.0")
        for xi, bi, ni, tie in zip(x, b, n, t):
            if not tie:
                ax.annotate(f"{ni - bi:+.1f}", (xi, min(bi, ni) - 5.5), ha="center",
                            fontsize=7, color=VERMILION)
        for xi, tie in zip(x, t):
            if tie:
                ax.text(xi, 87, "n.s.", ha="center", fontsize=6.8, color=GRAY)
        ax.set_xticks(x, conds, rotation=0)
        ax.set_xlabel("Signal-to-noise ratio")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color="#ececec", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("WER (%)")
    axes[0].set_ylim(15, 92)
    axes[0].legend(loc="lower right")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_noise.pdf")


# --------------------------------------------------------------------------------------
# Figure: per-speaker results and browser deployment cost (E-038, E-041)
# --------------------------------------------------------------------------------------
def fig_speakers() -> None:
    per = load_json("runs/E-038/metrics.json")
    br = load_json("runs/E-041/readout/browser-2/metrics.json")
    voices = ["S1", "S2", "S3", "pooled"]
    wer = np.array([per["per_speaker"][v]["wer"] if v != "pooled" else per["pooled"]["wer"]
                    for v in voices]) * 100
    wer_lo = np.array([per["per_speaker"][v]["wer_ci"][0] if v != "pooled" else per["pooled"]["wer_ci"][0]
                       for v in voices]) * 100
    wer_hi = np.array([per["per_speaker"][v]["wer_ci"][1] if v != "pooled" else per["pooled"]["wer_ci"][1]
                       for v in voices]) * 100
    eta = np.array([per["per_speaker"][v]["eta"] if v != "pooled" else per["pooled"]["eta"]
                    for v in voices]) * 100
    eta_lo = np.array([per["per_speaker"][v]["eta_ci"][0] if v != "pooled" else per["pooled"]["eta_ci"][0]
                       for v in voices]) * 100
    eta_hi = np.array([per["per_speaker"][v]["eta_ci"][1] if v != "pooled" else per["pooled"]["eta_ci"][1]
                       for v in voices]) * 100

    # E-041 browser readout, pinned post-merge run.
    b_wer = np.array([br["per_speaker"][v]["browser"]["wer"] if v != "pooled"
                      else br["browser"]["overall"]["wer"] for v in voices]) * 100
    b_eta = np.array([br["per_speaker"][v]["browser"]["eta"] if v != "pooled"
                      else br["browser"]["overall"]["eta"] for v in voices]) * 100

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.65), gridspec_kw={"wspace": 0.38})
    x = np.arange(len(voices))
    ax.bar(x, wer, width=0.55, color=VERMILION, alpha=0.85, label="Python ONNX runner")
    ax.errorbar(x, wer, yerr=[wer - wer_lo, wer_hi - wer], fmt="none", ecolor="black",
                elinewidth=0.8, capsize=3, capthick=0.8)
    ax.scatter(x, b_wer, marker="D", s=26, color=NAVY, zorder=4, label="Browser (ORT-Web)")
    for xi, v in zip(x, wer):
        ax.text(xi, 9.0, f"{v:.1f}", ha="center", fontsize=7.2, color="white", weight="bold")
    ax.set_xticks(x, voices)
    ax.set_ylabel("WER (%)")
    ax.set_ylim(0, 34)
    ax.set_title("(a) Word error rate by voice", loc="left")
    ax.grid(axis="y", color="#ececec", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")

    ax2.bar(x, eta, width=0.55, color=TEAL, alpha=0.85, label="Python ONNX runner")
    ax2.errorbar(x, eta, yerr=[eta - eta_lo, eta_hi - eta], fmt="none", ecolor="black",
                 elinewidth=0.8, capsize=3, capthick=0.8)
    ax2.scatter(x, b_eta, marker="D", s=26, color=NAVY, zorder=4, label="Browser (ORT-Web)")
    for xi, v in zip(x, eta):
        ax2.text(xi, 5.0, f"{v:.1f}", ha="center", fontsize=7.2, color="white", weight="bold")
    ax2.set_xticks(x, voices)
    ax2.set_ylabel("ETA (%)")
    ax2.set_ylim(0, 78)
    ax2.set_title("(b) English-term accuracy by voice", loc="left")
    ax2.grid(axis="y", color="#ececec", linewidth=0.6)
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig_speakers.pdf")


# --------------------------------------------------------------------------------------
# Figure: seed dispersion and rare-term target (E-040)
# --------------------------------------------------------------------------------------
def fig_seeds() -> None:
    # runs/E-040/RESULT.md, seeds 0-3.
    frozen = [24.3406, 23.8819, 23.9679, 24.0252]
    last2 = [24.2259, 24.1972, 24.1399, 24.2833]
    rare_frozen = [39, 38, 38, 38]
    rare_last2 = [41, 39, 38, 39]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.35), gridspec_kw={"wspace": 0.35})
    x = np.arange(4)
    rng = np.random.default_rng(0)
    for values, colour, label, dx in [
        (frozen, NAVY, "frozen arm", -0.07),
        (last2, TEAL, "last-2-blocks arm", 0.07),
    ]:
        ax.scatter(x + dx + rng.uniform(-0.015, 0.015, 4), values, s=28, color=colour, label=label)
        ax.hlines(np.mean(values), x.min() - 0.3, x.max() + 0.3, color=colour, linewidth=0.9,
                  linestyle=(0, (2, 2)))
    ax.axhline(24.0826, color=VERMILION, linewidth=1.0, linestyle="-")
    ax.text(3.35, 24.0826 + 0.02, "0.8.1", color=VERMILION, fontsize=7, va="bottom", ha="right")
    ax.set_xticks(x, [f"seed {i}" for i in x])
    ax.set_ylabel("T-600 WER (%)")
    ax.set_xlim(-0.4, 3.4)
    ax.set_title("(a) Same recipe, four seeds", loc="left")
    ax.grid(axis="y", color="#ececec", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left")

    ax2.axhline(42, color=GRAY, linestyle=(0, (3, 2)), linewidth=0.9)
    ax2.text(3.4, 42.2, "pre-registered target (42/81)", ha="right", fontsize=6.8, color=GRAY)
    for values, colour, label, dx in [
        (rare_frozen, NAVY, "frozen", -0.07),
        (rare_last2, TEAL, "last-2", 0.07),
    ]:
        ax2.scatter(x + dx, values, s=28, color=colour, label=label)
    ax2.axhline(38, color=VERMILION, linewidth=0.9, linestyle=(0, (2, 2)), alpha=0.8)
    ax2.text(3.4, 37.5, "0.8.1 (38/81)", color=VERMILION, fontsize=7, va="top", ha="right")
    ax2.set_xticks(x, [f"seed {i}" for i in x])
    ax2.set_ylabel("Rare aligned hits (of 81)")
    ax2.set_ylim(35, 43.5)
    ax2.set_xlim(-0.4, 3.4)
    ax2.set_title("(b) Rare-term readout", loc="left")
    ax2.grid(axis="y", color="#ececec", linewidth=0.6)
    ax2.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig_seeds.pdf")


# --------------------------------------------------------------------------------------
# Figure: rare-term progression and term-group accuracy
# --------------------------------------------------------------------------------------
def fig_rare() -> None:
    # runs/E-031/MODEL_CARD.md "Rare terms" table, shipped arms.
    releases = ["0.7.0", "0.8.0", "0.8.1"]
    aligned = [22.2, 28.4, 46.9]
    groups = {
        "natural (817)": [441 / 817 * 100, 436 / 817 * 100, 432 / 817 * 100],
        "gold (179)": [130 / 179 * 100, 126 / 179 * 100, 127 / 179 * 100],
        "rare (47)": [8 / 47 * 100, 11 / 47 * 100, 24 / 47 * 100],
    }

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.55), gridspec_kw={"wspace": 0.38})
    colours = [LIGHT, NAVY, VERMILION]
    bars = ax.bar(releases, aligned, width=0.55, color=colours, edgecolor="white", linewidth=0.6)
    for bar, v in zip(bars, aligned):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", fontsize=7.5)
    ax.set_ylabel("Aligned rare-term ETA (%)")
    ax.set_ylim(0, 55)
    ax.set_title("(a) Rare-term accuracy", loc="left")
    ax.grid(axis="y", color="#ececec", linewidth=0.6)
    ax.set_axisbelow(True)

    x = np.arange(len(groups))
    width = 0.22
    release_labels = ["0.7.0", "0.8.0", "0.8.1"]
    for i, (name, vals) in enumerate(groups.items()):
        ax2.bar(x + (i - 1) * width, vals, width=width, color=colours[i],
                label=release_labels[i], edgecolor="white", linewidth=0.4)
    ax2.set_xticks(x, list(groups.keys()))
    ax2.set_ylabel("Accuracy / presence (%)")
    ax2.set_ylim(0, 86)
    ax2.set_title("(b) Term groups on T-600", loc="left")
    ax2.grid(axis="y", color="#ececec", linewidth=0.6)
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper right", ncol=1, title="Moth release", title_fontsize=7,
               handlelength=1.0)
    fig.tight_layout()
    save(fig, "fig_rare.pdf")


# --------------------------------------------------------------------------------------
# Figure: package composition and measured INT8 cost (E-037)
# --------------------------------------------------------------------------------------
def fig_package() -> None:
    rel = load_json("runs/E-031/release-0.8.1.json")
    files = rel["files"]
    order = [
        ("model.int8.onnx", "acoustic graph (int8)", NAVY),
        ("lm_bpe3.bin", "trigram LM", TEAL),
        ("silero_vad.onnx", "Silero VAD", GOLD),
        ("mel_filters.bin", "mel filters", LIGHT),
        ("vocab.json", "vocabulary", GRAY),
        ("config.json", "config", "#999999"),
        ("NOTICE", "NOTICE", "#cccccc"),
    ]
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH, 2.75), gridspec_kw={"wspace": 0.42}
    )
    left = 0.0
    for key, label, colour in order:
        mb = files[key]["bytes"] / 1e6
        ax.barh([0], [mb], left=left, color=colour, edgecolor="white", linewidth=0.5, label=label)
        if mb > 2:
            ax.text(left + mb / 2, 0, f"{mb:.2f}", ha="center", va="center",
                    color="white", fontsize=7, weight="bold")
        left += mb
    ax.axvline(20, color=VERMILION, linestyle=(0, (3, 2)), linewidth=0.9)
    ax.text(19.9, 0.44, "20 MB ceiling", rotation=0, va="bottom", ha="right", fontsize=7,
            color=VERMILION)
    ax.set_xlim(0, 21)
    ax.set_ylim(-1.15, 0.55)
    ax.set_yticks([])
    ax.set_xlabel("Megabytes (decimal)")
    ax.set_title(f"(a) Shipped package, {rel['total_bytes'] / 1e6:.2f} MB", loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, handlelength=1.0,
              columnspacing=1.0, fontsize=6.6)

    # runs/E-037/RESULT.md, INT8 - fp32 with the A3 LM.
    labels = ["dev macro", "T-600", "T-S3", "FLEURS-IT", "FLEURS-EN"]
    delta = [0.1574, 0.4874, 0.4259, -0.1559, 0.4459]
    lo = [0.0539, -0.0290, -0.4160, -0.3950, 0.0087]
    hi = [0.2664, 0.9963, 1.2153, 0.0739, 0.8652]
    y = np.arange(len(labels))[::-1]
    sig = [(l > 0 or h < 0) for l, h in zip(lo, hi)]
    for yi, d, l, h, s in zip(y, delta, lo, hi, sig):
        colour = VERMILION if s else GRAY
        ax2.errorbar(d, yi, xerr=[[d - l], [h - d]], fmt="o", color=colour, markersize=5,
                     elinewidth=1.0, capsize=3, capthick=0.8)
    ax2.axvline(0, color="black", linewidth=0.7)
    ax2.set_yticks(y, labels)
    ax2.set_xlabel("INT8 $-$ fp32 (pp)")
    ax2.set_xlim(-0.6, 1.45)
    ax2.set_title("(b) Quantization cost, paired", loc="left")
    ax2.grid(axis="x", color="#ececec", linewidth=0.6)
    ax2.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig_package.pdf")


# --------------------------------------------------------------------------------------
# Figure: development curves from the retained training logs
# --------------------------------------------------------------------------------------
def fig_training() -> None:
    runs = [
        ("E-005", "full fine-tune", NAVY),
        ("E-006", "+2 blocks", TEAL),
        ("E-014", "+spliced IT/EN", GOLD),
        ("E-024", "+noise (0.7.0)", VERMILION),
        ("E-028", "+S2, gentle (0.8.0)", "#7b4f9d"),
    ]
    series = {}
    for run, label, colour in runs:
        steps, dev = [], []
        with open(ROOT / f"runs/{run}/train.log.jsonl", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("event") == "dev":
                    steps.append(rec["step"])
                    dev.append(rec["wer"] * 100)
        series[run] = (steps, dev, label, colour)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.55), gridspec_kw={"wspace": 0.32})
    steps, dev, label, colour = series["E-005"]
    ax.plot(steps, dev, marker="o", markersize=2.8, linewidth=1.3, color=colour)
    ax.set_xlim(0, 76000)
    ax.set_ylim(33, 59)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Dev macro WER (%)")
    ax.set_title("(a) E-005: full fine-tune", loc="left")
    ax.grid(axis="y", color="#ececec", linewidth=0.6)
    ax.set_axisbelow(True)
    # best dev is stated in the caption; an in-plot label would sit on the curve.

    for run in ["E-006", "E-014", "E-024", "E-028"]:
        steps, dev, label, colour = series[run]
        ax2.plot(steps, dev, marker="o", markersize=2.6, linewidth=1.2, color=colour, label=label)
    ax2.set_xlim(0, 39000)
    ax2.set_ylim(30, 37.4)
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("Dev macro WER (%)")
    ax2.set_title("(b) Continuation arms", loc="left")
    ax2.grid(axis="y", color="#ececec", linewidth=0.6)
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper right", ncol=1)
    fig.tight_layout()
    save(fig, "fig_training.pdf")


if __name__ == "__main__":
    fig_size_wer()
    fig_decoder()
    fig_noise()
    fig_speakers()
    fig_seeds()
    fig_rare()
    fig_package()
    fig_training()
