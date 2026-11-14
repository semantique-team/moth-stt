# Moth paper

LaTeX source for the preprint **Moth: A 19.7 MB Italian–English Code-Switching
Speech Recognizer Built on a Pretrained Whisper-Tiny Encoder**.

The paper is an empirical system/case-study report. It deliberately states the
evidence boundary: all code-switching quality numbers come from scripted read
speech and three known voices, and the public FLEURS results are read speech from
other speakers. Do not strengthen those claims without new recordings or a new
holdout.

## Template

The paper uses the widely used **arxiv-style** template:

- Upstream: <https://github.com/kourgeorge/arxiv-style>
- `arxiv.sty` is copied to this directory unmodified; `template/` keeps the
  upstream `template.tex`, `references.bib`, `orcid.pdf` and `License.txt`.
- Licence: MIT, Copyright (c) 2020 George Kour. See `template/License.txt`.
- The paper's `references.bib` replaces the template's example bibliography.

`main.tex` adds `hyperref` with `hidelinks`, `booktabs`, `tikz`, `longtable`
and the usual math packages. The layout, fonts and page geometry are the
template's.

## Build

The repository uses [tectonic](https://tectonic-typesetting.github.io/) because
it is available on the development machine and resolves LaTeX packages on
demand:

```bash
cd paper
tectonic -X compile main.tex
```

The output is `paper/main.pdf`. A second pass is normally unnecessary; tectonic
runs the required passes. To inspect warnings:

```bash
tectonic -X compile main.tex --print 2>&1 | grep -iE "undefined|overfull|warning"
```

## Figures

Every quantitative figure is generated from retained repository artifacts:

```bash
uv run --with matplotlib --with numpy python figures/make_figures.py
```

The script writes vector PDFs into `figures/`:

| Figure | Data source |
|---|---|
| `fig_size_wer.pdf` | sizes and WER transcribed from `docs/13-current-model-and-paper-readiness.md`; Moth size from `runs/E-031/release-0.8.1.json` |
| `fig_decoder.pdf` | `runs/E-022/RESULT.md` |
| `fig_noise.pdf` | `runs/E-024/comparison.json`; E-025 values from `runs/E-025/RESULT.md` |
| `fig_speakers.pdf` | `runs/E-038/metrics.json`, `runs/E-041/readout/browser-2/metrics.json` |
| `fig_seeds.pdf` | `runs/E-040/RESULT.md` |
| `fig_rare.pdf` | `runs/E-031/MODEL_CARD.md` |
| `fig_package.pdf` | `runs/E-031/release-0.8.1.json`, `runs/E-037/RESULT.md` |
| `fig_training.pdf` | `runs/E-00*/train.log.jsonl`, `runs/E-014`, `runs/E-024`, `runs/E-028` |

Figure 1 (the architecture) is TikZ inside `main.tex`. Values that have no
machine-readable artifact are transcribed from the named result report and are
marked as such in the script.

## Structure

1. Introduction
2. Related work
3. System (task, architecture, tokenizer, decoding, runtime)
4. Data (open corpora, recorded mixed speech, splicing/noise, splits, leakage)
5. Training (hardware, recipe, lineage, gates)
6. Evaluation protocol
7. Results (main, decoder, rare terms, noise, quantization/deployment, speakers/seeds)
8. Negative results
9. Validity and discussion
10. Conclusion
- Appendix A: release manifest
- Appendix B: full experiment index
- Appendix C: reproduction notes

## Before submitting to arXiv

1. Confirm the author name, affiliation and email in `main.tex`.
2. Choose the primary category (`cs.CL` or `cs.SD` are the natural choices) and
   the arXiv licence.
3. Re-read the validity boundary and the caveats in Sections 7.6 and 9. The
   paper intentionally does not claim spontaneous code-switching, unseen-speaker
   generalisation on the mixed task, or equivalence from a "tie".
4. If the release changes, update the release record, the package table
   (`tab:release`) and the hashes; do not edit only the prose.
5. The checkpoint behind 0.7.0 has a recorded step conflict (26,000 in the log,
   26,442 in the release label). It is disclosed in Appendix C and should stay
   disclosed unless the checkpoint hash resolves it.
