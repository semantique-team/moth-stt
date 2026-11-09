# Publishing a model to `semantique-team` on Hugging Face

> Written 2026-11-09 for a colleague who is already an admin of the org. Commands verified against
> `hf` CLI **2.2.0** (`huggingface_hub`). Not moth-specific: any model follows the same steps.

## In five lines

```bash
pip install -U huggingface_hub          # provides the `hf` command
hf auth login                           # paste a write token
hf repos create semantique-team/my-model --type model
hf upload semantique-team/my-model ./local-folder --type model
```

Then open `https://huggingface.co/semantique-team/my-model` and check it.

## 1. Install the CLI

Python ≥ 3.9. Pick one:

```bash
pip install -U huggingface_hub      # inside a venv
uv tool install huggingface_hub     # with uv, no venv to manage
pipx install huggingface_hub        # with pipx
```

Verify: `hf --version`.

> On macOS do **not** use `brew install huggingface-cli`: it installs the old CLI, which still works
> but has a different syntax from the one below.

## 2. Authenticate

Create a token at <https://huggingface.co/settings/tokens> → *New token* → type **write**.

```bash
hf auth login
```

It asks for the token and stores it (system keyring where available). **Never paste a token into a
chat, a repository file or a script** — `hf auth login` is the way.

Check it and that the org is reachable:

```bash
hf auth list        # stored tokens
hf auth whoami      # who you are
```

If the org is missing, or the first upload returns **403**: your account needs *admin* or *write* on
the org, and if the org uses *resource groups* you must be in the right one (another admin confirms
from *Settings → Resource groups*).

Remove it when done: `hf auth logout`.

## 3. Create the repository

```bash
hf repos create semantique-team/my-model --type model
```

Useful: `--private` for staging, then make it public from the repo page. `hf upload` also creates
the repo if it does not exist, so this step is optional.

## 4. Prepare the folder

```
my-model/
├── README.md            ← the model card (below)
├── config.json
├── model.onnx
└── ...
```

The `README.md` **is** the model page: the YAML block at the top is what the Hub reads as metadata.
Minimum useful set:

```markdown
---
license: apache-2.0
language:
  - it
tags:
  - speech-to-text
library_name: onnxruntime
pipeline_tag: automatic-speech-recognition
---

# Model name

What it does, how to use it, with which numbers, and **what it cannot do**.
```

Worth filling in: `base_model` if it is a fine-tune, `datasets` for Hub datasets, `model-index` for
the results tab.

**State the limitations.** A card that admits its defects gets used; one that hides them is found
out on first use and costs the repo its credibility.

## 5. Upload

```bash
hf upload semantique-team/my-model ./my-model --type model
```

The folder's *contents* land at the repo **root**. Options worth knowing:

| Flag | What for |
|---|---|
| `--private` | create the repo private if it does not exist |
| `--include "*.onnx" --exclude "*.pt"` | upload only part of a folder |
| `--commit-message "..."` | commit message |
| `--every 30` | auto-commit every 30 minutes on very large folders |
| `--create-pr` | open a pull request instead of committing to `main` |

Large files: `hf upload` uses the HTTP API and handles multi-GB files **without git-lfs**. If you
clone the repo and `git push` instead, you need `git lfs` and the org's size limits apply.

## 6. Verify (the step most people skip)

```bash
hf download semantique-team/my-model --repo-type model --local-dir /tmp/check
shasum -a 256 /tmp/check/* | head
```

Then **load the model** with the library your card declares, as a user would. A successful upload is
not the same thing as a model that loads.

## 7. If you do not want to install anything

From the repo page: *Files → Add file → Upload files*, and write the card in the editor. Fine under
~50 files; beyond that it is awkward and not reproducible.

## 8. Traps

1. **Wrong token type.** A *read* token works locally and fails on upload with 403. Use **write**.
2. **Wrong folder level.** `hf upload repo ./my-model` uploads the *contents* of `my-model/`. If you
   expected `my-model/config.json` in the repo, you are one level off.
3. **Local cache.** If you re-upload a fix and still see the old file, it is the read cache:
   `hf cache`, or reload the raw URL.
4. **Inconsistent licence.** `license:` in the card and the `LICENSE` file must agree. If the model
   derives from someone else's weights, the attribution goes in the card and in `NOTICE`.
5. **`model-index` not showing.** It is ignored if the YAML is malformed or the dataset name does not
   match the Hub's.
6. **Personal data.** Do not upload audio or transcripts of real people without written consent,
   not even in a private org repo.

## 9. Bonus: a demo Space

```bash
hf repos create semantique-team/my-demo --type space --space-sdk static
hf upload semantique-team/my-demo ./static-site --type space
```

Two things learned publishing Moth's demo:

- **static** Spaces live on a different subdomain: `https://<owner>-<space>.static.hf.space` (the
  plain subdomain returns 404 with an error page — do not conclude the deploy failed);
- static Spaces do **not** send COOP/COEP headers, so there is no `SharedArrayBuffer`: a WebAssembly
  model that wants threads runs at 1 thread.

For a Python demo (Gradio/Streamlit) use `--space-sdk gradio` and upload `app.py` plus
`requirements.txt`.

## 10. This repository's own publishers

The two are version-specific and verify before uploading, so prefer them over raw `hf upload` here:

```bash
uv run python runs/E-021/upload_hf.py --repo semantique-team/moth-stt --push   # release + card + logo
uv run python runs/E-022/publish_space.py --push                              # the demo Space
```

Both read the release record (`runs/E-024/release-0.7.0.json` for 0.7.0), refuse if a hash or the
card's declared version does not match, and accept the token from `hf auth login`, `HF_TOKEN`, or
`HUGGING_FACE_HUB_TOKEN`.
