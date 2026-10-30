<div align="center">

<img src="docs/assets/icon.png" alt="Veto icon" width="128" height="128">

# Veto

**A privacy-first classifier for unsolicited political text messages. Train it and run it entirely on your own device.**

[![CI](https://github.com/Carmeisel101/veto/actions/workflows/ci.yml/badge.svg)](https://github.com/Carmeisel101/veto/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![Python 3.10–3.13](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
![Status: early development](https://img.shields.io/badge/status-early%20development-yellow)
![Data: never committed](https://img.shields.io/badge/data-never%20committed-lightgrey)

[Quick start](#quick-start) ·
[How it works](#how-it-works) ·
[Core ML export](#export-to-core-ml) ·
[Docs](#documentation) ·
[Contributing](#contributing) ·
[Contact](#contact)

</div>

---

Veto detects **unsolicited political outreach in SMS**: campaign messages, donation requests,
get-out-the-vote texts and similar. This repository is the **training and export pipeline**: it
builds a corpus from your own messages, labels it, trains a small MiniLM-based classifier,
evaluates it, and exports it to [Core ML](https://developer.apple.com/documentation/coreml).

The iOS runtime (Swift tokenizer, Core ML inference and a Message Filter Extension) lives in the
separate [`veto-ios`](https://github.com/Carmeisel101/veto-ios) repository, which consumes this
project's exported bundle.

> **Status: early development.** Veto v1 is a first baseline, not a production filter. The
> current private dataset has 117 positive examples (24 each in validation and test), so metrics
> are rough. See [Current limitations](#current-limitations).

## Contents

- [Features](#features)
- [How it works](#how-it-works)
- [What counts as political solicitation](#what-counts-as-political-solicitation)
- [Privacy](#privacy)
- [Setup](#setup)
- [Quick start](#quick-start)
- [Try it without your own data](#try-it-without-your-own-data)
- [Export to Core ML](#export-to-core-ml)
- [Models are not distributed](#models-are-not-distributed-train-your-own)
- [Current limitations](#current-limitations)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Contact](#contact)
- [License](#license)

## Features

- **Private by design.** Data, models and artifacts are gitignored, and CI rejects pull requests
  that add them.
- **Local end to end.** Corpus processing, training and inference make no network calls.
- **Leak-aware splits.** Duplicate and near-duplicate message families are never split across
  train, validation and test.
- **Two approaches, compared.** A frozen-embedding logistic regression baseline against a
  fine-tuned `BertForSequenceClassification` head. The better one is selected and saved.
- **Core ML export.** One FP16 ML Program (about 46 MB with tokenizer assets), verified against
  Python inference.
- **Runnable without your data.** A synthetic example dataset ships in the repo.

## How it works

```text
Messages → Corpus → Labels → Model development → Evaluation
        → Core ML export → iOS Message Filter Extension (veto-ios)
```

| Stage | Command | Output |
|---|---|---|
| Extract (macOS) | `uv run python scripts/extract_raw.py` | `data/raw/data.parquet` |
| Build corpus | `uv run veto corpus build` | `data/corpus/corpus.parquet` |
| Inspect | `uv run veto corpus inspect` | `artifacts/dataset_report.*` |
| Train | `uv run veto train` | `models/veto/` |
| Evaluate | `uv run veto evaluate` | `artifacts/evaluation.json` |
| Predict | `uv run veto predict "..."` | label, probability, `should_filter` |
| Export | `uv run veto export coreml` | `dist/coreml/` |

**Base model:** `sentence-transformers/all-MiniLM-L6-v2`. Two approaches are trained and
compared: frozen embeddings with logistic regression (baseline) and a fine-tuned classifier head.
The selected model, its decision threshold and metadata are saved to `models/veto/`. URLs, phone
numbers and other numbers are masked for duplicate grouping only.

## What counts as political solicitation

Veto answers one question: *is this unsolicited political outreach the user would want
filtered?*

- **Positive:** campaign messages, fundraising, advocacy, petitions, GOTV and election-information
  outreach, polls used as outreach, and event or volunteer asks.
- **Negative:** a friend discussing the election. Veto detects political *outreach*, not
  political *language*.

Subtypes (`campaign`, `fundraising`, ...) are recorded for analysis, but the model predicts the
binary target (`political_solicitation` / `not_political_solicitation`). Ambiguous or
contentless messages are marked `review_required` and excluded from training.

## Privacy

Message datasets are private and **never committed**. `data/` is gitignored; the repository
contains only code to reproduce the pipeline. Corpus processing is local and makes no network
calls. Tests use synthetic messages only.

Do not commit or share your `data/`, `models/`, `artifacts/` or `dist/` folders. CI enforces this:
pull requests that add files under those folders (or notebooks with saved outputs) are rejected.

## Setup

Veto uses [uv](https://docs.astral.sh/uv/). The project pins Python 3.12 (`.python-version`)
because PyTorch does not yet publish wheels for the newest Python releases.

```bash
git clone https://github.com/Carmeisel101/veto && cd veto
uv run pytest            # creates the environment on first run, then runs the tests
```

> **Keep the project outside iCloud Drive** (e.g. `~/Code/veto`, not under
> `~/Library/Mobile Documents/...`). iCloud marks files in the virtualenv as hidden, which breaks
> `import veto`, and syncing `.git` and your private message data is risky. If you must stay in
> iCloud, see [docs/raw_extraction.md](docs/raw_extraction.md#running-inside-icloud-drive).

## Quick start

```bash
uv run python scripts/extract_raw.py   # chat.db -> data/raw/data.parquet (needs Full Disk Access)
uv run veto corpus build               # raw -> data/corpus/corpus.parquet
uv run veto corpus inspect             # label join, duplicate groups, splits, dataset report
uv run veto train                      # baseline + fine-tune, compare, save best to models/veto/
uv run veto evaluate                   # print saved metrics (artifacts/evaluation.json)
uv run veto predict "Election Day is tomorrow. Vote YES on Proposition 12."
```

Extraction is only needed to refresh the raw data; every other command reads `data/raw/`.
Training takes roughly 20 minutes on an M-series Mac (mostly cross-validation).

Labels live in the private `data/annotations/solicitation_labels.csv` (columns
`message_id,label,subtype,review_required,review_reason,label_source`). Python use:

```python
from veto.inference import VetoClassifier

VetoClassifier.load("models/veto").predict("...")
# {'label': ..., 'probability': ..., 'threshold': ..., 'should_filter': ...}
```

## Try it without your own data

A synthetic example dataset (1,000 invented messages, 10% political solicitations) ships in
`examples/synthetic/`, so you can run the whole pipeline immediately:

```bash
uv run veto demo        # about 2 minutes; outputs go to demo/ (gitignored)
uv run veto evaluate --artifacts demo/artifacts
```

See [docs/synthetic.md](docs/synthetic.md). The demo model illustrates the pipeline only; it is
not a useful filter.

## Export to Core ML

The selected model can be exported as a single, self-describing Core ML artifact for the iOS app:

```bash
uv sync --extra coreml            # once (macOS)
uv run veto export coreml         # -> dist/coreml/{veto.mlpackage, metadata.json, tokenizer/}
```

| | |
|---|---|
| Format | Core ML ML Program, FP16, iOS 17+ |
| Inputs | `input_ids`, `attention_mask` (int32, `[1, 256]`) |
| Output | `political_solicitation_probability` |
| Threshold | in `metadata.json`, applied by the app (not in the graph) |
| Size | about 46 MB including tokenizer assets |
| Validation | zero decision disagreements on the committed test strings |

The export is verified against Python inference. See [docs/coreml_export.md](docs/coreml_export.md)
for the tokenizer contract, model interface, parity results and limitations. `dist/` is gitignored
because it contains trained weights.

## Models are not distributed: train your own

Trained model files are **not** published with this repository, and `models/` is gitignored.
Veto's training data is a person's private text messages, and a fine-tuned model can memorize
parts of its training data (names, phone numbers, message text). Publishing weights would risk
leaking that data, so each user trains their own model **on their own device, from their own
messages**.

1. Extract your messages (macOS only, needs Full Disk Access):
   `uv run python scripts/extract_raw.py`
2. Build the corpus: `uv run veto corpus build`
3. Label it. Write `data/annotations/solicitation_labels.csv` following the definitions above
   and the schema in [docs/corpus.md](docs/corpus.md). Labels are specific to you and are not
   shipped either.
4. Train: `uv run veto train`. The model, threshold and metadata are saved to `models/veto/`.
5. Use it: `uv run veto predict "..."` or `VetoClassifier.load("models/veto")`.

Everything runs locally; no message or model is uploaded anywhere.

## Current limitations

- 117 positives among about 9,000 labeled messages; validation and test hold 24 positives each,
  so a single miss moves recall by about 4 points. Treat metrics as rough.
- Positives come from a handful of campaigns (a few candidates dominate), so the model may be
  learning those names rather than political outreach in general. Issue advocacy, events,
  petitions and nonpartisan election information are under-represented and are the main misses.
- The baseline-vs-fine-tune selection rule (pooled cross-validation PR-AUC) produced a tie that
  went to the baseline; the fine-tuned model had fewer cross-validation false positives. Revisit
  the rule before treating "v1" as final.
- Labels were assigned by an AI assistant reading the text against the definitions above
  (`label_source = automated_review`); they have not been human-audited. 387 messages are
  excluded as `review_required`.
- Only a single recipient's messages; no group-by-sender analysis yet.

## Documentation

| Document | Contents |
|---|---|
| [docs/model_card.md](docs/model_card.md) | Model specs, data, results and limits |
| [docs/corpus.md](docs/corpus.md) | Schema, labeling definitions and rebuild details |
| [docs/coreml_export.md](docs/coreml_export.md) | Export, tokenizer contract and parity results |
| [docs/raw_extraction.md](docs/raw_extraction.md) | Extracting messages from `chat.db` |
| [docs/synthetic.md](docs/synthetic.md) | The synthetic example dataset |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup and privacy rules |

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) first; it includes the
privacy rules (never share real messages) and development setup.

## Contact

Questions, or anything you'd rather not post publicly? Email **dev.meisel+veto@gmail.com**.
Please don't include private message content in your email.

## License

Released under the [MIT License](LICENSE). The license covers the code in this repository only; it
does not cover, and the repository does not contain, any message data or trained models.
