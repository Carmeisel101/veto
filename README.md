# Veto

Veto is a privacy-focused project for identifying political text messages. The long-term goal is
an on-device classifier that routes unsolicited political SMS from unknown senders out of the
primary Messages inbox.

## Status

Early development. Implemented: corpus construction, binary labeling, a MiniLM embedding
baseline, MiniLM fine-tuning, evaluation/error analysis, and a Python inference API. **Veto v1 is
a first baseline, not a production filter**: the current private dataset has 117 positive
examples (24 each in validation and test), so metrics are rough. No Swift/iOS/Core ML code
exists.

## What counts as political solicitation

Veto answers: *is this unsolicited political outreach the user would want filtered?* Positives
include campaign messages, fundraising, advocacy, petitions, GOTV/election-information outreach,
polls used as outreach, and event/volunteer asks. A friend discussing the election is **not**
solicitation; Veto detects political outreach, not political language. Subtypes
(`campaign`, `fundraising`, ...) are recorded for analysis; the model predicts the binary target
(`political_solicitation` / `not_political_solicitation`). Ambiguous or contentless messages are
marked `review_required` and excluded from training.

## Architecture

Base model: `sentence-transformers/all-MiniLM-L6-v2`. Two approaches are trained and compared:
frozen-embedding + logistic regression (baseline) and a fine-tuned
`BertForSequenceClassification` head. The selected model, its decision threshold and metadata are
saved to `models/veto/`. Duplicate/near-duplicate families (URLs, phones, numbers masked for
grouping only) are never split across train/validation/test.

## Long-term pipeline

```text
Messages → Corpus → Feature Discovery → Model Development → Evaluation
        → On-device Deployment → iOS Message Filter Extension
```

## Privacy

Message datasets are private and **never committed**. `data/` is gitignored; the repository
contains only code to reproduce the pipeline. Corpus processing is local and makes no network
calls. Tests use synthetic messages only.

## Setup

Veto uses [uv](https://docs.astral.sh/uv/). The project pins Python 3.12 (`.python-version`)
because PyTorch does not yet publish wheels for the newest Python releases.

```bash
git clone <repo> && cd veto
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
uv run veto corpus inspect             # label join, duplicate groups, splits, artifacts/dataset_report.*
uv run veto train                      # baseline + fine-tune, compare, save best to models/veto/
uv run veto evaluate                   # print saved metrics (artifacts/evaluation.json)
uv run veto predict "Election Day is tomorrow. Vote YES on Proposition 12."
```

Extraction is only needed to refresh the raw data; every other command reads `data/raw/`.
Training takes roughly 20 minutes on an M-series Mac (mostly cross-validation).

Labels live in the private
`data/annotations/solicitation_labels.csv` (columns `message_id,label,subtype,review_required,
review_reason,label_source`). Python use:

```python
from veto.inference import VetoClassifier
VetoClassifier.load("models/veto").predict("...")
# {'label': ..., 'probability': ..., 'threshold': ..., 'should_filter': ...}
```

Artifacts (`artifacts/`, `models/`) are gitignored: they can contain message text.

## Current limitations

- 117 positives among ~9,000 labeled messages; validation and test hold 24 positives each, so a
  single miss moves recall by about 4 points. Treat metrics as rough.
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

See [docs/corpus.md](docs/corpus.md) for the schema, labeling definitions and rebuild details.
