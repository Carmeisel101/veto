# Veto

Veto is a privacy-focused project for identifying political text messages. The long-term goal is
an on-device classifier that routes unsolicited political SMS from unknown senders out of the
primary Messages inbox.

## Status

Early development. Implemented: corpus construction, binary labeling, a MiniLM embedding
baseline, MiniLM fine-tuning, evaluation/error analysis, and a Python inference API. **Veto v1 is
a pipeline proof, not a usable filter**: the current private dataset has only 3 positive
examples, so every reported metric is anecdotal. No Swift/iOS/Core ML code exists.

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

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
veto corpus build        # raw -> data/corpus/corpus.parquet
veto corpus inspect      # label join, duplicate groups, splits, artifacts/dataset_report.*
veto train               # baseline + fine-tune, compare, save best to models/veto/
veto evaluate            # print saved metrics (artifacts/evaluation.json)
veto predict "Election Day is tomorrow. Vote YES on Proposition 12."
```

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

- Only 3 positives in ~89 labeled messages; validation and test each hold 1 positive.
- Threshold selection and model comparison are therefore essentially noise.
- Labels were assigned by an AI assistant reading the text against the definitions above
  (`label_source = claude_manual_review`); they have not been human-audited.
- Only a single recipient's messages; no group-by-sender analysis yet.

See [docs/corpus.md](docs/corpus.md) for the schema, labeling definitions and rebuild details.
