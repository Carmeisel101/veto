# Veto

Veto is a privacy-focused project for identifying political text messages. The long-term goal is
an on-device classifier that routes unsolicited political SMS from unknown senders out of the
primary Messages inbox.

## Status

Early development — **corpus construction**. Implemented so far: a reproducible pipeline that
turns an extracted Messages export into a cleaned, schema-validated corpus with room for human
labels. No model, feature analysis, or iOS code exists yet.

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
python -m veto.corpus --raw data/raw/data.parquet --out data/corpus/corpus.parquet
```

See [docs/corpus.md](docs/corpus.md) for the schema, labeling definitions and rebuild details.
