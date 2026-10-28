# Synthetic example dataset

Veto's real training data is private, so the repository ships a **fully synthetic** dataset that
lets anyone run the whole pipeline end to end without their own messages.

Everything in it is invented: names, organizations, brands, phone numbers (reserved fictional
`555-01XX` numbers), and URLs (the reserved `.example` domain). No real message or person was
used, and nothing was derived from the author's data.

## Files

| file | contents |
|---|---|
| `examples/synthetic/raw.parquet` | 1,000 messages in the raw-export schema (`docs/raw_extraction.md`) |
| `examples/synthetic/solicitation_labels.csv` | labels in the annotation schema (`docs/corpus.md`), `label_source = synthetic` |

## Composition (1,000 messages)

- **100 positives (10%)**: political solicitation in the styles Veto targets (campaign messages
  and attack messaging, fundraising, issue advocacy, election information, polls, petitions,
  rallies/volunteer asks). Candidates and organizations have random invented names.
- **881 negatives**: personal chat (most), transactional (codes, appointments, shipping),
  commercial promos, non-political spam/scams, recruiting and non-political fundraising.
  Includes **hard negatives**: friends discussing politics, news alerts that mention candidates,
  and promos that use political words ("Vote for your favorite flavor!", "Election Day Sale").
- **19 `review_required`** rows (attachment-only messages and an ambiguous survey), so the
  review/exclusion path is exercised.
- Campaign templates repeat with different names and links, so duplicate/near-duplicate grouping
  has something to do. About 26% of texts are exact duplicates.

Labels are known by construction, so they are ground truth for the synthetic data, but they say
nothing about how the model would do on real messages.

## Run it

```bash
uv run veto demo        # build corpus -> label -> split -> baseline + fine-tune -> save (about 2 min)
uv run veto evaluate --artifacts demo/artifacts
uv run veto predict --model-dir demo/models/veto "Dana Whitfield here. Ravi Oyelaran has spent years raising your taxes. Make your plan to vote early: vote-now.example/Ab12"
```

All outputs go to `demo/` (gitignored), separate from your real `data/`, `models/` and
`artifacts/`. Use `--raw`, `--labels` and `--out-dir` to point the demo at other files.

The demo model is only a demonstration of the pipeline. With 100 template-driven positives it
misses some political messages (for example, shorter election-information texts). Treat its
metrics as a smoke test, not a benchmark.

## Regenerating

```bash
uv run python scripts/make_synthetic_data.py [--n 1000] [--positive-rate 0.10] [--seed 42]
```

The generator (`src/veto/synthetic.py`) is deterministic for a given seed and is covered by tests
(`tests/test_synthetic.py`).
