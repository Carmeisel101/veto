# Veto corpus

## What it is

The corpus is the cleaned, structured, annotatable collection of messages used later for feature
discovery and model development. It is **not** a training set: no splitting, deduplication,
feature extraction or modeling happens during construction.

All corpus artifacts derive from the user's private Messages data and live under `data/`, which is
gitignored. Never commit them.

## Raw input

A Parquet file (default `data/raw/data.parquet`), treated as immutable, produced from macOS
`chat.db` by `scripts/extract_raw.py` (logic in `src/veto/extraction/`). The extraction keeps
incoming messages only, does **not** filter on non-null `text`, and recovers bodies from the
`attributedBody` blob when `message.text` is NULL (common on recent macOS). By default it keeps
the last 6 months (`--months 0` for everything); a message in several chats yields one row.
Required columns:

| column | meaning |
|---|---|
| `message_id` | Messages `ROWID`; must be non-null and unique |
| `text` | message body (null/empty rows are removed and counted) |
| `is_from_me` | 0/1 |
| `date` | timezone-aware datetime, or Apple timestamps (ns since 2001-01-01 UTC) |
| `sender` | handle (phone/email) |
| `chat_id`, `chat_identifier`, `chat_name` | chat metadata; may be null |

Extra raw columns (e.g. `group_title`) are not carried into the corpus.

## Schema (`data/corpus/corpus.parquet`)

| column | notes |
|---|---|
| `message_id` | int64 |
| `text` | original, unmodified |
| `normalized_text` | see rules below |
| `text_hash` | SHA-256 of `normalized_text`, for duplicate analysis and future grouped splits |
| `sender` | private; provenance/grouping only — **not a predictive feature** |
| `received_at` | UTC timezone-aware datetime |
| `chat_id`, `chat_identifier`, `chat_name` | provenance; empty chat names become null |
| `is_group_chat` | nullable bool; see below |
| `is_from_me` | bool |
| `suggested_label` | reserved for future machine suggestions; always null today |
| `label` | human ground truth: `political`, `non_political`, `uncertain`, or null (unreviewed) |
| `label_source` | `human` when `label` is set, else null |
| `label_confidence` | optional float in [0, 1] |

Rows are sorted by `message_id`; identical input yields identical output.

## Transformation rules

`normalized_text` is deliberately conservative: Unicode NFC, line endings to `\n`, runs of
horizontal whitespace to a single space, per-line trim, 3+ newlines to 2, outer trim. Case,
punctuation, emoji, URLs, numbers and line structure are preserved.

`received_at`: tz-aware dates are converted to UTC; numeric values are treated as Apple
timestamps. Naive datetimes are rejected as ambiguous.

`is_group_chat`: true when `chat_identifier` matches `chat<digits>` (how Messages names group
chats; 1:1 chats use the counterpart's phone/email); false otherwise; null when the message has no
chat. Group chats are kept so they can be included or excluded deliberately.

Duplicates are kept. Compare `text_hash` values to analyze them; deduplication is a later,
explicit decision.

## Labels

- `political`: primary content is political advocacy, campaigning, persuasion, fundraising,
  electioneering, attack messaging, candidate/ballot advocacy or similar solicitation. Judge
  meaning and context, not keywords.
- `non_political`: everything else, including personal messages, codes, reminders, receipts,
  ordinary marketing, recruiting, and non-political spam or scams. *Non-political does not mean
  legitimate.*
- `uncertain`: needs review (e.g. news alerts, civic reminders, unclear issue advocacy). Review
  before using in supervised learning.

Labels come only from humans. Rules are never used to generate ground truth.

### Providing annotations

Annotations live outside the corpus (e.g. `data/annotations/labels.csv`) with columns
`message_id`, `label` and optional `label_confidence`. Pass them on rebuild; matching rows get
`label_source = "human"`. Annotations for unknown ids are counted and warned about.

## Rebuilding

```bash
uv run veto corpus build        # defaults: data/raw/data.parquet -> data/corpus/corpus.parquet

# custom paths (and optional annotations in the original political/non_political scheme):
uv run python -m veto.corpus --raw data/raw/data.parquet --out data/corpus/corpus.parquet \
    --annotations data/annotations/labels.csv
```

Logs contain aggregate counts only, never message text.

## Validation

Hard errors (`ValidationError`): missing columns, missing/duplicate `message_id`, malformed or
naive timestamps, unsupported labels, malformed annotations. Null and empty texts are removed and
reported in the build summary.

## Deliberately not done

Train/test splitting, deduplication, rule-based labeling, feature engineering, TF-IDF/embeddings,
SignaPy analysis, modeling, Core ML, or any iOS work.

## Solicitation labels (training phase)

The binary training labels are kept apart from the corpus in
`data/annotations/solicitation_labels.csv` and joined by `veto corpus inspect` into
`data/processed/labeled.parquet` (adds `label`, `label_name`, `subtype`, `review_required`,
`review_reason`, `group_id`, `split`). The corpus's own `political/non_political/uncertain`
columns are not used for training. `group_id` comes from aggressive normalization (URLs, phone
numbers, digits masked; word-3-shingle Jaccard >= 0.5) and is used only for grouped splitting.
