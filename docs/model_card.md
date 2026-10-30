# Veto model card (scorecard)

A plain-language summary of what Veto's classifier is, how it was built and evaluated, and where
it should and should not be used. The numbers below come from the author's own training run on a
private dataset (aggregates only; no message text appears here). **Trained weights are not
distributed**: each user trains their own model locally (see the README), and their numbers will
differ.

## At a glance

| | |
|---|---|
| **Task** | Binary text classification: `political_solicitation` vs `not_political_solicitation` |
| **Input** | One SMS/iMessage text (first 256 tokens are used) |
| **Output** | Probability of political solicitation, a stored decision threshold, and `should_filter` |
| **Base model** | `sentence-transformers/all-MiniLM-L6-v2` (22.7M parameters, Apache-2.0) |
| **Approaches compared** | (a) frozen embeddings + logistic regression, (b) fine-tuned MiniLM classifier |
| **Model saved as v1** | (a) the baseline, because a selection-rule tie went to the simpler model |
| **Training data** | One person's private messages: 9,035 labeled, 117 positive (1.3%) |
| **Labels** | Assigned by an AI assistant reading each message (`automated_review`); not human-audited |
| **Headline test result** | Fine-tuned: precision 1.00, recall 0.875 (21 of 24). Baseline: precision 1.00, recall 0.79 (19 of 24) |
| **Reliability** | Low-to-moderate: only 24 positives in the test set |
| **Status** | Research prototype. **Not production-ready** |

## Intended use

Veto asks: *is this unsolicited political outreach that the user would want filtered?* The
intended use is a research prototype for routing political solicitation texts (campaign
messages, fundraising, advocacy, election information, polls, petitions, event asks) from
unknown senders away from a user's main inbox, running on the user's own device.

It detects **political outreach, not political language**. A friend discussing the election
should not be flagged; a nonpartisan "make your plan to vote" reminder should.

**Out of scope / not recommended**
- General "politics detection", opinion classification, or measuring political content.
- Monitoring, profiling or censoring people's conversations, or filtering messages from known
  contacts.
- Any decision with legal, electoral or safety consequences.
- Languages other than English (the data is almost entirely English; a few Spanish polling texts).
- Treating its output as ground truth without human review.

## Model details

### Pipeline

```text
message ─▶ tokenizer (BERT uncased WordPiece, max 256 tokens)
        ─▶ MiniLM encoder (6 layers, 384-d)
        ─▶ classifier ─▶ P(political_solicitation) ─▶ compare with stored threshold ─▶ should_filter
```

### Base model: `all-MiniLM-L6-v2`

Specs below were read from the model's configuration and loaded weights.

| spec | value |
|---|---|
| Architecture | BERT-style Transformer encoder (`model_type: bert`) |
| Transformer layers | 6 |
| Hidden size | 384 |
| Attention heads | 12 (32 dimensions per head) |
| Feed-forward (intermediate) size | 1,536, GELU activation |
| Vocabulary | 30,522 WordPiece tokens, **uncased** (text is lower-cased) |
| Max position embeddings | 512 (the sentence-transformers wrapper caps input at 256) |
| Output embedding | 384-dimensional (mean-pooled, L2-normalized in the sentence-transformers wrapper) |
| **Total parameters** | **22,713,216** |
| of which: embeddings | 11,918,592 (52%) |
| of which: encoder layers | 10,646,784 (47%) |
| of which: pooler | 147,840 (0.7%) |
| Size on disk / in memory (fp32) | about 91 MB of weights (about 4 bytes per parameter) |
| License | Apache-2.0 |

Background (from the model's Hugging Face card): it is a sentence-embedding model produced by
fine-tuning a small MiniLM Transformer (a distilled model from Microsoft's MiniLM research) with a
contrastive objective on over a billion sentence pairs. It is general-purpose and was not trained
for political or SMS text.

Two consequences worth knowing:
- **Uncased input.** The model never sees capitalization, even though Veto's corpus preserves it.
  Signals like `URGENT` or `STOP` in capitals are lost to the model.
- **Long messages are truncated** at 256 tokens. Veto's median message is 28 characters, the
  90th percentile 101, but the longest is over 6,500 characters.

### Variant A: embeddings + logistic regression (the saved v1)

- The MiniLM encoder is **frozen** and produces a 384-d normalized embedding per message.
- A logistic regression classifier (385 parameters) is trained on those embeddings: `C = 1.0`,
  class-balanced weights, `max_iter = 1000`.
- Cheap to train and easy to inspect, but it can only use what the frozen embedding captures.

### Variant B: fine-tuned MiniLM classifier

- `BertForSequenceClassification` initialized from the same checkpoint, with a new 2-class head
  (770 parameters) on top of the pooled output. **22,713,986 parameters in total**, all trained.
- Class-weighted cross-entropy (positive class weight capped at 10, because positives are rare).
- AdamW, learning rate 3e-5, weight decay 0.01, 10% linear warmup then linear decay, batch size
  8, up to 6 epochs, gradient clipping at 1.0.
- Early stopping with patience 2 on validation PR-AUC; the best epoch is restored.
- Seed 42 (Python, NumPy, PyTorch); every setting is in `TrainConfig`
  (`src/veto/training/config.py`) and is saved in the model metadata.

### Speed and footprint (measured)

Single-message inference with the fine-tuned model on an Apple M2 Pro (PyTorch, batch size 1):

| device | median latency | p95 |
|---|---|---|
| CPU | about 6 ms | about 7 ms |
| Apple GPU (MPS) | about 4-5 ms | about 6 ms |

The full pipeline (both models plus cross-validation) trains in roughly 20 minutes on that
machine. A Core ML export of the saved v1 exists (see [coreml_export.md](coreml_export.md)); it measured
2.2 ms per message on this Mac's Neural Engine/GPU path, but on-device (iPhone) performance has
**not** been measured.

### Decision threshold

The threshold is chosen on the validation set as the one with the highest recall subject to
validation precision ≥ 0.95 (false positives are costly, since they hide real messages). The
saved v1 threshold is 0.87. It is stored in the model metadata and is **not** tuned on the test
set. With only 24 validation positives, the chosen threshold is itself uncertain.

## Training data

(Aggregates only; the data is private and never published.)

- **Source:** one person's incoming SMS/iMessage history from the last six months, extracted
  locally from macOS Messages. Mostly personal conversation.
- **Size:** 9,422 messages after cleaning; 9,035 labeled and usable (117 positive, 8,918
  negative); 387 marked `review_required` and excluded (376 with no text, such as attachments;
  11 ambiguous).
- **Positive subtypes:** 59 campaign, 25 fundraising, 19 polling or survey, 8 advocacy, 3
  election information, 2 political event, 1 other.
- **Negative subtypes:** 8,642 personal, 176 commercial, 52 transactional, 42 other, 6
  non-political spam. Hard negatives (friends discussing politics, news alerts, non-political
  fundraising, promos using political words) are included.
- **Duplicates:** 1,469 messages share their exact text with another; messages in the same
  duplicate/near-duplicate family are never split across train, validation and test.
- **Splits (grouped, stratified, seed 42):** train 5,421 (69 positives), validation 1,807 (24),
  test 1,807 (24).
- **Labels:** assigned by an AI assistant reading each message against the written definitions
  (README and `docs/corpus.md`). No human has audited them, and metrics measure agreement with
  these labels.

## Evaluation

Test and validation figures are at each model's own validation-selected threshold. The
"cross-validation" column is grouped, stratified 3-fold cross-validation over train + validation
(93 positives, a fixed 0.5 threshold), a steadier estimate than the 24-positive test set.

| | Baseline (saved v1) | Fine-tuned MiniLM |
|---|---|---|
| **Test precision** | 1.00 | 1.00 |
| **Test recall** | 0.79 (19 of 24) | 0.875 (21 of 24) |
| **Test F1** | 0.88 | 0.93 |
| Test accuracy | 0.997 | 0.998 |
| Test ROC-AUC / PR-AUC | 0.997 / 0.95 | 0.999 / 0.97 |
| **Test confusion matrix** (TN / FP / FN / TP) | 1783 / 0 / 5 / 19 | 1783 / 0 / 3 / 21 |
| Validation precision / recall | 1.00 / 1.00 | 0.96 / 1.00 |
| **CV false positives** | 33 | **6** |
| CV false negatives | 2 | 2 |
| CV precision / recall / F1 | 0.73 / 0.98 / 0.84 | 0.94 / 0.98 / 0.96 |
| CV PR-AUC | 0.971 | 0.970 |

**Reading these numbers**
- Accuracy is misleading here, because 98.7% of messages are negative. Look at precision and
  recall for the political class.
- With 24 positives, one missed message moves test recall by about 4 points.
- **Why the baseline was saved:** the selection rule (pooled cross-validation PR-AUC) produced a
  tie, 0.971 versus 0.970, and ties go to the simpler model. The fine-tuned model has better test
  recall and far fewer cross-validation false positives, so this choice should be revisited.

### Error analysis

The five baseline misses on the test set (three of them also missed by the fine-tuned model)
are all **political outreach without the usual cues** (no donation ask, no attack on an opponent,
no "vote"):

- Three are near-identical event and advocacy invitations from one advocacy group about a
  technology-policy panel (about 53-96 words, with a link and an opt-out line). Because they are
  one family, they behave like a single under-represented example, not three independent ones.
- One is a short (20-word) advocacy message announcing a guide on a school-policy issue.
- One is a candidate thank-you note with a link but no request.

The fine-tuned model missed three: one of the panel invitations (scored 0.90, just under its
0.986 threshold) and the last two items above, which it scored near zero (0.0003 and 0.001).
That means it was confidently wrong on phrasing it had not seen, not just unsure. It also made
one validation false positive: a personal message scored 0.996 despite having no political
content. Cross-validation false positives (33 baseline, 6 fine-tuned) were not broken down by
type.

### Synthetic demo (for contributors)

`veto demo` trains on the synthetic dataset in `examples/synthetic/` (1,000 invented messages,
10% positive). That run is a pipeline smoke test only: the fine-tuned demo model reached test
precision 1.00 and recall 0.74, and it misses some short election-information messages. Its
numbers say nothing about performance on real messages.

## Limitations and risks

- **Small, narrow data.** 117 positives from a single person's inbox, dominated by a few
  campaigns in one election cycle and region. The model may partly learn candidate names and
  campaign templates rather than the general idea of political outreach, and could degrade on new
  candidates, other regions, other elections and other languages.
- **Under-represented categories.** Election information (3 examples), political events (2),
  petitions (0) and issue advocacy (8) barely appear, and these are the main failures.
- **Label noise.** Labels came from an AI assistant, not a human panel. Ambiguous cases (surveys
  with no stated topic, nonpartisan reminders) were handled by judgment calls and may not match
  other people's definitions.
- **Overconfidence.** Scores are not calibrated; a message can get a near-zero or near-one
  score without a meaningful uncertainty estimate.
- **Cost of errors.** A false positive hides a message the user wanted. The threshold favors
  precision, but with so few positives, any precision figure is rough.
- **Uncased, truncated input** (see above) and no use of sender, time or conversation context by
  design: sender numbers are deliberately not used because campaigns rotate them.
- **Possible misuse.** A political-message classifier could in principle be used to monitor or
  suppress speech. Veto is intended for opt-in filtering on a user's own device and is not
  designed or tested for anything else.

## Privacy and ethics

- Training data is private personal messages. It stays on the author's machine, is gitignored
  and is never uploaded; CI rejects commits that add it.
- **Trained weights are not published.** A fine-tuned language model can memorize pieces of its
  training text, such as names or phone numbers, so each user trains their own model from their
  own messages.
- Training and inference run entirely locally with no network calls (apart from downloading the
  public base model).
- Public figures appear by name in some messages the model was trained on; the repository
  itself contains only invented examples.

## How to use or reproduce

```bash
uv run veto corpus build && uv run veto train     # train your own model (see the README)
uv run veto evaluate                              # view accuracy, confusion matrices, etc.
uv run veto predict "..."
```

```python
from veto.inference import VetoClassifier
VetoClassifier.load("models/veto").predict("...")
# {"label": ..., "probability": ..., "threshold": ..., "should_filter": ...}
```

Metrics and confusion matrices are saved to `artifacts/evaluation.json`; the saved model's
threshold, hyperparameters and metrics are in `models/veto/metadata.json`.

## Version and contact

- **Version:** v1 (research prototype), trained with seed 42 on the author's data.
- **Open items before any stronger claims:** human audit of labels; more advocacy, event and
  election-information examples; revisit the baseline-vs-fine-tune selection rule; evaluate on a
  second person's data.
- **License:** the code is MIT (see `LICENSE`); the base model is Apache-2.0.
- **Contact:** dev.meisel+veto@gmail.com
