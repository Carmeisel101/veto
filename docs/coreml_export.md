# Core ML export

How Veto's selected model is exported as a single Core ML artifact, how that artifact is checked
against Python inference, and the contract a consumer (the future `veto-ios` repository) relies
on. **No Swift code lives in this repository**; this milestone ends at a validated artifact
boundary.

## Why this exists

The training pipeline produces a PyTorch/scikit-learn model that only runs in Python. A phone app
needs a compact, self-contained model plus an exact description of how to feed it. The export
turns the selected model into:

```text
Veto internals ──▶ veto export coreml ──▶ dist/coreml/ ──▶ veto-ios
```

```text
dist/coreml/
├── veto.mlpackage/            the model (Core ML ML Program, FP16)
├── metadata.json              threshold, labels, shapes, tokenizer contract, versions, validation
└── tokenizer/
    ├── tokenizer.json         full tokenizer definition (Hugging Face "tokenizers" format)
    ├── tokenizer_config.json
    ├── vocab.txt              WordPiece vocabulary, one token per line, line number = id
    └── golden_tokenization.json   example texts with their exact token ids (for testing)
```

`dist/` is gitignored and enforced by CI: the package contains weights trained on private
messages, so it is never committed (see the README on why weights are not published).

## What gets exported

The export follows the model selected by Veto's training (`models/veto/metadata.json`). The
current v1 is the **baseline**:

```text
Python (training/inference)               Core ML (one graph)
─────────────────────────────             ───────────────────────────────────────────
text                                      input_ids, attention_mask   (from the tokenizer)
 └─ tokenizer (BERT uncased, 256)          └─ MiniLM encoder (6 layers, 384-d)
     └─ MiniLM encoder                          └─ mean pooling over attention_mask
         └─ mean pooling                             └─ L2 normalization
             └─ L2 normalization                          └─ linear layer (logistic regression)
                 └─ logistic regression                        └─ sigmoid
                     └─ probability                                 └─ political_solicitation_probability
```

The learned logistic-regression coefficients (384 weights and the intercept) are copied into a
linear layer, so the app does not reimplement pooling or regression. If the fine-tuned model is
selected instead, the exporter reproduces that path (encoder, pooler, 2-class head, softmax); that
path is covered by tests but has not been exported from real data.

**Nothing is retrained.** The wrapper (`src/veto/export/wrapper.py`) loads the saved parameters
and `verify` checks it against the existing Python inference (differences of about 4e-7 for the
baseline).

### Why the encoder forward is written out explicitly

Tracing Hugging Face's stock `BertModel` forward with transformers 5.x fails in Core ML
conversion (it builds attention masks with Python `int()` casts on tensors). Rather than change
the learned behavior or pin an old library, the wrapper runs the same trained `Linear` and
`LayerNorm` layers through an explicit forward pass (plain matmul, softmax, layer norm, GELU).
No operation in the model is unsupported by Core ML.

## Tokenization contract

Tokenization happens **outside** the Core ML model: the app converts text to two integer tensors.

| item | value |
|---|---|
| Tokenizer | BERT WordPiece, `all-MiniLM-L6-v2` vocabulary (30,522 tokens) |
| Casing | **Lower-cased.** Capital letters are lost before the model sees the text |
| Accents | Stripped (follows lower-casing in the Hugging Face BERT normalizer) |
| Cleaning | Control characters removed; CJK characters split into single tokens |
| Unknown characters | Map to `[UNK]` (id 100). **Emoji are `[UNK]`.** |
| Template | `[CLS] tokens… [SEP]` (ids 101 … 102) |
| Max length | 256 tokens, including `[CLS]` and `[SEP]` |
| Truncation | Longest-first, right side |
| Padding | Right-pad to exactly 256 with `[PAD]` (id 0) |
| `attention_mask` | 1 for real tokens (including `[CLS]`/`[SEP]`), 0 for padding |
| `token_type_ids` | Not an input; the model assumes a single segment |
| dtype | `int32` |

`tokenizer/golden_tokenization.json` lists the committed, invented fixture texts with the exact
unpadded `input_ids` Python produces. A Swift tokenizer is correct only if it reproduces those ids
for all examples, including the emoji, accent and punctuation cases. (The Hugging Face
`swift-transformers` package can read `tokenizer.json`; otherwise implement BERT basic
tokenization plus WordPiece over `vocab.txt`.)

## Model interface

| | name | shape | dtype |
|---|---|---|---|
| input | `input_ids` | `[1, 256]` | int32 |
| input | `attention_mask` | `[1, 256]` | int32 |
| output | `political_solicitation_probability` | `[1]` | float |

Batch size is fixed at 1 (one message at a time) and the sequence length is fixed at 256, which
keeps shapes static for Core ML. Messages shorter than 256 tokens are padded and masked, so the
result matches unpadded Python inference.

The output is a probability in [0, 1] of `political_solicitation`.

## Threshold

The decision threshold is **not** part of the graph. It lives in `metadata.json`:

```text
should_filter = political_solicitation_probability >= metadata.threshold
```

The current value is 0.8700 (chosen on the validation set for high precision). Changing the
threshold does not require re-exporting the model.

```text
String
  │
  ▼
tokenizer (see contract above)
  │
  ├── input_ids
  └── attention_mask
          │
          ▼
    veto.mlpackage
          │
          ▼
   probability
          │
          ▼
   metadata.threshold
          │
          ▼
    should_filter
```

**Heads-up on the current threshold.** Veto's threshold selection places the threshold exactly at
one validation positive's own score (a tie of zero margin). Any reduced-precision runtime can
flip that single message. A small safety margin in `select_threshold` (for example, the midpoint
between neighboring scores) would remove the edge case. That is a training-side change and is not
made here.

## Precision and deployment target

- Format: Core ML **ML Program** (`mlprogram`), **FP16** compute precision.
- Minimum deployment target: **iOS 17** (default; `--deployment-target iOS16|iOS17|iOS18`).
- No quantization or palettization was applied. FP16 comes to 45.2 MB, and weight compression was
  intentionally not explored until parity was established (see Limitations).

## How to run

```bash
uv sync --extra coreml                  # once: installs coremltools
uv run veto export coreml               # exports models/veto -> dist/coreml and verifies it
```

Options:

| flag | default | meaning |
|---|---|---|
| `--model-dir` | `models/veto` | the trained model to export (the selected one) |
| `--output` | `dist/coreml` | output directory (replaced if it exists) |
| `--precision` | `fp16` | `fp16` or `fp32` |
| `--deployment-target` | `iOS17` | `iOS16`, `iOS17` or `iOS18` |
| `--no-verify` | off | skip the parity check |
| `--local-sample N` | all | number of local validation/test messages to compare |
| `--benchmark` | off | measure latency on this Mac |

Core ML export and prediction require macOS. The export takes about 100 seconds (the parity check
on 3,614 messages dominates). `uv run pytest` runs the export tests only where `coremltools` is
installed.

## How parity is verified

The reference is Veto's own Python inference (`VetoClassifier`). Core ML is judged as a
deployment-equivalent copy, not as a new model. `veto export coreml` compares them on:

1. **Committed fixtures** (`src/veto/export/parity_fixtures.json`): 70 invented messages covering
   campaign solicitation, fundraising, advocacy, election reminders, candidate updates, ballot
   measures, polls, personal conversation, personal political conversation, transactional,
   commercial, spam, short and long messages, URLs, and STOP/END language.
2. **Threshold bands**: invented messages whose *Python* probability is well below, near and above
   the threshold (25 each), generated deterministically at run time.
3. **Local corpus** (private, never committed): all 3,614 validation and test messages.

Each comparison reports max, mean and p95 absolute probability difference, and the number of
`should_filter` disagreements. The command runs it for Core ML FP16 on all compute units, FP16 on
CPU only, and an **FP32 Core ML reference** (to separate precision effects from conversion bugs).
Only aggregates are written to `artifacts/coreml_export_report.json`; no message text.

### Acceptance criteria

- **Committed fixtures: 0 classification disagreements**, and max probability difference ≤ **0.01**
  (FP16) or ≤ 1e-4 (FP32).
- **Tolerance rationale:** observed FP16 maxima are 0.0071 (fixtures), 0.0061 (3,614 local
  messages) and 0.0036 (near threshold). 0.01 leaves about 1.4x headroom and is roughly 100x
  tighter than a real conversion bug (wrong pooling, mask or weights) would produce.
- **Elsewhere:** a disagreement is accepted ("explained") only if the message sits closer to the
  threshold than the largest probability deviation observed. Anything else fails.

### Results (current v1, M2 Pro, FP16)

| comparison | n | max diff | mean diff | p95 diff | disagreements |
|---|---|---|---|---|---|
| Fixtures (ALL units) | 70 | 0.0071 | 0.0011 | 0.0033 | **0** |
| Fixtures (CPU only) | 70 | 0.0061 | 0.0009 | 0.0036 | **0** |
| Band: below threshold | 25 | 0.0047 | 0.0022 | 0.0045 | 0 |
| Band: near threshold | 25 | 0.0036 | 0.0018 | 0.0035 | 2 (explained) |
| Band: above threshold | 25 | 0.0008 | 0.0003 | 0.0006 | 0 |
| Local corpus (ALL units) | 3,614 | 0.0061 | 0.00037 | 0.0015 | 1 (explained) |
| Local corpus (CPU only) | 3,614 | 0.0043 | 0.00017 | 0.0007 | 0 |
| **FP32 Core ML reference** | 3,614 | 1e-6 | 0.0 | 0.0 | **0** |

Findings:

- The FP32 Core ML model matches Python to about 1e-6 with no disagreements anywhere, so the
  wrapper and the conversion are exact. All deviations come from FP16 rounding.
- The near-threshold flips are messages whose Python probability is within 0.0002 of the
  threshold; the single flip among the 3,614 real messages is the validation message whose own
  score defines the threshold (see "Heads-up" above). No flip involves a message farther from the
  threshold than the FP16 noise.
- False positives and false negatives against the labels are unchanged (0 and 5) on the local
  set in FP32 and in FP16 on CPU. In FP16 on all compute units one extra false negative appears
  (the threshold-defining message).
- The verdict is therefore **PASS_WITH_NOTES**.

## Artifact size

| part | size |
|---|---|
| `veto.mlpackage` | 45.21 MB (45,207,168 bytes) |
| `tokenizer/` (4 files) | 0.98 MB |
| `metadata.json` | 3.7 KB |
| **Total deployment footprint** | **46.19 MB** |

Most of the size is the embedding table: about 11.9M of the model's 22.7M parameters.

## Performance (development Mac only)

Apple M2 Pro, 200 timed runs after 20 warm-up runs, batch size 1. **These are Mac measurements,
not iPhone measurements.**

| step | median | p95 |
|---|---|---|
| Tokenization (Python, Hugging Face fast tokenizer) | 0.15 ms | 0.19 ms |
| Core ML inference, all compute units | 2.24 ms | 2.56 ms |
| Tokenization + inference, all compute units | 2.43 ms | 2.82 ms |
| Core ML inference, CPU only | 5.61 ms | 6.32 ms |

Swift tokenization speed will differ from the Python figure.

## Current limitations

- **Not measured on an iPhone.** Latency, memory and Neural Engine behavior on device are untested.
  A message-filter extension runs under tight memory limits that were not checked here; 46 MB of
  model plus runtime memory should be measured on a real device before building on it.
- **No weight compression.** FP16 was validated and kept. If device measurements show the size is
  a problem, compressing the embedding table (for example, 8-bit palettization) is the obvious
  experiment, and must be re-validated with this same parity suite.
- **Fixed shape** `[1, 256]`: every message is padded to 256 tokens, so short messages cost the
  same as long ones. Enumerated sequence lengths are a possible later optimization.
- **FP16 resolution near the threshold**: messages within about 0.004 of the threshold can change
  decision versus Python. See the threshold note above.
- **Tokenizer fidelity is the consumer's job**: lower-casing, accent stripping and emoji-to-`[UNK]`
  behavior must be reproduced exactly (see the golden file).
- **Toolchain**: `coremltools` 9.0 officially supports up to torch 2.7; this project uses 2.14.1
  and works, with warnings. `torch.jit.trace` is deprecated in torch 2.14, so a future torch
  release may require switching the export path. `coremltools` also prints an internal
  "overflow encountered in cast" warning during conversion that does not affect results (parity
  was verified).
- **Fine-tuned model path** is implemented and tested on a tiny model but has not been exported
  from a trained fine-tuned model of real data.
- **Determinism**: exports of the same model give identical predictions; the `.mlpackage` files
  themselves are not byte-identical (they embed identifiers).
