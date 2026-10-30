# Contributing to Veto

Thanks for your interest in Veto! This project identifies unsolicited political text messages
using a small on-device-friendly classifier. Contributions of code, tests, docs, labeling
guidelines and ideas are welcome.

Please read the **privacy rules** first. They matter more here than in most projects.

## Privacy rules (non-negotiable)

Veto is built from people's private text messages. To keep contributors and bystanders safe:

- **Never commit, paste or attach real messages**, phone numbers, email addresses, names of
  private individuals, or message screenshots. This applies to code, tests, docs, notebooks,
  issues, PR descriptions and comments.
- **Use synthetic data** for tests and examples. The generator in `src/veto/synthetic.py` and
  the dataset in `examples/synthetic/` exist for this purpose (see
  [docs/synthetic.md](docs/synthetic.md)). Made-up values are fine; use reserved ones such as
  `555-01XX` phone numbers and `.example` domains.
- **Clear notebook outputs before committing** (`Cell > All Output > Clear`). Saved outputs
  often contain message text.
- **Do not commit** `data/`, `models/`, `artifacts/`, `checkpoints/`, `outputs/`, `dist/` or `eda/`. They
  are gitignored; don't override that. Files in `artifacts/` (such as `error_analysis.json`)
  contain message text.
- **Do not upload** models trained on private data. A fine-tuned model can memorize parts of its
  training text. Everyone trains their own model locally (see the README).
- **Don't send your data anywhere**: no external APIs, no hosted LLMs, no uploading to model hubs.
  The pipeline is designed to run entirely on your machine.
- When reporting a bug or a mislabeled example, **describe it or rewrite it** instead of pasting
  the original message. Replace names, numbers and links with placeholders.

### Automated checks

Every pull request runs three checks in GitHub Actions (`.github/workflows/ci.yml`), and all
must pass before merging:

- **No private data committed**: fails if any commit in the PR adds a file under `data/`,
  `models/`, `artifacts/`, `checkpoints/`, `outputs/`, `eda/` or `demo/`, a Messages database,
  trained-model weights, a Parquet file outside `examples/`, or a notebook with saved outputs.
  It looks at every commit in your branch, so deleting a file in a later commit does not hide it.
  Run the same check locally with `python3 scripts/check_private_files.py --base main`.
- **Lint**: `ruff check .`
- **Tests**: `pytest`, run offline (model downloads are blocked, so tests must not need real
  weights).

If you accidentally commit private data, email dev.meisel+veto@gmail.com right away so the
history can be rewritten. Deleting the file in a later commit is not enough.

## Getting set up

Veto uses [uv](https://docs.astral.sh/uv/) and pins Python 3.12 (PyTorch does not yet ship
wheels for the newest Python releases).

```bash
git clone <repo-url> && cd veto
uv run pytest            # creates the environment on first run, then runs the tests
uv run veto demo         # optional: run the whole pipeline on the synthetic dataset
```

Keep your clone **outside iCloud Drive** (for example `~/Code/veto`). iCloud hides files inside
the virtualenv and breaks `import veto`. See
[docs/raw_extraction.md](docs/raw_extraction.md#running-inside-icloud-drive) for details.

## Making changes

1. **Open an issue first** for anything bigger than a small fix, so we can agree on the
   approach before you invest time.
2. **Branch** from `main` with a short descriptive name, for example `fix/threshold-tie` or
   `feature/char-ngrams`.
3. **Keep PRs focused.** One logical change per PR is much easier to review.
4. **Write tests** for new behavior and bug fixes. Tests must use synthetic data only and must
   not download large model weights; use the existing fixtures and fakes where you can (see
   `tests/inference/test_inference.py` for a tiny offline model example).
5. **Run the checks** before you push:
   ```bash
   uv run pytest
   uv run ruff check .
   python3 scripts/check_private_files.py --base main   # no private files in your commits
   ```
6. **Update the docs** (`README.md`, `docs/`) when you change behavior, options or file formats.

### Code style

- Python 3.10+ code, typed function signatures, short docstrings where behavior is not obvious.
- Lint with `ruff` (line length 100, configured in `pyproject.toml`).
- Match the surrounding code: small focused modules, no hidden global state, no hard-coded
  absolute paths or usernames, and clear exceptions instead of swallowed errors.
- Keep I/O separate from transformation logic, so the logic can be tested without files.
- Keep dependencies minimal. Say why in the PR if you add one.
- **Logging must never include message text.** Log aggregate counts only.

### Commit messages

Short imperative summaries; a conventional prefix is welcome:
`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`.
Example: `fix: handle empty validation split in threshold selection`.

## Areas where help is especially useful

- **More diverse synthetic data** that covers under-represented cases (issue advocacy, events,
  petitions, nonpartisan election information, other languages).
- **Evaluation and error analysis** improvements, such as better handling of tiny validation sets.
- **Labeling guidelines**: clearer definitions and edge cases.
- **Tests and docs.**

## Changing the labeling definitions

Veto detects **political outreach, not political language**. A friend discussing the election
is *not* a solicitation; a nonpartisan "make your plan to vote" reminder *is*. The definitions
in the README and [docs/corpus.md](docs/corpus.md) drive every downstream result, so changes to
them need an issue and discussion first, and the affected tests, docs and synthetic labels must
be updated in the same PR. Do not label examples with a keyword rule and call the result ground
truth; labels should reflect a human reading the message.

## Scope

The current scope is the Python training and inference pipeline. These are intentionally **out
of scope** for now: the Swift/iOS app, the iOS Message Filter Extension, Core ML conversion, a
hosted API or server, and on-device optimization. PRs for those will likely be asked to wait.

## Reporting bugs

Open an issue with: what you ran, what you expected, what happened, your OS and Python version,
and (if relevant) the output of `uv run veto evaluate`. Remember to **remove any private
message content** from logs and screenshots before posting.

## Security

If you find a privacy or security problem, such as a path where message text could leak, please
report it privately by email to **dev.meisel+veto@gmail.com** instead of opening a public issue.

## Contact

Questions about Veto, or anything you'd rather not post publicly? Email
**dev.meisel+veto@gmail.com**. Please don't include private message content in your email either.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
