# Raw message extraction

How Veto turns the macOS Messages database into `data/raw/data.parquet`, the immutable input to
corpus construction (see [corpus.md](corpus.md)).

The output is **private**: it contains message contents and phone numbers. It lives under
`data/`, which is gitignored. Extraction runs entirely locally and makes no network calls.

## Quick start

```bash
uv run python scripts/extract_raw.py --overwrite          # last 6 months (default)
uv run python scripts/extract_raw.py --overwrite --months 0   # everything
```

Expected log (counts only, never message text):

```text
Read 21530 incoming message rows.
Collapsed 6 rows for messages in multiple chats.
Text NULL for 21349 messages; recovered 21223 from attributedBody.
Dropped 12015 messages older than 6 months.
Saved 9509 messages to data/raw/data.parquet.
```

(The numbers above are from one real run; yours will differ.)

### Options

| flag | default | meaning |
|---|---|---|
| `--db` | `~/Library/Messages/chat.db` | path to the Messages database |
| `--out` | `data/raw/data.parquet` | output file |
| `--months` | `6` | keep the last N months; `0` keeps everything |
| `--timezone` | `America/Chicago` | timezone of the output `date` column |
| `--overwrite` | off | replace an existing output file |

Raw data is treated as immutable, so the script refuses to replace an existing output unless you
pass `--overwrite`.

### Requirements

- **Full Disk Access.** `chat.db` is protected by macOS. Grant your terminal app (Terminal,
  iTerm, or the Claude desktop app) access in System Settings → Privacy & Security → Full Disk
  Access, then restart it. Without this, the script fails with
  `sqlite3.OperationalError: unable to open database file`.
- The database is opened **read-only** (`mode=ro`); extraction never modifies Messages.
- Messages must be present in `chat.db`, so use a Mac signed in to the relevant iMessage/SMS
  account (e.g. with Text Message Forwarding enabled for SMS from an iPhone).

## Why `message.text` is usually empty

On recent macOS versions (Ventura and later), the `text` column of the `message` table is
frequently `NULL`, even for ordinary messages. The body is stored instead in the
`attributedBody` column as a binary blob: a serialized `NSAttributedString`, which Apple encodes
in the legacy "typedstream" format. The blob holds the text plus formatting metadata.

A query that filters on `message.text IS NOT NULL` therefore silently discards most messages. In
one real database, only 175 of 21,524 incoming messages had `text` populated; 21,223 of the rest
could be recovered from `attributedBody`. That filter is deliberately **not** used here.

Messages with no text anywhere (attachment-only messages, reactions, and similar) remain in the
raw file with a null or placeholder text. They are not dropped at extraction time. Corpus
construction removes null/empty text and counts the removals.

## Decoding `attributedBody`

Implemented in `decode_attributed_body()` in
[`src/veto/extraction/messages.py`](../src/veto/extraction/messages.py). It does not parse the
whole typedstream; it extracts just the text string:

1. Find the class-name marker `NSString` in the blob.
2. Skip the marker and 5 bytes of typedstream framing that follow it.
3. Read the length prefix:
   - first byte < `0x80`: that byte is the length, text starts at the next byte;
   - `0x81`: length is the next 2 bytes, little-endian, text follows;
   - `0x82`: length is the next 4 bytes, little-endian, text follows.
4. Read that many bytes and decode as UTF-8 (undecodable bytes become `U+FFFD`).

The length is in **bytes**, not characters, so emoji and other multibyte text are handled
correctly. If the marker is missing or the blob is truncated, the function returns `None`.

The decoded text is used only when `message.text` is NULL; a populated `text` column always
wins. The blob itself is not saved to the output.

This follows the layout seen on recent macOS versions. It is a heuristic, not a full typedstream
parser, so a future macOS change to the format could break it. If many messages come back empty
or with `U+FFFD`, inspect a sample blob and adjust the decoder (the tests in
`tests/extraction/test_messages.py` show the expected layout).

## What the extraction does

1. Query incoming messages (`is_from_me = 0`), joining `handle` (sender) and `chat` metadata.
2. Collapse messages that belong to several chats into one row (the lowest chat id), so
   `message_id` is unique, as corpus validation requires.
3. Fill NULL `text` from the decoded `attributedBody`.
4. Convert Apple's timestamp (nanoseconds since 2001-01-01 UTC) to a timezone-aware datetime in
   the chosen timezone.
5. Apply the `--months` window.
6. Write Parquet with the columns `message_id, text, is_from_me, group_title, date, sender,
   chat_id, chat_identifier, chat_name`.

## Running inside iCloud Drive

**Recommendation: keep the project outside iCloud Drive**, for example at `~/Code/veto`.
In a normal local folder, `uv run` works with no extra steps, Git is safe, and your private
message data under `data/` is not synced to iCloud. (Note that `~/Cloud` and `~/iCloud` can be
shortcuts into iCloud Drive; don't use those either.)

If the repository does live in iCloud Drive (for example
`~/Library/Mobile Documents/com~apple~CloudDocs/...`), the editable install of the project
breaks. Symptom:

```text
ModuleNotFoundError: No module named 'veto'
```

Cause: the editable install creates a `.pth` file in the virtualenv that points Python at `src/`.
iCloud marks files in the synced folder as *hidden*, and Python ignores hidden `.pth` files, so
the `veto` package is never found. Clearing the flag (`chflags nohidden ...`) does not last,
because iCloud re-applies it.

Workarounds if you cannot move the project:

**1. Point Python at `src/` explicitly** (no setup; use per command):

```bash
PYTHONPATH=src python scripts/extract_raw.py --overwrite
```

**2. Keep iCloud away from the venv** with a `.nosync` folder. iCloud skips anything ending in
`.nosync`:

```bash
deactivate
mv .venv .venv.nosync
ln -s .venv.nosync .venv
chflags nohidden .venv.nosync/lib/python3.12/site-packages/*.pth
source .venv/bin/activate
```

**3. Put uv's environment elsewhere**, by adding this to `~/.zshrc`:

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/veto"
```

Moving the project is simpler than any of these. Copy it (for example with
`rsync -a --exclude='.venv' <old path>/veto ~/Code/`), check it with `uv run pytest`, and only
then delete the iCloud copy. The old `.venv` cannot be moved (it contains absolute paths); `uv`
recreates it.

Without moving, the private data under `data/` is also inside iCloud Drive and will sync to your
iCloud account. That is separate from Git and does not expose it publicly, but keep it in mind if
iCloud sharing or other devices are a concern.

## After extraction

```bash
uv run veto corpus build
```

The label annotations are keyed by `message_id`. A larger or differently filtered raw file
contains messages with no labels yet, so label them before running `veto corpus inspect` or
`veto train`.
