import pandas as pd
import pytest

from veto.corpus import CORPUS_COLUMNS, ValidationError, build_corpus, build_corpus_from_files


def test_schema_columns_and_order(raw):
    corpus, _ = build_corpus(raw)
    assert list(corpus.columns) == list(CORPUS_COLUMNS)


def test_original_text_preserved_and_normalized(raw):
    corpus, _ = build_corpus(raw)
    row = corpus.set_index("message_id").loc[1]
    assert row["text"] == "  Vote for Jane Doe 🗳️ now!!  "
    assert row["normalized_text"] == "Vote for Jane Doe 🗳️ now!!"


def test_removes_null_and_empty_and_reports(raw):
    corpus, report = build_corpus(raw)
    assert sorted(corpus["message_id"]) == [1, 3]
    assert (report.loaded, report.removed_null_text, report.removed_empty_text) == (4, 1, 1)
    assert report.written == 2


def test_raw_not_mutated(raw):
    before = raw.copy(deep=True)
    build_corpus(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_received_at_utc(raw):
    corpus, _ = build_corpus(raw)
    assert str(corpus["received_at"].dt.tz) == "UTC"


def test_group_chat_derivation(raw):
    corpus, report = build_corpus(raw)
    flags = corpus.set_index("message_id")["is_group_chat"]
    assert flags[1] is True or flags[1] == True  # noqa: E712
    assert flags[3] == False  # noqa: E712
    assert report.group_chat_messages == 1


def test_unknown_chat_is_null():
    raw = pd.DataFrame(
        {
            "message_id": [1],
            "text": ["hi"],
            "is_from_me": [0],
            "date": pd.to_datetime(["2026-01-01"]).tz_localize("UTC"),
            "sender": ["x"],
            "chat_id": [None],
            "chat_identifier": [None],
            "chat_name": [None],
        }
    )
    corpus, report = build_corpus(raw)
    assert corpus["is_group_chat"].isna().all()
    assert report.unknown_chat_messages == 1


def test_duplicates_are_kept_and_hashed(raw):
    dup = pd.concat([raw, raw.assign(message_id=[13, 11, 12, 14])], ignore_index=True)
    corpus, report = build_corpus(dup)
    assert len(corpus) == 4
    assert report.exact_duplicate_texts == 4
    assert corpus["text_hash"].nunique() == 2


def test_missing_column_fails(raw):
    with pytest.raises(ValidationError, match="missing required columns"):
        build_corpus(raw.drop(columns=["sender"]))


def test_missing_and_duplicate_ids_fail(raw):
    with pytest.raises(ValidationError, match="missing message_id"):
        build_corpus(raw.assign(message_id=[1, 2, None, 4]))
    with pytest.raises(ValidationError, match="duplicate message_id"):
        build_corpus(raw.assign(message_id=[1, 1, 2, 3]))


def test_naive_dates_fail_clearly(raw):
    raw = raw.assign(date=raw["date"].dt.tz_localize(None))
    with pytest.raises(ValidationError, match="invalid date"):
        build_corpus(raw)


def test_error_messages_do_not_leak_text(raw):
    with pytest.raises(ValidationError) as exc:
        build_corpus(raw.assign(message_id=[1, 1, 2, 3]))
    assert "Vote for Jane" not in str(exc.value)


def test_reproducible(raw):
    a, _ = build_corpus(raw)
    b, _ = build_corpus(raw.sample(frac=1, random_state=0))
    pd.testing.assert_frame_equal(a, b)


def test_file_roundtrip_and_raw_untouched(raw, tmp_path):
    raw_path, out = tmp_path / "raw.parquet", tmp_path / "corpus" / "c.parquet"
    raw.to_parquet(raw_path, index=False)
    before = raw_path.read_bytes()
    build_corpus_from_files(raw_path, out)
    assert raw_path.read_bytes() == before
    assert len(pd.read_parquet(out)) == 2
    first = out.read_bytes()
    build_corpus_from_files(raw_path, out)
    assert out.read_bytes() == first


def test_output_cannot_overwrite_raw(raw, tmp_path):
    p = tmp_path / "raw.parquet"
    raw.to_parquet(p, index=False)
    with pytest.raises(ValueError, match="differ"):
        build_corpus_from_files(p, p)
