import pandas as pd

from veto.evaluation.errors import analyze_errors
from veto.evaluation.metrics import compute_metrics
from veto.evaluation.threshold import select_threshold


def test_metrics_confusion_and_values():
    m = compute_metrics([1, 1, 0, 0], [0.9, 0.2, 0.6, 0.1], 0.5)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["precision"] == 0.5 and m["recall"] == 0.5
    assert m["roc_auc"] is not None


def test_metrics_single_class_has_no_auc():
    m = compute_metrics([0, 0], [0.1, 0.2], 0.5)
    assert m["roc_auc"] is None and m["pr_auc"] is None


def test_threshold_prefers_precision_target():
    y = [1, 1, 1, 0, 0, 0]
    p = [0.95, 0.9, 0.5, 0.6, 0.2, 0.1]
    sel = select_threshold(y, p, target_precision=0.95)
    assert sel["threshold"] == 0.9  # 0.5 would admit a false positive at 0.6


def test_threshold_falls_back_when_unreachable_or_one_class():
    assert select_threshold([1, 0], [0.4, 0.6], 0.95)["threshold"] == 0.5
    assert "default" in select_threshold([0, 0], [0.1, 0.2])["reason"]


def test_threshold_boundary_is_inclusive():
    m = compute_metrics([1], [0.5], 0.5)
    assert m["recall"] == 1.0


def test_error_analysis_flags():
    df = pd.DataFrame({"message_id": [1, 2], "label": [1, 0], "subtype": ["campaign", "personal"],
                       "normalized_text": ["Vote early https://a.co/x", "lol ok"]})
    errs = analyze_errors(df, [0.1, 0.1], 0.5, "test")
    assert len(errs) == 1 and errs[0]["error_type"] == "false_negative"
    assert errs[0]["has_url"] and errs[0]["has_election_vocab"]
