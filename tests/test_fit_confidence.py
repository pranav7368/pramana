"""Workflow guards; these invented fixtures are not evaluation evidence."""
from copy import deepcopy

import pytest
from scripts.fit_confidence import fit_reviews


def fixture_rows(split):
    return [{"qid": f"{split}-{i}", "parallel_id": f"parallel-{split}-{i}",
             "question": f"fixture {split} policy {i}", "language": "en", "split": split,
             "human_reviewed": True, "reviewer": "fixture-only", "fully_supported": bool(i % 2),
             "features": {"s2_supported_ratio": float(i % 2)}} for i in range(20)]


def test_fit_workflow_uses_separate_dev_calibration_and_reports_no_test_evidence():
    model, report = fit_reviews(fixture_rows("train"), fixture_rows("dev"), "en")
    assert model.feature_names == ("s2_supported_ratio",)
    assert report["heldout_test_evaluated"] is False
    assert 0 <= report["dev_ece"] <= 1


@pytest.mark.parametrize("field,value", [
    ("human_reviewed", False), ("reviewer", ""), ("split", "test"),
    ("fully_supported", 1), ("language", "ta"), ("question", ""),
    ("features", {"s2_supported_ratio": float("nan")}),
])
def test_fitting_refuses_invalid_or_nonhuman_labels(field, value):
    train, dev = fixture_rows("train"), fixture_rows("dev")
    dev[0][field] = value
    with pytest.raises(ValueError):
        fit_reviews(train, dev, "en")


@pytest.mark.parametrize("field", ["qid", "parallel_id", "question"])
def test_train_dev_leakage_is_refused(field):
    train, dev = fixture_rows("train"), deepcopy(fixture_rows("dev"))
    dev[0][field] = train[0][field]
    with pytest.raises(ValueError, match="overlapping"):
        fit_reviews(train, dev, "en")
