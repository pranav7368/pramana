"""Synthetic exporter fixtures test workflow integrity, not empirical accuracy."""
import hashlib
import json

import pytest
from scripts import prepare_review_packet as exporter
from scripts.pilot_acceptance import assess
from scripts.validate_operations import ControlledProvider, percentile, validation_app

from pramana.generation.base import GenerationRequest, Message, ProviderError
from pramana.ingestion.corpus import load_corpus


def toy_report(monkeypatch, tmp_path):
    monkeypatch.setattr(exporter, "ROOT", tmp_path)
    directory = tmp_path / "examples/corpus/en"
    directory.mkdir(parents=True)
    policy = directory / "policy.md"
    policy.write_text("Appeals are allowed within 60 days.", encoding="utf-8")
    identifier = load_corpus(tmp_path / "examples/corpus", ("en",))["en"][0].chunk_id
    report = {"complete": True, "measurement": "synthetic_live_integration", "languages": ["en"],
              "corpus_sha256": {"en/policy.md": hashlib.sha256(policy.read_bytes()).hexdigest()},
              "results": [{"case": "answer", "status": 200, "language": "en", "elapsed_seconds": 1,
                           "request": {"query": "What is the appeal deadline?"},
                           "response": {"answer": "60 days", "abstained": False, "claims": [],
                                        "retrieved_chunk_ids": [identifier],
                                        "confidence": {"score": 0.9, "calibrator": "none", "signals": {}}}}]}
    path = tmp_path / "fixture-report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path, policy


def test_review_export_does_not_invent_labels_or_make_smoke_heldout(monkeypatch, tmp_path):
    path, _ = toy_report(monkeypatch, tmp_path)
    result = exporter.prepare(path, tmp_path / "packet")
    row = json.loads((tmp_path / "packet/smoke-reviews.jsonl").read_text())
    assert result["human_labels_filled"] == 0
    assert row["split"] == "smoke" and row["human_reviewed"] is False
    assert row["fully_supported"] is None and row["should_abstain"] is None
    assert row["baseline_latency_ms"] is None and row["confidence_calibrated"] is False
    assert row["retrieved_evidence"][0]["text"] == "Appeals are allowed within 60 days."
    with pytest.raises(ValueError, match="held-out"):
        assess([row], ("en",))
    with pytest.raises(ValueError, match="fresh"):
        exporter.prepare(path, tmp_path / "packet")


def test_changed_corpus_cannot_be_attached_to_old_outputs(monkeypatch, tmp_path):
    path, policy = toy_report(monkeypatch, tmp_path)
    policy.write_text("Appeals take 90 days.", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        exporter.prepare(path, tmp_path / "packet")
    assert not (tmp_path / "packet").exists()


def test_controlled_server_cannot_be_enabled_by_normal_launcher(monkeypatch):
    monkeypatch.delenv("PRAMANA_VALIDATION_ONLY", raising=False)
    with pytest.raises(RuntimeError, match="isolated"):
        validation_app()


def test_controlled_provider_releases_tracking_after_simulated_failure(monkeypatch):
    monkeypatch.setattr("scripts.validate_operations.time.sleep", lambda _: None)
    provider = ControlledProvider()
    with pytest.raises(ProviderError):
        provider.generate(GenerationRequest((Message("user", "OPERATIONS_FAILURE_SENTINEL"),), ""))
    assert provider.metrics() == {"calls": 1, "active": 0, "peak_active": 1}


def test_nearest_rank_load_percentiles():
    assert percentile([], 0.95) is None
    assert percentile([4, 2, 3, 1], 0.5) == 2
    assert percentile([4, 2, 3, 1], 0.95) == 4
