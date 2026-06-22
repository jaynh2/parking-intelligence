"""
Tests for the model promotion gate and training lineage collector.

These are pure unit tests — no artifact fixtures needed since both
functions are stateless (lineage just reads the environment, and the
gate just does arithmetic on the MAE values it's given).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from config.settings import Settings
from pipeline.train_pipeline import _check_promotion_gate, _collect_lineage


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def _prev(mae: float) -> dict:
    return {"forecast_mae": mae, "run_id": "prev_run"}


# ------------------------------------------------------------------ #
# Promotion gate
# ------------------------------------------------------------------ #
class TestPromotionGate:
    def test_first_run_always_promotes(self):
        s = Settings(require_promotion_gate=True, max_mae_regression_pct=10.0)
        ok, reason = _check_promotion_gate(new_mae=5.0, prev_metrics=None, settings=s)
        assert ok is True
        assert "first run" in reason.lower()

    def test_improved_mae_promotes(self):
        s = Settings(require_promotion_gate=True, max_mae_regression_pct=10.0)
        ok, reason = _check_promotion_gate(new_mae=1.0, prev_metrics=_prev(1.5), settings=s)
        assert ok is True
        assert "better" in reason.lower()

    def test_within_tolerance_promotes(self):
        # 9% regression is within 10% tolerance
        s = Settings(require_promotion_gate=True, max_mae_regression_pct=10.0)
        ok, _ = _check_promotion_gate(new_mae=1.09, prev_metrics=_prev(1.0), settings=s)
        assert ok is True

    def test_exceeds_tolerance_rejected(self):
        # 15% regression exceeds 10% tolerance
        s = Settings(require_promotion_gate=True, max_mae_regression_pct=10.0)
        ok, reason = _check_promotion_gate(new_mae=1.15, prev_metrics=_prev(1.0), settings=s)
        assert ok is False
        assert "REJECTED" in reason
        assert "1.1500" in reason  # new MAE in the message

    def test_exactly_at_ceiling_promotes(self):
        # new_mae == prev_mae * 1.1 is exactly at the ceiling — should pass
        s = Settings(require_promotion_gate=True, max_mae_regression_pct=10.0)
        ok, _ = _check_promotion_gate(new_mae=1.10, prev_metrics=_prev(1.0), settings=s)
        assert ok is True

    def test_just_above_ceiling_rejected(self):
        s = Settings(require_promotion_gate=True, max_mae_regression_pct=10.0)
        ok, _ = _check_promotion_gate(new_mae=1.101, prev_metrics=_prev(1.0), settings=s)
        assert ok is False

    def test_force_promotes_despite_regression(self):
        s = Settings(require_promotion_gate=True, max_mae_regression_pct=10.0)
        ok, reason = _check_promotion_gate(new_mae=99.0, prev_metrics=_prev(1.0), settings=s, force=True)
        assert ok is True
        assert "force" in reason.lower()

    def test_gate_disabled_always_promotes(self):
        s = Settings(require_promotion_gate=False)
        ok, reason = _check_promotion_gate(new_mae=99.0, prev_metrics=_prev(1.0), settings=s)
        assert ok is True
        assert "disabled" in reason.lower()

    def test_prev_metrics_missing_mae_key_promotes(self):
        # A previous run that predates the forecast MAE field shouldn't block
        s = Settings(require_promotion_gate=True, max_mae_regression_pct=10.0)
        ok, reason = _check_promotion_gate(new_mae=5.0, prev_metrics={"run_id": "old"}, settings=s)
        assert ok is True

    def test_custom_threshold_respected(self):
        # Strict 1% threshold
        s = Settings(require_promotion_gate=True, max_mae_regression_pct=1.0)
        ok_tight, _ = _check_promotion_gate(new_mae=1.02, prev_metrics=_prev(1.0), settings=s)
        assert ok_tight is False

        # Loose 50% threshold
        s2 = Settings(require_promotion_gate=True, max_mae_regression_pct=50.0)
        ok_loose, _ = _check_promotion_gate(new_mae=1.49, prev_metrics=_prev(1.0), settings=s2)
        assert ok_loose is True


# ------------------------------------------------------------------ #
# Training lineage
# ------------------------------------------------------------------ #
class TestLineage:
    def test_contains_required_keys(self, tmp_path):
        lineage = _collect_lineage(None)
        assert "python_version" in lineage
        assert "library_versions" in lineage
        assert "git_commit" in lineage

    def test_csv_hash_computed_when_file_exists(self, tmp_path):
        csv = tmp_path / "test.csv"
        csv.write_bytes(b"id,lat,lon\n1,12.9,77.5\n")
        lineage = _collect_lineage(csv)
        assert "input_csv_sha256" in lineage
        assert len(lineage["input_csv_sha256"]) == 64  # SHA-256 hex

    def test_different_file_contents_produce_different_hash(self, tmp_path):
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        f1.write_bytes(b"content-a")
        f2.write_bytes(b"content-b")
        assert _collect_lineage(f1)["input_csv_sha256"] != _collect_lineage(f2)["input_csv_sha256"]

    def test_missing_csv_returns_unavailable(self):
        lineage = _collect_lineage(Path("/this/does/not/exist.csv"))
        assert lineage["input_csv_sha256"] == "unavailable"

    def test_library_versions_include_pandas(self, tmp_path):
        lineage = _collect_lineage(None)
        assert "pandas" in lineage["library_versions"]
        assert lineage["library_versions"]["pandas"] != "unknown"

    def test_git_commit_graceful_on_non_repo(self, tmp_path):
        # Simulate git not being available
        with patch("subprocess.check_output", side_effect=Exception("no git")):
            lineage = _collect_lineage(None)
        assert "git_commit" in lineage
        assert lineage["git_commit"] == "unknown"


# ------------------------------------------------------------------ #
# Settings: business rules YAML loading
# ------------------------------------------------------------------ #
class TestBusinessRulesYaml:
    def test_yaml_overrides_severity_mapping(self, tmp_path):
        rules_yaml = tmp_path / "rules.yaml"
        rules_yaml.write_text("severity_mapping:\n  TEST_OFFENSE: 9.9\n")
        s = Settings(business_rules_path=rules_yaml)
        assert s.severity_mapping.get("TEST_OFFENSE") == 9.9

    def test_yaml_overrides_vehicle_weight(self, tmp_path):
        rules_yaml = tmp_path / "rules.yaml"
        rules_yaml.write_text("vehicle_weight_mapping:\n  SPACESHIP: 10\n")
        s = Settings(business_rules_path=rules_yaml)
        assert s.vehicle_weight_mapping.get("SPACESHIP") == 10

    def test_missing_yaml_uses_python_defaults(self, tmp_path):
        s = Settings(business_rules_path=tmp_path / "nonexistent.yaml")
        # Default includes "CAR"
        assert "CAR" in s.vehicle_weight_mapping

    def test_malformed_yaml_falls_back_gracefully(self, tmp_path):
        rules_yaml = tmp_path / "bad.yaml"
        rules_yaml.write_text(": : this is not valid yaml : :")
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s = Settings(business_rules_path=rules_yaml)
        assert any("Could not load business rules" in str(warning.message) for warning in w)
        # Should still have defaults
        assert "CAR" in s.vehicle_weight_mapping
