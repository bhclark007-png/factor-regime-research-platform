from __future__ import annotations

import dashboard


def test_dashboard_can_load_latest_run_result(
    monkeypatch, fixture_run_result_path
) -> None:
    monkeypatch.setattr(dashboard, "LATEST_JSON", fixture_run_result_path)
    result = dashboard.load_result()

    assert result["schema_version"] == "0.6.1"
    assert result["run_id"] == "fixture"
