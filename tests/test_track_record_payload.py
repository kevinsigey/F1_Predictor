"""The track-record payload stays small on the wire, and a repeat read is cached.

Fixture: a synthetic database with 20 drivers across 45 race sessions, which is
3,600 recorded rows and is in the same range as the shipped `data/tracking.db`
(3,572 rows, 1,193,461 bytes uncompressed). Measured before this change:
the default `/api/track-record` body was 1,193,461 bytes and took ~0.3 s per
call, recomputed on every request.

Budgets, on that fixture:
  * the frontend's own first-load call (`include_picks=false`) is under 8 KB
    uncompressed. Measured: 6,146 bytes here (45 rounds in `by_round`);
    1,114 bytes on the wire for the shipped DB;
  * the default call, gzipped on the wire (`Content-Length` under gzip), is
    under 40 KB. Measured after the change: ~29 KB.
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from f1_predictor.api.main import app
from f1_predictor.tracking import store

DRIVERS = [f"driver_{i:02d}" for i in range(20)]
ROUNDS = 45


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Never the shipped `data/tracking.db`."""
    monkeypatch.setattr(store, "TRACKING_DB_PATH", tmp_path / "tracking.db")
    store._TRACK_RECORD_CACHE.clear()


def _sim(seed: int) -> pd.DataFrame:
    rows = []
    for i, driver in enumerate(DRIVERS):
        p = max(0.01, 0.5 - 0.02 * i + 0.001 * seed)
        rows.append({
            "driver_id": driver, "constructor_id": f"team_{i // 2}", "p_win": p,
            "p_podium": min(0.95, 3 * p), "p_points_finish": min(0.98, 5 * p), "p_dnf": 0.05,
        })
    return pd.DataFrame(rows)


def _results(round_: int) -> pd.DataFrame:
    return pd.DataFrame([
        {"season": 2026, "round": round_, "driver_id": d, "position": i + 1, "dnf": i == 19}
        for i, d in enumerate(DRIVERS)
    ])


def _seed_database() -> None:
    for round_ in range(1, ROUNDS + 1):
        store.record_session_predictions(
            _sim(round_), 2026, round_, f"GP {round_}", "race", "post_qualifying",
            f"2026-03-{(round_ % 28) + 1:02d}T13:00:00+00:00",
        )
        store.reconcile_session_predictions(_results(round_), "race")


def test_the_fixture_is_realistic_in_size():
    _seed_database()
    picks = store.get_session_track_record(session_type="race")["per_pick"]
    assert len(picks) >= 3500


def test_the_frontend_first_load_payload_is_under_budget():
    _seed_database()
    response = TestClient(app).get("/api/track-record?session_type=race&include_picks=false")
    assert response.status_code == 200
    assert len(response.content) < 8_000
    assert response.json()["per_pick"] == []


def test_the_default_payload_is_under_budget_on_the_wire():
    _seed_database()
    response = TestClient(app).get("/api/track-record", headers={"Accept-Encoding": "gzip"})
    assert response.status_code == 200
    assert response.headers.get("content-encoding") == "gzip"
    assert int(response.headers["content-length"]) < 40_000
    assert len(response.json()["per_pick"]) >= 3500, "the default contract still carries per_pick"


def test_a_repeat_read_of_an_unchanged_database_is_served_from_cache(monkeypatch):
    _seed_database()
    calls = {"n": 0}
    real = store._compute_session_track_record

    def counting(**kwargs):
        calls["n"] += 1
        return real(**kwargs)

    monkeypatch.setattr(store, "_compute_session_track_record", counting)
    client = TestClient(app)
    first = client.get("/api/track-record?session_type=race&include_picks=false").json()
    second = client.get("/api/track-record?session_type=race&include_picks=false").json()
    assert calls["n"] == 1
    assert first == second


def test_a_write_invalidates_the_cache():
    _seed_database()
    before = store.get_session_track_record(session_type="race")["n_resolved"]
    store.record_session_predictions(
        _sim(99), 2026, ROUNDS + 1, "GP new", "race", "post_qualifying", "2026-12-01T13:00:00+00:00",
    )
    store.reconcile_session_predictions(_results(ROUNDS + 1), "race")
    after = store.get_session_track_record(session_type="race")["n_resolved"]
    assert after == before + 80, "a recorded session must show up on the next read, not from cache"
