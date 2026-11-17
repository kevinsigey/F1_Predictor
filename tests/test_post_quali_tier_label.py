"""A race prediction is labelled post_qualifying only when it has the grid.

Measured on the shipped `data/tracking.db`: for 2026 rounds 15-20, every
`post_qualifying` race row has the same `predicted_prob` as its `post_practice`
row (552 of 552 pairs). Four of those post-qualifying snapshots were taken
10-51 minutes after qualifying started, before results were published, so the
frame had no grid. The tier was still labelled from the clock.
"""

import pandas as pd
import pytest

from f1_predictor.api import routes
from f1_predictor.data import jolpica
from f1_predictor.features import session_state
from f1_predictor.models import championship_projection

DRIVERS = ["norris", "leclerc"]


@pytest.fixture
def race_row():
    # Qualifying started long ago, so the clock says post_qualifying.
    return pd.Series({
        "round": 15, "qualifying_datetime": pd.Timestamp("2026-01-01T12:00:00Z"),
        "fp1_datetime": None, "fp2_datetime": None, "fp3_datetime": None,
        "sprint_quali_datetime": None,
    })


@pytest.fixture
def mocked_history(monkeypatch):
    schedule = pd.DataFrame({"round": [15]})
    base = pd.DataFrame({
        "driver_id": DRIVERS, "constructor_id": ["mclaren", "ferrari"],
        "grid": [float("nan")] * 2, "quali_position": [float("nan")] * 2,
        "quali_gap_to_pole": [float("nan")] * 2,
    })
    monkeypatch.setattr(jolpica, "fetch_season_schedule", lambda season, force_refresh=False: schedule)
    monkeypatch.setattr(jolpica, "load_history", lambda season: (pd.DataFrame(), pd.DataFrame(), None))
    monkeypatch.setattr(jolpica, "fetch_driver_standings", lambda season, *a, **k: pd.DataFrame({"driver_id": DRIVERS}))
    monkeypatch.setattr(
        championship_projection, "build_future_feature_rows",
        lambda *args, **kwargs: base.copy(),
    )


def _quali(positions):
    return pd.DataFrame({
        "season": 2026, "round": 15, "driver_id": DRIVERS,
        "constructor_id": ["mclaren", "ferrari"], "quali_position": positions,
        "q1": [None, None], "q2": [None, None], "q3": [None, None],
    })


def test_without_published_qualifying_the_frame_is_not_labelled_post_qualifying(monkeypatch, race_row, mocked_history):
    monkeypatch.setattr(jolpica, "fetch_qualifying", lambda season, round_, force_refresh=False: pd.DataFrame())

    frame, tier, _ = routes._future_feature_frame(2026, 15, race_row)

    assert tier == session_state.TIER_POST_PRACTICE
    assert frame["grid"].isna().all(), "no grid was merged, so none may be claimed"


def test_with_published_qualifying_the_frame_is_post_qualifying_and_carries_the_grid(monkeypatch, race_row, mocked_history):
    monkeypatch.setattr(jolpica, "fetch_qualifying", lambda season, round_, force_refresh=False: _quali([1, 2]))

    frame, tier, _ = routes._future_feature_frame(2026, 15, race_row)

    assert tier == session_state.TIER_POST_QUALIFYING
    assert frame.set_index("driver_id")["grid"].to_dict() == {"norris": 1.0, "leclerc": 2.0}
