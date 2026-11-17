"""Guards for models/grid_layer.py and its serving wiring. No network."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import grid_layer_eval  # noqa: E402

from f1_predictor.api import routes  # noqa: E402
from f1_predictor.features import session_state  # noqa: E402
from f1_predictor.models import grid_layer, race_outcome  # noqa: E402

BLOCK = {"enabled": True, "temperature": 0.6, "grid_coef": 1.45}
SCORES = {"a": 1.0, "b": 0.8, "c": 0.2, "d": -0.3, "e": -1.0}
GRID = {"a": 5, "b": 1, "c": 2, "d": 3, "e": 4}
KW = dict(n_trials=4000, seed=3)


def test_exact_plackett_luce_matches_the_simulator():
    theta = np.exp(np.array(list(SCORES.values())))
    w, p = grid_layer_eval.pl_win_podium(theta)
    sim = race_outcome.simulate_race(dict(zip(SCORES, theta)), n_trials=20000, seed=1).set_index("driver_id")
    assert w.sum() == pytest.approx(1) and p.sum() == pytest.approx(3)
    assert np.abs(sim.loc[list(SCORES), "p_win"].to_numpy() - w).max() < 0.02
    assert np.abs(sim.loc[list(SCORES), "p_podium"].to_numpy() - p).max() < 0.02


def test_win_sums_to_one_and_top10_is_the_raw_one():
    raw = race_outcome.simulate_race(race_outcome.theta_from_xgb_scores(SCORES), **KW).set_index("driver_id")
    out = grid_layer.simulate(SCORES, GRID, BLOCK, **KW).set_index("driver_id")
    assert out["p_win"].sum() == pytest.approx(1.0)
    assert out["p_podium"].sum() == pytest.approx(3.0, abs=1e-9)
    pd.testing.assert_series_equal(out["p_points_finish"], raw.loc[out.index, "p_points_finish"])
    pd.testing.assert_series_equal(out["raw_p_win"], raw.loc[out.index, "p_win"], check_names=False)
    # the layer moves win mass to the pole sitter and away from the P5 starter
    assert out.loc["b", "p_win"] > out.loc["b", "raw_p_win"]
    assert out.loc["a", "p_win"] < out.loc["a", "raw_p_win"]


def test_disabled_or_missing_block_is_the_plain_simulation():
    raw = race_outcome.simulate_race(race_outcome.theta_from_xgb_scores(SCORES), **KW)
    for block in (None, {**BLOCK, "enabled": False}):
        pd.testing.assert_frame_equal(grid_layer.simulate(SCORES, GRID, block, **KW), raw)


def test_fit_puts_weight_on_grid_when_the_grid_decides_the_race():
    rng = np.random.default_rng(0)
    rows = []
    for r in range(40):
        grid = rng.permutation(10) + 1
        winner = np.argmin(grid + rng.normal(0, 0.7, 10))
        rows.append(pd.DataFrame({"season": 2022, "round": r, "driver_id": range(10), "grid": grid,
                                  "score": rng.normal(size=10), "position": np.where(np.arange(10) == winner, 1, 2)}))
    t, a = grid_layer.fit(pd.concat(rows))
    assert a > 0.5 and abs(t) < 0.5


def test_pit_lane_and_unknown_grid_are_the_back_of_the_field():
    assert grid_layer.log_grid([0, np.nan, 20]).tolist() == pytest.approx([np.log(20)] * 3)


@pytest.mark.parametrize("tier,expect_layer", [(session_state.TIER_POST_QUALIFYING, True),
                                               (session_state.TIER_POST_PRACTICE, False),
                                               (session_state.TIER_PRE_WEEKEND, False)])
def test_serving_applies_the_layer_at_post_qualifying_only(monkeypatch, tier, expect_layer):
    df = pd.DataFrame({"driver_id": list(SCORES), "constructor_id": "x", "grid": list(GRID.values())})
    seen = {}

    def spy(scores, grid, layer, dnf_prob=None, **kw):
        seen["layer"] = layer
        return race_outcome.simulate_race(race_outcome.theta_from_xgb_scores(scores), n_trials=50, seed=0)

    monkeypatch.setattr(routes, "_future_feature_frame", lambda *a: (df, tier, None))
    monkeypatch.setattr(routes, "_load_models", lambda: ("xgb_ranker", object(), object()))
    monkeypatch.setattr(routes, "_require_manifest", lambda: {"grid_layer": BLOCK})
    monkeypatch.setattr(race_outcome, "xgb_scores_for_race", lambda *a: dict(SCORES))
    monkeypatch.setattr(routes.dnf_model, "predict_dnf_prob", lambda *a, **k: {})
    monkeypatch.setattr(routes.grid_layer, "simulate", spy)

    routes._predict_upcoming_race(2026, 1, pd.Series(dtype=object))

    assert (seen["layer"] == BLOCK) if expect_layer else (seen["layer"] is None)


def test_fit_block_is_disabled_when_there_is_no_out_of_fold_season():
    one_season = pd.DataFrame({"season": [2024], "round": [1], "driver_id": ["a"], "grid": [1], "position": [1]})
    assert grid_layer.fit_block(one_season, [], None, enabled=True)["enabled"] is False
