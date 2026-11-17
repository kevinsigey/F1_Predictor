"""grid_layer.py -- a 2-parameter calibration layer on the race ranker's scores.

    z_i = T * score_i - a * log(grid_i)        theta_i = exp(z_i - max z)

Win and podium come from a Plackett-Luce draw over the layered theta; top-10
(p_points_finish) stays on the raw theta, because the layer hurts it (see
docs/f1_grid_layer.md). T and a are fitted ONLY on out-of-fold scores of
earlier seasons (each season scored by a ranker trained on seasons before it),
by minimising multiclass win log-loss. Needs a real grid, so it is applied at
the post_qualifying tier only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import race_outcome

MAX_GRID = 22  # largest field on the grid in the training window


def log_grid(grid) -> np.ndarray:
    """log of the starting slot. Unknown or pit-lane (0) -> back of an ordinary field."""
    g = pd.Series(grid, dtype=float)
    return np.log(g.where(g > 0, 20.0).fillna(20.0).clip(1, MAX_GRID).to_numpy())


def layered_scores(scores: np.ndarray, grid, temperature: float, grid_coef: float) -> np.ndarray:
    return temperature * np.asarray(scores, dtype=float) - grid_coef * log_grid(grid)


def _win_nll(par, races) -> float:
    t, a = par
    tot = 0.0
    for s, lg, w in races:
        z = t * s - a * lg
        z = z - z.max()
        tot -= (z - np.log(np.exp(z).sum()))[w]
    return tot / len(races)


def fit(oof: pd.DataFrame) -> tuple[float, float]:
    """(T, a) from out-of-fold rows with columns season, round, score, grid, position."""
    races = []
    for _, g in oof.groupby(["season", "round"]):
        winner = np.flatnonzero((g["position"] == 1).to_numpy())
        if len(winner) == 1:
            races.append((g["score"].to_numpy(float), log_grid(g["grid"]), int(winner[0])))
    if not races:
        raise ValueError("no races with exactly one winner to fit the grid layer on")
    res = minimize(_win_nll, x0=[1.0, 0.0], args=(races,), method="Nelder-Mead")
    return float(res.x[0]), float(res.x[1])


def oof_scores(post_quali: pd.DataFrame, feature_cols: list[str], first: int, last: int,
               hyperparams: dict | None = None) -> pd.DataFrame:
    """Score every season in [first, last] with a ranker trained on the seasons before it."""
    parts = []
    for season in range(first, last + 1):
        tr, va = post_quali[post_quali["season"] < season], post_quali[post_quali["season"] == season]
        if tr.empty or va.empty:
            continue
        ranker = race_outcome.train_ranker(tr, feature_cols, hyperparams=hyperparams)
        out = va[["season", "round", "driver_id", "grid", "position"]].copy()
        out["score"] = ranker.predict(va[feature_cols])
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def fit_block(post_quali: pd.DataFrame, feature_cols: list[str], hyperparams: dict | None, enabled: bool) -> dict:
    """The manifest entry. Fitted on every available out-of-fold season."""
    seasons = sorted(post_quali["season"].unique())
    if len(seasons) < 2:  # nothing out-of-fold to fit on
        return {"enabled": False, "temperature": None, "grid_coef": None, "tier": "post_qualifying", "fit_seasons": [], "fit_races": 0}
    oof = oof_scores(post_quali, feature_cols, seasons[1], seasons[-1], hyperparams)
    t, a = fit(oof)
    return {
        "enabled": bool(enabled),
        "temperature": t,
        "grid_coef": a,
        "tier": "post_qualifying",
        "fit_seasons": [int(oof["season"].min()), int(oof["season"].max())],
        "fit_races": int(oof.groupby(["season", "round"]).ngroups),
    }


def simulate(scores: dict[str, float], grid: dict[str, float], block: dict | None,
             dnf_prob: dict[str, float] | None = None, **sim_kw) -> pd.DataFrame:
    """simulate_race with the layer on win/podium when `block` is enabled.

    Raw win/podium are kept beside the corrected ones as raw_p_win / raw_p_podium.
    Corrected win is renormalised to sum to 1 across the race.
    """
    raw = race_outcome.simulate_race(race_outcome.theta_from_xgb_scores(scores), dnf_prob, **sim_kw)
    if not block or not block.get("enabled"):
        return raw
    ids = list(scores)
    z = layered_scores(np.array([scores[d] for d in ids]), [grid[d] for d in ids],
                       block["temperature"], block["grid_coef"])
    theta = race_outcome.theta_from_xgb_scores(dict(zip(ids, z)))
    cal = race_outcome.simulate_race(theta, dnf_prob, **sim_kw).set_index("driver_id")
    out = raw.rename(columns={"p_win": "raw_p_win", "p_podium": "raw_p_podium"})
    out["p_win"] = out["driver_id"].map(cal["p_win"])
    out["p_win"] = out["p_win"] / out["p_win"].sum()
    out["p_podium"] = out["driver_id"].map(cal["p_podium"])
    return out.sort_values("p_win", ascending=False).reset_index(drop=True)
