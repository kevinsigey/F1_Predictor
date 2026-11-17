"""Walk-forward ship test for models/grid_layer.py. MEASUREMENT + one manifest write.

Folds are the repo's own (evaluate/walk_forward.prepare_folds, min 3 training seasons):
validate 2022..2026, each with the production ranker (manifest tuned params, post_qualifying
rows) trained on the seasons before it. The layer for fold v is fitted ONLY on out-of-fold
scores of seasons < v (season u scored by a ranker trained on < u, u >= 2020), then applied
to fold v. Nothing is tuned on the fold being scored.

Probabilities are exact Plackett-Luce (no Monte Carlo noise, no DNF -- the repo's walk-forward
also runs dnf_prob=None). Paired race-level bootstrap, 95% percentile CI.
Top-10 is not layered, so it is identical by construction and not re-measured.

    PYTHONPATH=src:scripts python scripts/grid_layer_eval.py [--write-manifest]
Writes docs/f1_grid_layer.json, docs/f1_grid_layer.md, docs/f1_grid_layer_rows.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from f1_predictor.data import jolpica
from f1_predictor.features import session_state
from f1_predictor.features.build import build_training_frame
from f1_predictor.models import grid_layer, manifest as manifest_module, race_outcome

DOCS = Path(__file__).resolve().parents[1] / "docs"
N_BOOT, SEED, EPS = 2000, 1, 1e-4
FIRST_FIT_SEASON, MIN_TRAIN = 2020, 3


def pl_win_podium(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact Plackett-Luce P(win), P(top 3) by enumerating the first three places."""
    th = np.asarray(theta, float)
    S = th.sum()
    p1 = th / S
    p2 = th * ((p1 / (S - th)).sum() - p1 / (S - th))
    A = np.outer(th / S, th / (S - th))
    np.fill_diagonal(A, 0.0)
    B = A / (S - th[:, None] - th[None, :])
    np.fill_diagonal(B, 0.0)
    p3 = th * (B.sum() - B.sum(1) - B.sum(0))
    return p1, p1 + p2 + p3


def _grid_lookup(train: pd.DataFrame):
    g = train.assign(gc=grid_layer.log_grid(train["grid"]).round(9))
    g = g.groupby("gc").apply(lambda d: pd.Series({"w": ((d.position == 1).sum() + .5) / (len(d) + 10),
                                                    "p": ((d.position <= 3).sum() + 1) / (len(d) + 10)}), include_groups=False)
    return g


def _loss_ll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _ece(p, y, k=10):
    p, y = np.asarray(p), np.asarray(y)
    b = np.clip((p * k).astype(int), 0, k - 1)
    return float(sum(abs(y[b == i].mean() - p[b == i].mean()) * (b == i).mean() for i in range(k) if (b == i).any()))


def _race_mean(R, loss):
    return pd.Series(loss).groupby([R["season"].to_numpy(), R["round"].to_numpy()]).mean()


def _race_win_nll(R, p):
    # -log P(actual winner), races with exactly one winner
    win = R["win"].to_numpy() == 1
    s = pd.Series(-np.log(np.clip(p[win], EPS, 1)), index=pd.MultiIndex.from_arrays([R["season"].to_numpy()[win], R["round"].to_numpy()[win]]))
    return s


def _boot(delta: np.ndarray, rng) -> tuple[float, float, float]:
    idx = rng.integers(0, len(delta), (N_BOOT, len(delta)))
    m = delta[idx].mean(1)
    return float(delta.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _metrics(R: pd.DataFrame, base: str, cand: str, rng) -> list[dict]:
    """improvement = base loss - candidate loss (positive = candidate better), race-level paired."""
    out = []
    y_w, y_p = R["win"].to_numpy(float), R["pod"].to_numpy(float)
    specs = {
        "win log-loss": (lambda c: _race_mean(R, _loss_ll(y_w, R[f"{c}_w"].to_numpy()))),
        "win multiclass NLL": (lambda c: _race_win_nll(R, R[f"{c}_w"].to_numpy())),
        "podium log-loss": (lambda c: _race_mean(R, _loss_ll(y_p, R[f"{c}_p"].to_numpy()))),
        "podium Brier": (lambda c: _race_mean(R, (y_p - R[f"{c}_p"].to_numpy()) ** 2)),
    }
    for name, f in specs.items():
        lb, lc = f(base), f(cand)
        d = (lb - lc).to_numpy()
        m, lo, hi = _boot(d, rng)
        out.append(dict(metric=name, base=float(lb.mean()), cand=float(lc.mean()), improvement=m, lo=lo, hi=hi))
    for name, col, y in [("win ECE", "w", y_w), ("podium ECE", "p", y_p)]:
        out.append(dict(metric=name, base=_ece(R[f"{base}_{col}"], y), cand=_ece(R[f"{cand}_{col}"], y)))
    return out


def run(write_manifest: bool) -> dict:
    seasons = jolpica.default_seasons()
    df, cols = build_training_frame(seasons=seasons)
    man = json.loads(manifest_module.MANIFEST_PATH.read_text())
    params = man.get("tuned_ranker_params")
    pq = df[df["tier"] == session_state.TIER_POST_QUALIFYING]

    # Out-of-fold scores: season u scored by a ranker trained on seasons < u, for every tier's rows.
    oof = {t: [] for t in session_state.TIER_ORDER}
    for u in seasons[1:]:
        tr = pq[pq["season"] < u]
        ranker = race_outcome.train_ranker(tr, cols, hyperparams=params)
        for t in session_state.TIER_ORDER:
            va = df[(df["season"] == u) & (df["tier"] == t)]
            o = va[["season", "round", "driver_id", "grid", "position"]].copy()
            o["score"] = ranker.predict(va[cols])
            oof[t].append(o)
    oof = {t: pd.concat(v, ignore_index=True) for t, v in oof.items()}

    rng = np.random.default_rng(SEED)
    result = {"folds": [], "tiers": {}}
    rows = []
    for tier in session_state.TIER_ORDER:
        tier_rows = []
        for v in seasons[MIN_TRAIN:]:
            inner = oof[tier][(oof[tier]["season"] >= FIRST_FIT_SEASON) & (oof[tier]["season"] < v)]
            T, a = grid_layer.fit(inner)
            lookup = _grid_lookup(pq[pq["season"] < v]) if tier == session_state.TIER_POST_QUALIFYING else None
            for (s, r), g in oof[tier][oof[tier]["season"] == v].groupby(["season", "round"]):
                sc = g["score"].to_numpy(float)
                raw_w, raw_p = pl_win_podium(np.exp(sc - sc.max()))
                z = grid_layer.layered_scores(sc, g["grid"], T, a)
                cal_w, cal_p = pl_win_podium(np.exp(z - z.max()))
                d = pd.DataFrame(dict(season=s, round=r, driver_id=g["driver_id"].to_numpy(), grid=g["grid"].to_numpy(),
                                      win=(g["position"] == 1).astype(int).to_numpy(), pod=(g["position"] <= 3).astype(int).to_numpy(),
                                      raw_w=raw_w, raw_p=raw_p, cal_w=cal_w, cal_p=cal_p, T=T, a=a))
                if lookup is not None:
                    key = grid_layer.log_grid(g["grid"]).round(9)
                    gw = lookup["w"].reindex(key).fillna(.5 / 10).to_numpy()  # unseen slot = empty Laplace cell
                    gp = lookup["p"].reindex(key).fillna(1 / 10).to_numpy()
                    d["grid_w"] = gw / gw.sum()
                    d["grid_p"] = np.clip(gp * 3 / gp.sum(), .001, .999)
                tier_rows.append(d)
        R = pd.concat(tier_rows, ignore_index=True)
        R.insert(0, "tier", tier)
        rows.append(R)
        block = {"n_races": int(R.groupby(["season", "round"]).ngroups), "vs_raw": _metrics(R, "raw", "cal", rng)}
        if "grid_w" in R:
            block["vs_grid_lookup"] = _metrics(R, "grid", "cal", rng)
            block["raw_vs_grid_lookup"] = _metrics(R, "grid", "raw", rng)
            block["per_season_win_ll"] = {
                int(s): float((_race_mean(g, _loss_ll(g["win"].to_numpy(float), g["raw_w"].to_numpy()))
                               - _race_mean(g, _loss_ll(g["win"].to_numpy(float), g["cal_w"].to_numpy()))).mean())
                for s, g in R.groupby("season")}
            block["per_season_params"] = {int(s): [float(g["T"].iloc[0]), float(g["a"].iloc[0])] for s, g in R.groupby("season")}
        result["tiers"][tier] = block

    pqm = {m["metric"]: m for m in result["tiers"][session_state.TIER_POST_QUALIFYING]["vs_raw"]}
    primary = pqm["win log-loss"]
    others_ok = all(m["improvement"] >= 0 for m in pqm.values() if "improvement" in m) and \
        all(m["cand"] <= m["base"] for m in pqm.values() if "improvement" not in m)
    ship = bool(primary["lo"] > 0 and others_ok)
    result["ship"] = {"rule": "win log-loss CI excludes 0 in favour AND no other reported metric worse (point)",
                      "primary_ci": [primary["lo"], primary["hi"]], "all_other_metrics_not_worse": others_ok, "ship": ship}
    result["seasons"] = seasons

    block = grid_layer.fit_block(pq, cols, params, enabled=ship)
    result["manifest_block"] = block
    DOCS.joinpath("f1_grid_layer.json").write_text(json.dumps(result, indent=2))
    pd.concat(rows, ignore_index=True).round(6).to_csv(DOCS / "f1_grid_layer_rows.csv", index=False)
    DOCS.joinpath("f1_grid_layer.md").write_text(_markdown(result))
    if write_manifest:
        man["grid_layer"] = block
        manifest_module.MANIFEST_PATH.write_text(json.dumps(man, indent=2, default=str))
    print(json.dumps(result["ship"], indent=2), block)
    return result


def _fmt(ms):
    lines = ["| metric | before | after | improvement [95% CI] |", "|---|---|---|---|"]
    for m in ms:
        imp = f"{m['improvement']:+.4f} [{m['lo']:+.4f}, {m['hi']:+.4f}]" if "improvement" in m else "-"
        lines.append(f"| {m['metric']} | {m['base']:.4f} | {m['cand']:.4f} | {imp} |")
    return "\n".join(lines)


def _markdown(r) -> str:
    pq = r["tiers"][session_state.TIER_POST_QUALIFYING]
    out = [f"# F1 grid calibration layer: walk-forward ship test\n",
           f"Seasons {r['seasons'][0]}-{r['seasons'][-1]}, validation folds {r['seasons'][MIN_TRAIN]}-{r['seasons'][-1]} "
           f"({pq['n_races']} races). Exact Plackett-Luce probabilities, no DNF, paired race-level bootstrap "
           f"({N_BOOT} resamples). Improvement = before minus after for losses (positive = layer better).\n",
           f"**Ship verdict: {'SHIP (flag on for post_qualifying)' if r['ship']['ship'] else 'DO NOT SHIP (flag off)'}**\n",
           "## post_qualifying: layer vs raw ranker (current model)\n", _fmt(pq["vs_raw"]),
           "\n## post_qualifying: layer vs grid-position lookup\n", _fmt(pq["vs_grid_lookup"]),
           "\n## post_qualifying: raw ranker vs grid-position lookup (for reference)\n", _fmt(pq["raw_vs_grid_lookup"]),
           "\nPer-season win log-loss improvement: " + ", ".join(f"{s}: {v:+.4f}" for s, v in pq["per_season_win_ll"].items()),
           "\nPer-fold fitted (T, a): " + ", ".join(f"{s}: ({p[0]:.2f}, {p[1]:.2f})" for s, p in pq["per_season_params"].items()),
           "\n## Other tiers (grid unknown, so the grid term is constant and the layer reduces to temperature)\n"]
    for t in (session_state.TIER_PRE_WEEKEND, session_state.TIER_POST_PRACTICE):
        out += [f"### {t} (temperature only; NOT applied in serving)\n", _fmt(r["tiers"][t]["vs_raw"]), ""]
    out += ["## Caveats\n",
            "- Measured without DNF (as the repo's walk-forward does); serving injects DNF into the same layered draw.",
            "- Serving applies the layer in `_predict_upcoming_race` only. `backtest_race` and already-tracked snapshots stay raw "
            "(a backtest fitted on all seasons would look ahead).",
            "- pre_weekend and post_practice have no grid; the temperature-only numbers above are diagnostics, not shipped.\n",
            "## Top-10\n", "p_points_finish is taken from the raw ranker in serving, so top-10 is unchanged by construction.\n",
            f"## Fitted production block (all out-of-fold seasons)\n\n`{json.dumps(r['manifest_block'])}`\n",
            "Rows: `docs/f1_grid_layer_rows.csv` (raw_* and cal_* side by side, plus grid-lookup). "
            "Reproduce: `PYTHONPATH=src:scripts python scripts/grid_layer_eval.py`.\n"]
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-manifest", action="store_true")
    run(ap.parse_args().write_manifest)
