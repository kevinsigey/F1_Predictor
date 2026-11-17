# F1 grid calibration layer: walk-forward ship test

Seasons 2019-2026, validation folds 2022-2026 (112 races). Exact Plackett-Luce probabilities, no DNF, paired race-level bootstrap (2000 resamples). Improvement = before minus after for losses (positive = layer better).

**Ship verdict: SHIP (flag on for post_qualifying)**

## post_qualifying: layer vs raw ranker (current model)

| metric | before | after | improvement [95% CI] |
|---|---|---|---|
| win log-loss | 0.1296 | 0.1036 | +0.0260 [+0.0145, +0.0370] |
| win multiclass NLL | 1.7757 | 1.3503 | +0.4254 [+0.2434, +0.5874] |
| podium log-loss | 0.2793 | 0.2364 | +0.0429 [+0.0278, +0.0579] |
| podium Brier | 0.0868 | 0.0706 | +0.0162 [+0.0113, +0.0212] |
| win ECE | 0.0152 | 0.0083 | - |
| podium ECE | 0.0333 | 0.0260 | - |

## post_qualifying: layer vs grid-position lookup

| metric | before | after | improvement [95% CI] |
|---|---|---|---|
| win log-loss | 0.1122 | 0.1036 | +0.0086 [+0.0034, +0.0136] |
| win multiclass NLL | 1.5011 | 1.3503 | +0.1508 [+0.0753, +0.2290] |
| podium log-loss | 0.2358 | 0.2364 | -0.0005 [-0.0105, +0.0093] |
| podium Brier | 0.0699 | 0.0706 | -0.0008 [-0.0036, +0.0020] |
| win ECE | 0.0130 | 0.0083 | - |
| podium ECE | 0.0246 | 0.0260 | - |

## post_qualifying: raw ranker vs grid-position lookup (for reference)

| metric | before | after | improvement [95% CI] |
|---|---|---|---|
| win log-loss | 0.1122 | 0.1296 | -0.0174 [-0.0294, -0.0052] |
| win multiclass NLL | 1.5011 | 1.7757 | -0.2746 [-0.4897, -0.0683] |
| podium log-loss | 0.2358 | 0.2793 | -0.0435 [-0.0614, -0.0261] |
| podium Brier | 0.0699 | 0.0868 | -0.0170 [-0.0232, -0.0109] |
| win ECE | 0.0130 | 0.0152 | - |
| podium ECE | 0.0246 | 0.0333 | - |

Per-season win log-loss improvement: 2022: +0.0264, 2023: +0.0203, 2024: +0.0229, 2025: +0.0321, 2026: +0.0280

Per-fold fitted (T, a): 2022: (0.45, 1.44), 2023: (0.37, 1.44), 2024: (0.51, 1.40), 2025: (0.50, 1.40), 2026: (0.54, 1.45)

## Other tiers (grid unknown, so the grid term is constant and the layer reduces to temperature)

### pre_weekend (temperature only; NOT applied in serving)

| metric | before | after | improvement [95% CI] |
|---|---|---|---|
| win log-loss | 0.1565 | 0.1519 | +0.0046 [+0.0021, +0.0070] |
| win multiclass NLL | 2.2548 | 2.1629 | +0.0919 [+0.0519, +0.1303] |
| podium log-loss | 0.3304 | 0.3274 | +0.0029 [-0.0021, +0.0078] |
| podium Brier | 0.1031 | 0.1041 | -0.0010 [-0.0028, +0.0007] |
| win ECE | 0.0174 | 0.0079 | - |
| podium ECE | 0.0466 | 0.0429 | - |

### post_practice (temperature only; NOT applied in serving)

| metric | before | after | improvement [95% CI] |
|---|---|---|---|
| win log-loss | 0.1565 | 0.1519 | +0.0046 [+0.0023, +0.0071] |
| win multiclass NLL | 2.2548 | 2.1629 | +0.0919 [+0.0533, +0.1301] |
| podium log-loss | 0.3304 | 0.3274 | +0.0029 [-0.0020, +0.0080] |
| podium Brier | 0.1031 | 0.1041 | -0.0010 [-0.0028, +0.0008] |
| win ECE | 0.0174 | 0.0079 | - |
| podium ECE | 0.0466 | 0.0429 | - |

## Caveats

- Measured without DNF (as the repo's walk-forward does); serving injects DNF into the same layered draw.
- Serving applies the layer in `_predict_upcoming_race` only. `backtest_race` and already-tracked snapshots stay raw (a backtest fitted on all seasons would look ahead).
- pre_weekend and post_practice have no grid; the temperature-only numbers above are diagnostics, not shipped.

## Top-10

p_points_finish is taken from the raw ranker in serving, so top-10 is unchanged by construction.

## Fitted production block (all out-of-fold seasons)

`{"enabled": true, "temperature": 0.6064242114232797, "grid_coef": 1.4515371535992916, "tier": "post_qualifying", "fit_seasons": [2020, 2026], "fit_races": 151}`

Rows: `docs/f1_grid_layer_rows.csv` (raw_* and cal_* side by side, plus grid-lookup). Reproduce: `PYTHONPATH=src:scripts python scripts/grid_layer_eval.py`.
