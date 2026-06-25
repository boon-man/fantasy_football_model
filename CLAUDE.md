# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

An R pipeline that predicts next-season NFL fantasy football points, blends those predictions with FantasyPros expert projections, and produces tiered draft rankings. There is no build system, linter, or test framework — scripts are run interactively in RStudio (`ff.Rproj`).

## Critical: How Scripts Are Run

**Do NOT run these scripts end-to-end with `Rscript`.** Per the warning at the top of `01_build_nfl_model.R`, scripts are intended to be evaluated in steps (interactively) so data quality and model behavior can be inspected along the way. The FantasyPros scrape in `02` has deliberate delays; model training runs Bayesian hyperparameter optimization that takes a long time.

## Pipeline Architecture

Scripts run in numbered order, passing data between stages via CSVs in `data/` (gitignored — files exist locally only):

1. **`00_globals.R`** — sourced by every other script. Installs/loads packages and defines globals: `EVAL_YEAR` (final training year), `PRED_YEAR` (= EVAL_YEAR + 1), `START_YEAR` (2006 — aligns with start of ESPN QBR coverage), `SCORING_TYPE` ("PPR"/"HALF"/"STANDARD"), `PPR_MULT`.

2. **`01_build_nfl_model.R`** — sources `functions.R` for data intake. Player data comes from **nflreadr/nflverse** (the legacy PFR scraper is retired — PFR blocks scraping): `build_player_season_stats()` pulls season-level player stats, rosters (age/draft capital/ESPN-id crosswalk via `load_ff_playerids`), ESPN QBR, and team wins from schedules, writing one cached table `data/player_stats_final.csv` keyed by **GSIS `player_id`** (controlled by `SKIP_DATA_LOAD`). Substitutions for columns PFR had but nflverse lacks are documented in the `build_player_season_stats` docblock (success rates → EPA-per-play; longs → air yards/YAC; awards → draft capital; QB record → team wins; GS/4QC/GWD dropped). `build_player_season_stats` also carries nflverse **opportunity-share / efficiency** columns through `transmute`: `target_share`/`air_yards_share`/`wopr`/`racr` (receiving) and `passing_cpoe`/`pacr`/`passing_adot` (QB; `passing_adot = passing_air_yards / attempts` is depth of target — the one passing metric that isolates air yards from YAC). ESPN QBR only exists for *qualified* QBs — backups get zero by design. The script then builds the `combined` dataset with fantasy points and ~80 engineered features (lags, 3-yr rolling averages of the above shares/efficiency stats included, career cumulatives, age interactions, injury flags), and trains **three position-specific XGBoost models** (`train_position_model`) with Bayesian optimization. Position grouping: QB alone; RB includes FB; **the "WR" model includes TEs**. `train_position_model` takes a `random_state` (the `RANDOM_STATE` config knob) threaded through the split, baseline, tuning, and final fit — change it to generate alternate draft scenarios. After training, `generate_prediction_intervals` runs a player-level (cluster) bootstrap — reusing each model's tuned hyperparameters, with multiple OOB-residual noise draws per fit — to produce per-player prediction intervals: `Floor`/`Ceiling` (p10/p90), the full percentile set, and `implied_upside` (a scale-free upside/downside ratio used as a within-tier draft tie-breaker). `tests/interval_coverage_backtest.R` validates their empirical coverage. Output: `data/model_pred_{PRED_YEAR}_{SCORING_TYPE}.csv`, keyed by **GSIS `player_id`** and carrying the interval columns alongside `Predicted`.

3. **`02_combine_projections.R`** — scrapes FantasyPros draft projections, joins them to model predictions on cleaned player names (`clean_player_name` strips suffixes/punctuation). FantasyPros ships **no player id**, so the expert side is name-matched, but the model side carries its **GSIS `player_id`** so same-named players (e.g. the two Adrian Petersons) stay distinct; when both match one FantasyPros row the projection attaches only to the highest-projected (active) player, and the other is dropped by the no-FantasyPros filter. Applies **manual per-player multipliers** (the big `case_when` block for injuries, depth-chart changes, etc. — updated by hand each season, and applied **only** to the model prediction, not the interval columns), then blends: `PRED_WEIGHT * model + PROJ_WEIGHT * dampened_expert`, with penalty factors when only an expert projection exists (rookies). The model's `Floor`/`Ceiling`/`implied_upside` ride through untouched. Output: `data/blended_proj_{PRED_YEAR}_{SCORING_TYPE}.csv` (carries `player_id` + interval columns).

4. **`03_create_player_tiers.R`** — trims the player pool to platform-specific positional cutoffs (commented blocks for Underdog/DraftKings/ESPN/Yahoo — uncomment the right one), computes `Relative_Value` as a 50/50 blend of z-score value and VORP (VORP scaled ×2.25, derived in `tests/estimate_vorp_zscore_blend.R`), applies positional dampening (`QB_DAMP`, etc.), then k-means clusters into overall and per-position tiers (use the elbow plots to pick k). Tier assembly reads straight from the clustered frame (no name-based self-join). Output: `data/final_projections_{PRED_YEAR}_{SCORING_TYPE}.csv`, carrying `player_id` and the model interval columns (`Floor`/`Ceiling`/`implied_upside`) through to the final sheet.

### Cross-script state dependency

`02_combine_projections.R`'s `plot_predicted_trajectories` calls reference the `combined` dataframe built in `01_build_nfl_model.R` — it must still be in the global environment (sessions are stateful; scripts are not fully self-contained).

### functions.R

Shared home for data intake (nflverse functions, new code uses `|>` pipes) and cleaning helpers (`clean_player_name`). The legacy PFR functions (`scrapeData`, `clean_traded_players`) remain only for the archived scripts in `tests/` — do not use them for new data pulls. `02_combine_projections.R` still redefines its own `clean_player_name`; remaining modeling/plot functions in `01` are a standing TODO to consolidate.

## `tests/` Directory

Not unit tests — these are one-off analysis/optimization scripts:
- `validate_nflreadr_migration.R` — compares the nflverse dataset against legacy PFR CSVs
- `model_proj_blend_analysis.R` — backtests blend weights against actual season results
- `estimate_vorp_zscore_blend.R` — regression analysis behind the 2.25 VORP multiplier
- `positional_tiering.R`, `nfl_model_old.R` — older exploratory/legacy versions

## Domain Conventions

- Fantasy scoring is computed in `01` (`points` column): 6 pts/rush+rec TD, 4 pts/pass TD, 0.1/rush+rec yd, 0.04/pass yd, −2 fumble/INT, `PPR_MULT` per reception. `rush_fbl` is *total* fumbles (all play types), matching the legacy PFR definition.
- Player `Pos` is normalized to each player's most recent position across their whole history (grouped by `player_id`).
- **`player_id` (GSIS) threads through every stage's CSV** and is the unique key for disambiguating players who share a name. Internal joins in `01`/`03` key on it (not name); only the FantasyPros match in `02` stays name-based because FantasyPros provides no id. FantasyPros-only rookies therefore have `player_id = NA`.
- `Floor`/`Ceiling`/`implied_upside` are the **model's own** prediction-interval expectation — deliberately *not* manually adjusted or blended with FantasyPros — while `Final_Projection` is the robust model+expert blend. The bands therefore will *not* necessarily bracket `Final_Projection`; that is by design (they describe the model, and match what `tests/interval_coverage_backtest.R` validates).
- nflverse season summaries arrive pre-aggregated for traded players — no "2TM" handling needed.
- **All engineered features must be strictly backward-looking** (predicting next-season points, so any feature using current/future-season data leaks). The rolling stats use `rollapplyr(..., align = "right")` and career cumulatives subtract the current season for this reason. `years_since_peak` uses a **trailing** argmax (`running_argmax` helper), not a whole-career `which.max`, which would otherwise recycle the index of a player's peak — including future seasons — into every row.
- Annual maintenance: bump `EVAL_YEAR`, set `SKIP_DATA_LOAD <- FALSE` once to refresh `data/player_stats_final.csv`, refresh the manual player-adjustment `case_when` in `02`, and verify positional cutoffs in `03`.
