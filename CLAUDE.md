# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

An R pipeline that predicts next-season NFL fantasy football points, blends those predictions with FantasyPros expert projections, and produces tiered draft rankings. There is no build system, linter, or test framework — scripts are run interactively in RStudio (`ff.Rproj`).

## Critical: How Scripts Are Run

**Do NOT run these scripts end-to-end with `Rscript`.** Per the warning at the top of `01_build_nfl_model.R`, scripts are intended to be evaluated in steps (interactively) so data quality and model behavior can be inspected along the way. The FantasyPros scrape in `02` has deliberate delays; model training runs Bayesian hyperparameter optimization that takes a long time.

## Pipeline Architecture

Scripts run in numbered order, passing data between stages via CSVs in `data/` (gitignored — files exist locally only):

1. **`00_globals.R`** — sourced by every other script. Installs/loads packages and defines globals: `EVAL_YEAR` (final training year), `PRED_YEAR` (= EVAL_YEAR + 1), `START_YEAR` (2006 — aligns with start of ESPN QBR coverage), `SCORING_TYPE` ("PPR"/"HALF"/"STANDARD"), `PPR_MULT`.

2. **`01_build_nfl_model.R`** — sources `functions.R` for data intake. Player data comes from **nflreadr/nflverse** (the legacy PFR scraper is retired — PFR blocks scraping): `build_player_season_stats()` pulls season-level player stats, rosters (age/draft capital/ESPN-id crosswalk via `load_ff_playerids`), ESPN QBR, and team wins from schedules, writing one cached table `data/player_stats_final.csv` keyed by **GSIS `player_id`** (controlled by `SKIP_DATA_LOAD`). Substitutions for columns PFR had but nflverse lacks are documented in the `build_player_season_stats` docblock (success rates → EPA-per-play; longs → air yards/YAC; awards → draft capital; QB record → team wins; GS/4QC/GWD dropped). ESPN QBR only exists for *qualified* QBs — backups get zero by design. The script then builds the `combined` dataset with fantasy points and ~80 engineered features (lags, 3-yr rolling averages, career cumulatives, age interactions, injury flags), and trains **three position-specific XGBoost models** (`train_position_model`) with Bayesian optimization. Position grouping: QB alone; RB includes FB; **the "WR" model includes TEs**. Output: `data/model_pred_{PRED_YEAR}_{SCORING_TYPE}.csv`.

3. **`02_combine_projections.R`** — scrapes FantasyPros draft projections, joins to model predictions on cleaned player names (`clean_player_name` strips suffixes/punctuation — name matching is a recurring pain point), applies **manual per-player multipliers** (the big `case_when` block for injuries, depth-chart changes, etc. — updated by hand each season), then blends: `PRED_WEIGHT * model + PROJ_WEIGHT * dampened_expert`, with penalty factors when only an expert projection exists (rookies). Output: `data/blended_proj_{PRED_YEAR}_{SCORING_TYPE}.csv`.

4. **`03_create_player_tiers.R`** — trims the player pool to platform-specific positional cutoffs (commented blocks for Underdog/DraftKings/ESPN/Yahoo — uncomment the right one), computes `Relative_Value` as a 50/50 blend of z-score value and VORP (VORP scaled ×2.25, derived in `tests/estimate_vorp_zscore_blend.R`), applies positional dampening (`QB_DAMP`, etc.), then k-means clusters into overall and per-position tiers (use the elbow plots to pick k). Output: `data/final_projections_{PRED_YEAR}_{SCORING_TYPE}.csv`.

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
- nflverse season summaries arrive pre-aggregated for traded players — no "2TM" handling needed.
- Annual maintenance: bump `EVAL_YEAR`, set `SKIP_DATA_LOAD <- FALSE` once to refresh `data/player_stats_final.csv`, refresh the manual player-adjustment `case_when` in `02`, and verify positional cutoffs in `03`.
