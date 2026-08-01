##############################################################################
### Prediction Interval Coverage Backtest
#
# Validates whether generate_prediction_intervals (in 01_build_nfl_model.R) produces
# *calibrated* floors/ceilings, by rolling the whole train -> predict -> interval cycle back
# one year so that the predicted season has already happened.
#
# WHY ROLLED BACK: at EVAL_YEAR there are no actuals - PRED_YEAR has not been played. The previous
# version of this script joined `actual = points_next_year` off pred_df, which is NA for every
# PRED_YEAR row; a blanket replace_na in 01 then turned that NA into 0, so the script silently
# compared every band against an actual of 0 and reported meaningless numbers. That replace_na now
# excludes points_next_year, and this script derives its own actuals from `points` so that a
# genuine zero can never be confused with a failed join.
#
# HOW TO RUN: this script is self-contained. It needs data/combined_features_{EVAL_YEAR}.csv, which
# 01 writes right after building `combined`. It imports 01's function definitions without executing
# the pipeline, then trains its own holdout models. Expect a few minutes per position (one Bayesian
# tune plus n_bootstrap refits), which is why the tuning budget below is smaller than production's -
# this tests the interval machinery, not the last 1% of point accuracy.
#
# A well-calibrated 80% interval [p10, p90] should contain ~80% of realized outcomes. Read
# miscalibration two ways: a *global* miss (too narrow or too wide everywhere) points at the noise
# model, while a miss that *varies by projection level* (section 3) points at the level
# conditioning itself.
#
# NOTE: one season of ~600 players is informative but noisy, and outcomes are correlated within a
# year. Set RUN_MULTI_YEAR <- TRUE to pool several holdout years before acting on any calibration
# change.
##############################################################################

source("00_globals.R")
source("functions.R")

# Loading only the function definitions and feature vectors from 01. Sourcing the file outright
# would run the entire pipeline, including model training and the plotting blocks.
import_from_script <- function(path, extra_names = character(0)) {
  exprs <- as.list(parse(path))
  keep <- vapply(exprs, function(e) {
    if (!is.call(e) || !identical(as.character(e[[1]]), "<-")) return(FALSE)
    lhs_wanted <- is.name(e[[2]]) && as.character(e[[2]]) %in% extra_names
    rhs_is_fn <- is.call(e[[3]]) && identical(as.character(e[[3]][[1]]), "function")
    rhs_is_fn || lhs_wanted
  }, logical(1))
  invisible(lapply(exprs[keep], eval, envir = globalenv()))
}

# RANDOM_STATE is defined in 01 rather than 00_globals.R, so it is imported here too and stays
# single-sourced - the backtest must use the same seed the production models were trained with.
import_from_script("01_build_nfl_model.R",
                   extra_names = c("qb_features", "rb_features", "wr_features", "RANDOM_STATE"))

# --- Backtest configuration ---
# HOLDOUT_YEAR is the season we pretend is the most recent: train on everything before it, predict
# from it, and score against the season after it, which is real historical data.
HOLDOUT_YEAR   <- EVAL_YEAR - 1
BT_INIT_POINTS <- 8      # smaller tuning budget than production - this is a calibration check
BT_N_ITER      <- 16
BT_N_BOOTSTRAP <- 30
BT_EPI_WEIGHT  <- 1      # scales the symmetric refit spread; try 1 / 0.5 / 0 and compare pin_avg
RUN_MULTI_YEAR <- FALSE  # TRUE pools several holdout years (costs a full tune per year, per position)

# as_tibble matters: fread returns a data.table, and train_position_model indexes with
# pos_df[, feature_cols], which data.table rejects (it wants ..feature_cols). In 01 the group_by /
# ungroup chain has already converted `combined` to a tibble, so this matches what 01 hands it.
combined_bt <-
  fread(paste0("data/combined_features_", as.character(EVAL_YEAR), ".csv")) %>%
  as_tibble() %>%
  mutate(Year = as.Date(Year))   # fwrite/fread round-trips a Date as IDate

# Function to run one full holdout cycle: train, predict, build intervals, attach realized outcomes
#
# Mirrors 01's model_df / pred_df construction exactly, one or more years earlier. Every feature in
# `combined` is strictly backward-looking, so shifting the window needs no data refresh.
#
# Parameters
# ----------
# holdout_year : numeric
#     Season to predict FROM. Training uses Year < holdout_year; actuals come from holdout_year + 1.
#
# Returns
# -------
# data.frame of the interval columns per player plus `actual`, `played_next`, `model_group`.
#
# Notes
# -----
# Players predicted for holdout_year with no holdout_year + 1 row left the league. Their realized
# fantasy total is 0, not missing. Dropping them would bias coverage UPWARD and hide exactly the
# floor failures these bands exist to warn about, so they are kept with actual = 0 and flagged via
# played_next so the report can show both readings side by side.
backtest_year <- function(holdout_year) {
  holdout_date <- as.Date(paste0(holdout_year, "-01-01"))
  next_date    <- as.Date(paste0(holdout_year + 1, "-01-01"))

  # Realized outcome for the season AFTER the holdout, taken from `points` rather than
  # points_next_year so a real 0 is never confused with a missing join
  actuals <- combined_bt %>%
    filter(Year == next_date) %>%
    select(player_id, actual_points = points, actual_games = G)

  bt_pred_df <- combined_bt %>%
    filter(Year == holdout_date, G > 0) %>%
    left_join(actuals, by = "player_id") %>%
    mutate(
      played_next = !is.na(actual_points),
      actual = coalesce(actual_points, 0)
    )

  # The training frame mirrors 01's model_df, low-points filter included, one year earlier
  bt_model_df <- combined_bt %>%
    filter(!is.na(points_next_year), G > 0) %>%
    filter(
      (Pos == "QB" & points_next_year > 35) |
        (Pos %in% c("RB", "WR") & points_next_year > 25) |
        (Pos == "TE" & points_next_year > 15)
    ) %>%
    filter(Year < holdout_date)

  # Guards that would have caught the original vacuity: no actuals at all means either the
  # target-NA fix in 01 is missing, or the holdout year is too recent to have a following season
  stopifnot(nrow(bt_pred_df) > 100, nrow(bt_model_df) > 300)
  if (all(bt_pred_df$actual == 0)) {
    stop("Every actual is 0 for holdout ", holdout_year,
         " - either the points_next_year NA fix in 01 is not in place, or ",
         holdout_year + 1, " is not in the data.")
  }
  cat("\nholdout", holdout_year, "| players:", nrow(bt_pred_df),
      "| never played again:", sum(!bt_pred_df$played_next),
      "| training rows:", nrow(bt_model_df), "\n")

  # Training each position group and generating its intervals, exactly as 01 does
  run_position <- function(position, feature_cols) {
    pos_pred <- bt_pred_df %>%
      filter(
        (position == "WR" & Pos %in% c("WR", "TE")) |
          (position == "QB" & Pos == "QB") |
          (position == "RB" & Pos %in% c("RB", "FB"))
      )

    mdl <- train_position_model(bt_model_df, position, feature_cols,
                                init_points = BT_INIT_POINTS, n_iter = BT_N_ITER,
                                random_state = RANDOM_STATE)

    # The shape models train on the UNTRUNCATED frame for this holdout year - same reasoning as in 01,
    # since model_df's points threshold would delete the busts the floor is meant to represent
    bt_shape_df <- combined_bt %>% filter(G > 0, Year < holdout_date)
    shape <- train_shape_models(mdl, bt_shape_df, position, random_state = RANDOM_STATE)

    generate_prediction_intervals(mdl, shape, bt_model_df, pos_pred, position,
                                  n_bootstrap = BT_N_BOOTSTRAP,
                                  random_state = RANDOM_STATE,
                                  epi_weight = BT_EPI_WEIGHT) %>%
      left_join(
        pos_pred %>% select(player_id, actual, played_next, actual_games),
        by = "player_id"
      ) %>%
      mutate(model_group = position, holdout = holdout_year)
  }

  bind_rows(
    run_position("QB", qb_features),
    run_position("RB", rb_features),
    run_position("WR", wr_features)
  )
}

coverage_all <-
  if (RUN_MULTI_YEAR) {
    bind_rows(lapply(2020:(EVAL_YEAR - 1), backtest_year))
  } else {
    backtest_year(HOLDOUT_YEAR)
  }

# --- 1. Empirical coverage vs nominal ---
# below_floor / above_ceiling reveal the *direction* of miscalibration (each should be ~0.10).
# Pinball loss rides alongside because coverage alone cannot distinguish a well-placed band from a
# lazily wide one - a band can hit 80% coverage by being uselessly broad.
pinball <- function(actual, q, tau) mean(pmax(tau * (actual - q), (tau - 1) * (actual - q)))

coverage_summary <- function(df) {
  df %>%
    summarise(
      n = n(),
      cover_80 = mean(actual >= pred_p10 & actual <= pred_p90),   # nominal 0.80
      cover_90 = mean(actual >= pred_p05 & actual <= pred_p95),   # nominal 0.90
      below_floor = mean(actual < pred_p10),                      # nominal 0.10
      above_ceiling = mean(actual > pred_p90),                    # nominal 0.10
      zero_below_floor = mean(actual == 0 & actual < pred_p10),    # how many misses are the vanish cases
      pin05 = pinball(actual, pred_p05, 0.05),
      pin50 = pinball(actual, pred_p50, 0.50),
      pin95 = pinball(actual, pred_p95, 0.95),
      pin_avg = (pin05 + pin50 + pin95) / 3,
      mean_width_80 = mean(pred_p90 - pred_p10),
      .groups = "drop"
    )
}

# Per position (the WR model splits back out into WR vs TE here) plus a pooled overall row
coverage_report <-
  bind_rows(
    coverage_all %>% group_by(Pos) %>% coverage_summary(),
    coverage_all %>% coverage_summary() %>% mutate(Pos = "ALL", .before = 1)
  ) %>%
  mutate(across(where(is.numeric), ~ round(.x, 3)))

cat("\n=== Interval coverage, ALL predicted players (nominal 80% ~0.80, 90% ~0.90) ===\n")
print(as.data.frame(coverage_report))

# The same table restricted to players who actually played again. The gap between the two tables is
# the contribution of the out-of-league cohort, usually the largest single driver of floor misses.
cat("\n=== Same, restricted to players who played the following season ===\n")
print(as.data.frame(
  bind_rows(
    coverage_all %>% filter(played_next) %>% group_by(Pos) %>% coverage_summary(),
    coverage_all %>% filter(played_next) %>% coverage_summary() %>% mutate(Pos = "ALL", .before = 1)
  ) %>%
    mutate(across(where(is.numeric), ~ round(.x, 3)))
))

# --- 2. Calibration curve ---
# For each predicted quantile, the share of actuals at or below it should equal the nominal level.
calibration_points <- function(df) {
  tibble(
    nominal = c(0.05, 0.10, 0.50, 0.90, 0.95),
    empirical = c(
      mean(df$actual <= df$pred_p05),
      mean(df$actual <= df$pred_p10),
      mean(df$actual <= df$pred_p50),
      mean(df$actual <= df$pred_p90),
      mean(df$actual <= df$pred_p95)
    )
  )
}

calib_df <- bind_rows(
  coverage_all %>% group_by(Pos) %>% group_modify(~ calibration_points(.x)) %>% ungroup(),
  calibration_points(coverage_all) %>% mutate(Pos = "ALL")
)

# Points below the 45-degree line at high quantiles (and above it at low quantiles) mean actuals
# spill outside the band more often than nominal -> intervals too narrow.
calibration_plot <-
  ggplot(calib_df, aes(x = nominal, y = empirical, color = Pos)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#999999") +
  geom_line(linewidth = 0.7, alpha = 0.8) +
  geom_point(size = 2) +
  scale_x_continuous(limits = c(0, 1)) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Prediction Interval Calibration",
    subtitle = paste0("Holdout ", HOLDOUT_YEAR, " -> realized ", HOLDOUT_YEAR + 1,
                      "; empirical share at/below each predicted quantile vs nominal"),
    x = "Nominal quantile",
    y = "Empirical fraction at or below",
    color = "Position"
  ) +
  coord_equal() +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(colour = "#262626", size = 16, face = "bold"),
    panel.grid.minor = element_blank()
  )

print(calibration_plot)

# --- 3. Conditional coverage by projection level ---
# Conditioning the noise on projection level is supposed to make coverage FLAT across tiers. This is
# the direct test: bin players within position by quartile of the point estimate, then pool by tier
# across positions. The failure mode it catches is a homoscedastic tilt - low-projection players
# over-covered and high-projection under-covered - showing as cover_80 drifting Q1 -> Q4.
level_coverage <-
  coverage_all %>%
  group_by(Pos) %>%
  mutate(level_bin = ntile(pred_mean, 4)) %>%   # pred_mean is the point estimate, == Predicted
  ungroup() %>%
  group_by(level_bin) %>%
  coverage_summary() %>%
  mutate(level_bin = factor(level_bin, labels = c("Q1 (low)", "Q2", "Q3", "Q4 (high)"))) %>%
  mutate(across(where(is.numeric), ~ round(.x, 3)))

cat("\n=== Coverage by projection level (within-position quartiles; cover_80 should stay ~0.80) ===\n")
print(as.data.frame(level_coverage))

# --- 4. Do the upside indices actually predict upside? ---
# This is the only test of whether the upside signals carry information, and it is the acceptance
# test for any change to how the bands are conditioned.
#
# IMPORTANT: the input must be ceiling_index, NOT a raw Ceiling / pred_mean ratio. The raw ratio is
# confounded by level - it simply re-selects low-projection players, who then underperform - which
# is precisely what level_adjust exists to remove. Binning on the level-neutral index asks the right
# question: among players at comparable projections, does a high ceiling_index mean a genuinely
# better shot at the upside?
index_signal <-
  coverage_all %>%
  filter(pred_mean > 0) %>%
  group_by(Pos) %>%
  mutate(
    ceil_q  = ntile(ceiling_index, 5),
    floor_q = ntile(floor_index, 5)
  ) %>%
  ungroup()

cat("\n=== ceiling_index quintile vs realized upside (Q5 should beat Q1 on boom_rate) ===\n")
print(as.data.frame(
  index_signal %>%
    group_by(ceil_q) %>%
    summarise(
      n = n(),
      boom_rate = mean(actual > pred_p90),                 # cleared its own ceiling
      med_actual_over_pred = median(actual / pred_mean),
      mean_surprise = mean((actual - pred_mean) / pred_mean),
      .groups = "drop"
    ) %>%
    mutate(across(where(is.numeric), ~ round(.x, 3)))
))

cat("\n=== floor_index quintile vs realized safety (Q5 should have the LOWEST bust rate) ===\n")
print(as.data.frame(
  index_signal %>%
    group_by(floor_q) %>%
    summarise(
      n = n(),
      bust_rate = mean(actual < pred_p10),
      pct_out_of_league = mean(!played_next),
      med_actual_over_pred = median(actual / pred_mean),
      .groups = "drop"
    ) %>%
    mutate(across(where(is.numeric), ~ round(.x, 3)))
))

cat(
  "\nReading the tables:\n",
  "- cover_80 << 0.80 -> intervals too NARROW.  cover_80 >> 0.80 -> too WIDE.\n",
  "- pin_avg is the sharpness-aware score: use it, not coverage, to compare two methods.\n",
  "- below_floor vs above_ceiling imbalance -> skew miscalibration (one tail off more than the other).\n",
  "- the played_next table vs the ALL table -> how much of the floor miss is the out-of-league cohort.\n",
  "- section 3: cover_80 drifting Q1 -> Q4 -> level dependence is not fully corrected.\n",
  "- section 4: if boom_rate is flat or inverted across ceiling_index quintiles, the index carries no\n",
  "  usable upside signal regardless of how well the bands themselves are calibrated.\n"
)
