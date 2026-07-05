##############################################################################
### Prediction Interval Coverage Backtest
#
# Validates whether generate_prediction_intervals (in 01_build_nfl_model.R) produces
# *calibrated* floors/ceilings. For the held-out EVAL_YEAR season we already know each
# player's actual next-season points (points_next_year is historical), and the interval
# percentiles predict that same quantity, so coverage is checkable with zero recompute.
#
# A well-calibrated 80% interval [p10, p90] should contain ~80% of the realized outcomes.
# The aleatoric noise is already heteroscedastic (OOB residuals are binned by fitted value in
# generate_prediction_intervals), so read miscalibration two ways: a *global* miss (too narrow or
# too wide everywhere) points to more noise draws or a calibration multiplier, while a miss that
# *varies by projection level* (see section 3) points back at the residual binning itself.
#
# HOW TO RUN: step through 01_build_nfl_model.R at least through the interval-generation
# block, leaving qb_intervals / rb_intervals / wr_intervals and qb_pred_df / rb_pred_df /
# wr_pred_df in the global environment, then source or step through this script.
#
# NOTE: this is a single-season check (~the most recent held-out year). One season of
# ~100 players per position is informative but noisy; for a sturdier estimate, repeat the
# train -> predict -> interval cycle across several EVAL_YEAR values and pool the results.
##############################################################################

source("00_globals.R")

# Guard: the script consumes objects produced by 01, fail early with a clear message if absent
required_objs <- c(
  "qb_intervals", "rb_intervals", "wr_intervals",
  "qb_pred_df", "rb_pred_df", "wr_pred_df"
)
missing_objs <- required_objs[!vapply(required_objs, exists, logical(1))]
if (length(missing_objs) > 0) {
  stop(
    "Missing required objects: ", paste(missing_objs, collapse = ", "),
    ".\nRun 01_build_nfl_model.R through the interval-generation step first."
  )
}

# Join each position's intervals to its realized next-year outcome.
# pred_df is filtered to !is.na(points_next_year) upstream, so every player has an actual.
attach_actuals <- function(intervals_df, pred_df) {
  intervals_df %>%
    inner_join(
      pred_df %>% select(player_id, actual = points_next_year),
      by = "player_id"
    )
}

coverage_all <- bind_rows(
  attach_actuals(qb_intervals, qb_pred_df),
  attach_actuals(rb_intervals, rb_pred_df),
  attach_actuals(wr_intervals, wr_pred_df)
)

# --- 1. Empirical coverage vs nominal ---
# below_floor / above_ceiling reveal *direction* of miscalibration (each should be ~0.10).
coverage_summary <- function(df) {
  df %>%
    summarise(
      n = n(),
      cover_80 = mean(actual >= pred_p10 & actual <= pred_p90),  # nominal 0.80
      cover_90 = mean(actual >= pred_p05 & actual <= pred_p95),  # nominal 0.90
      below_floor = mean(actual < pred_p10),                     # nominal 0.10
      above_ceiling = mean(actual > pred_p90),                   # nominal 0.10
      mean_width_80 = mean(pred_p90 - pred_p10),
      .groups = "drop"
    )
}

# Per position (WR model splits back out into WR vs TE here) plus a pooled overall row
coverage_report <-
  bind_rows(
    coverage_all %>% group_by(Pos) %>% coverage_summary(),
    coverage_all %>% coverage_summary() %>% mutate(Pos = "ALL", .before = 1)
  ) %>%
  mutate(across(where(is.numeric), ~ round(.x, 3)))

cat("\n=== Interval coverage (nominal 80% should be ~0.80, 90% ~0.90) ===\n")
print(coverage_report)

# --- 2. Calibration curve ---
# For each predicted quantile, the share of actuals at or below it should equal the nominal
# level. Five stored percentiles give five calibration points: 0.05, 0.10, 0.50, 0.90, 0.95.
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

# Points below the 45-degree line at high quantiles (and above it at low quantiles) mean
# actuals spill outside the band more often than nominal -> intervals too narrow.
calibration_plot <-
  ggplot(calib_df, aes(x = nominal, y = empirical, color = Pos)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#999999") +
  geom_line(linewidth = 0.7, alpha = 0.8) +
  geom_point(size = 2) +
  scale_x_continuous(limits = c(0, 1)) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Prediction Interval Calibration",
    subtitle = "Empirical share of actuals at/below each predicted quantile vs. nominal level",
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
# The whole point of binning the residuals by fitted value is to make coverage FLAT across
# projection tiers. This is the direct test: bin players within their position by quartile of the
# point estimate (pred_p50), then pool by tier across positions. If the heteroscedastic noise is
# working, cover_80 stays ~0.80 in every tier. The failure mode it catches is a homoscedastic tilt
# - low-projection players over-covered (bands too wide) and high-projection under-covered (too
# narrow), which would show as cover_80 rising from Q1 to Q4 (or the reverse).
level_coverage <-
  coverage_all %>%
  group_by(Pos) %>%
  mutate(level_bin = ntile(pred_p50, 4)) %>%   # within-position quartile of the point estimate
  ungroup() %>%
  group_by(level_bin) %>%
  coverage_summary() %>%
  mutate(level_bin = factor(level_bin, labels = c("Q1 (low)", "Q2", "Q3", "Q4 (high)"))) %>%
  mutate(across(where(is.numeric), ~ round(.x, 3)))

cat("\n=== Coverage by projection level (within-position quartiles; cover_80 should stay ~0.80 across tiers) ===\n")
print(level_coverage)

cat(
  "\nReading the tables:\n",
  "- cover_80 << 0.80  -> intervals too NARROW (raise n_noise_draws or add a calibration multiplier).\n",
  "- cover_80 >> 0.80  -> intervals too WIDE.\n",
  "- below_floor vs above_ceiling imbalance -> skew miscalibration (one tail off more than the other).\n",
  "- section 3: cover_80 drifting Q1 -> Q4 -> heteroscedastic tilt remains (residual binning not fully\n",
  "  correcting the level dependence); flat across tiers -> the binned noise is doing its job.\n",
  "On the plot, points sitting below the dashed line at high quantiles indicate under-coverage.\n"
)
