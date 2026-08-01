##############################################################################
### Interval Reliability / Stability Check
#
# Asks two SEPARATE questions about the bands from generate_prediction_intervals:
#
#   1. DETERMINISM - with the same random_state, are the bands bit-identical? This must be TRUE.
#      Before the sampled outcome noise was replaced by an exact evaluation of the predictive
#      mixture it was FALSE: set.seed ran once OUTSIDE the per-player loop, so two identical
#      players drew different noise and received different ceilings.
#
#   2. SEED VARIATION - across two random_states the bootstrap resamples different players and the
#      shape models refit, so the bands SHOULD move. This variation is WANTED, not a defect: a
#      different resample is a genuinely different fitted view of a player, and running several seeds
#      is how alternate draft scenarios and player-exposure hedging get generated.
#
#      So do NOT try to drive these numbers to zero. Read them as a PAIR:
#
#        spearman    - is it the same READ on the player, or a different ordering entirely?
#        noise_ratio = sd(index_a - index_b) / sd(index_a) - how much the magnitude moves.
#
#      Healthy for hedging: spearman comfortably high (say > 0.7) with a substantial noise_ratio -
#      the board broadly agrees with itself while individual players shift enough to diversify across.
#      Unhealthy: spearman near 0 - then the ordering is arbitrary rather than an alternate view, and
#      the variation is noise wearing the costume of variance.
#
#      NOTE the distinction that matters. Two kinds of randomness were in here, and only one was a
#      bug: WITHIN-run draw noise (two identical players getting different ceilings in the same run
#      because sample() returned different numbers) reflected no view of anybody and has been removed
#      by evaluating the predictive mixture exactly. ACROSS-run model variation is the useful kind and
#      is deliberately preserved.
#
# WHY THIS MATTERS: the band EDGES and the INDICES behave very differently, because each index is a
# level-detrended studentized residual - it deliberately throws away the level signal and keeps only
# the small remainder, so it amplifies whatever noise is left. On synthetic data the edges came out
# stable (pred_p95 noise_ratio ~0.055 at n_bootstrap = 30) while ceiling_index was still ~0.57,
# improving only as ~1/sqrt(n_bootstrap): 120 refits reached ~0.50 at four times the cost. If that
# pattern reproduces here on real features, more bootstrap iterations are not the fix - the epistemic
# refit spread is simply too noisy at any affordable count to be the primary index signal, which is
# the case for conditioning the outcome distribution on features instead (tests/similarity_bakeoff.R).
#
# HOW TO RUN: step through 01_build_nfl_model.R far enough to have the trained models and prediction
# frames in the global environment (qb_model / rb_model / wr_model, model_df, *_pred_df), then source
# this script. It refits the bootstrap a few times per position, so expect a few minutes.
##############################################################################

source("00_globals.R")

required_objs <- c("qb_model", "rb_model", "wr_model", "model_df",
                   "qb_shape", "rb_shape", "wr_shape",
                   "qb_pred_df", "rb_pred_df", "wr_pred_df")
missing_objs <- required_objs[!vapply(required_objs, exists, logical(1))]
if (length(missing_objs) > 0) {
  stop("Missing required objects: ", paste(missing_objs, collapse = ", "),
       ".\nRun 01_build_nfl_model.R through model training first.")
}

SEED_A <- 12345
SEED_B <- 999
N_BOOT <- 30    # match production; raise to confirm the ~1/sqrt(B) scaling noted above

metrics <- c("ceiling_index", "floor_index", "upside_index", "pred_p05", "pred_p95")

# Function to run the determinism and seed-reliability checks for one position group
#
# Returns one row per metric with the two-seed Spearman and noise_ratio, plus a determinism flag
# covering the whole band table.
check_position <- function(model_object, shape_models, pred_df, position) {
  cat("\n--- ", position, " ---\n", sep = "")

  a1 <- generate_prediction_intervals(model_object, shape_models, model_df, pred_df, position,
                                      n_bootstrap = N_BOOT, random_state = SEED_A)
  a2 <- generate_prediction_intervals(model_object, shape_models, model_df, pred_df, position,
                                      n_bootstrap = N_BOOT, random_state = SEED_A)
  b  <- generate_prediction_intervals(model_object, shape_models, model_df, pred_df, position,
                                      n_bootstrap = N_BOOT, random_state = SEED_B)

  # Determinism: identical seed must reproduce the bands exactly, to zero tolerance
  is_deterministic <- isTRUE(all.equal(a1[, metrics], a2[, metrics], tolerance = 0))
  cat("same-seed bit-identical:", is_deterministic, "\n")
  if (!is_deterministic) {
    warning(position, ": bands are NOT reproducible at a fixed seed - ",
            "something in the interval path is still drawing unseeded randomness.")
  }

  cmp <- a1 %>%
    select(player_id, all_of(metrics)) %>%
    inner_join(b %>% select(player_id, all_of(metrics)), by = "player_id",
               suffix = c("_a", "_b"))

  bind_rows(lapply(metrics, function(m) {
    xa <- cmp[[paste0(m, "_a")]]
    xb <- cmp[[paste0(m, "_b")]]
    tibble(
      Pos = position,
      metric = m,
      n = length(xa),
      spearman = cor(xa, xb, method = "spearman", use = "complete.obs"),
      noise_ratio = sd(xa - xb, na.rm = TRUE) / sd(xa, na.rm = TRUE),
      deterministic = is_deterministic
    )
  }))
}

reliability <- bind_rows(
  check_position(qb_model, qb_shape, qb_pred_df, "QB"),
  check_position(rb_model, rb_shape, rb_pred_df, "RB"),
  check_position(wr_model, wr_shape, wr_pred_df, "WR")
) %>%
  mutate(across(where(is.numeric), ~ round(.x, 3)))

cat("\n=== Seed-to-seed reliability (noise_ratio: <0.3 mostly signal, ~1.0 mostly artifact) ===\n")
print(as.data.frame(reliability))

cat(
  "\nReading the table:\n",
  "- deterministic must be TRUE everywhere. FALSE means unseeded randomness leaked back in, which\n",
  "  breaks reproducibility - a given seed must always rebuild the same board.\n",
  "- noise_ratio is NOT a defect metric here. Seed-to-seed movement is wanted: it is how alternate\n",
  "  draft scenarios and player-exposure hedging get generated. Read it beside spearman.\n",
  "- spearman > ~0.7 with a substantial noise_ratio is the healthy pattern - the same read on the\n",
  "  player, jittered enough to diversify exposure across seeds.\n",
  "- spearman near 0 IS a problem: the ordering is arbitrary rather than an alternate view, so the\n",
  "  movement is noise rather than a second opinion. That is the case worth acting on.\n"
)
