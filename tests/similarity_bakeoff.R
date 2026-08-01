##############################################################################
### Outcome-Simulation Conditioning Bake-Off
#
# THE QUESTION: a player's simulated outcome distribution has to be conditioned on something. Today
# it is conditioned only on his PROJECTION LEVEL - the outcome noise comes from a pool of pooled
# out-of-bag residuals chosen by which of ten fitted-value deciles his prediction lands in. So two
# players with the same projection get the same outcome spread no matter their age, role volatility,
# or injury history: the bands carry no player-specific information beyond level.
#
# The intuitive fix is to simulate from COMPARABLE PLAYERS instead - find historical analogs and use
# what those analogs actually did the following season. This script tests whether that actually pays,
# comparing three ways to build the outcome pool:
#
#   global - status quo. Pool chosen by fitted-value decile. Level-only conditioning.
#   traj   - analogs by career TRAJECTORY: recent points curve, its slope, career stage, age.
#   leaf   - analogs by XGBoost LEAF CO-OCCURRENCE: the share of trees in which two players land in
#            the same leaf, i.e. similarity as the trained model itself sees it, over all features.
#
# WHY THIS IS A GATE AND NOT A FOREGONE CONCLUSION: on a reduced 20-feature proxy model across
# holdouts 2021-2024 (2,406 player-seasons) the answer SPLIT BY POSITION. Level-only conditioning won
# all 4 WR cells (margins 0.07-3.81%) but only 1 of 4 at QB and 1 of 4 at RB, where traj/leaf won by
# 0.04-2.30% - 6 of 12 cells overall. Level-only still won the POOLED pinball number, but that is an
# artifact of WR being 1,463 of the 2,406 player-seasons: the aggregate is a WR result wearing a
# league-wide label.
#
# So read the per-position table below, NOT the pooled one, as the decision. The models are already
# position-specific, so it is entirely coherent to condition QB/RB on analogs and leave WR on the
# level-only pool. The margins are small either way, and the proxy understated leaf more than traj
# because leaf similarity depends directly on model quality - which is why this re-runs against the
# REAL tuned per-position models and full feature sets.
#
# SCORING, in priority order:
#   1. pin_avg  - pinball loss averaged over the 5th/50th/95th percentiles. THE headline number:
#                 unlike coverage, it rewards a band for being well-placed AND sharp, so it cannot be
#                 gamed by widening. Lower is better.
#   2. flatness of cover_80 across projection quartiles - is the conditioning working at every level?
#   3. the ceiling-signal test in tests/interval_coverage_backtest.R section 4.
#
# HOW TO RUN: self-contained. Needs data/combined_features_{EVAL_YEAR}.csv, written by 01 right after
# it builds `combined`. Tuning happens ONCE per position and the parameters are reused across holdout
# years - this tests the conditioning layer, not hyperparameters. Expect several minutes per position.
##############################################################################

source("00_globals.R")
source("functions.R")

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

# --- Configuration ---
BAKEOFF_YEARS  <- (EVAL_YEAR - 3):(EVAL_YEAR - 1)  # holdout seasons to predict FROM
K_GRID         <- c(50, 100, 200)   # analog counts to try; K was left untuned in the proxy run
N_FOLDS        <- 5                 # folds for the honest out-of-fold fitted values
MULT_FLOOR     <- 20                # denominator floor when forming outcome multiples
BO_INIT_POINTS <- 8
BO_N_ITER      <- 16

# Trajectory signature: the recent points curve, its direction, career stage and age. This is the
# "players whose careers looked like this" notion, expressed in columns combined already carries.
TRAJ_COLS <- c("points", "points_last_year", "avg_points_3yr", "points_trend_3yr",
               "seasons_played", "Age")

# as_tibble matters: fread returns a data.table, and train_position_model indexes with
# pos_df[, feature_cols], which data.table rejects (it wants ..feature_cols). In 01 the group_by /
# ungroup chain has already converted `combined` to a tibble, so this matches what 01 hands it.
combined_bo <-
  fread(paste0("data/combined_features_", as.character(EVAL_YEAR), ".csv")) %>%
  as_tibble() %>%
  mutate(Year = as.Date(Year))

pinball <- function(actual, q, tau) mean(pmax(tau * (actual - q), (tau - 1) * (actual - q)))

# Weighted empirical quantile of a simulated outcome sample. Exact and deterministic - the outcome
# pool IS the distribution, so there is no need to resample it and no Monte Carlo error to carry.
weighted_quantile <- function(x, w, probs) {
  o <- order(x)
  x <- x[o]
  cum_w <- cumsum(w[o]) / sum(w)
  vapply(probs, function(p) x[which(cum_w >= p)[1]], numeric(1))
}

PROBS <- c(0.05, 0.10, 0.50, 0.90, 0.95)

# Function to build the analog outcome table for one position group and holdout year
#
# Every row is a historical player-season paired with what that player ACTUALLY did the following
# season, expressed as a multiple of what was honestly expected of him.
#
# Notes
# -----
# Two things here are easy to get wrong and both make the simulated bands far too narrow or too high:
#
#   1. The expectation must be OUT-OF-SAMPLE. In-sample fit is too tight, which shrinks every multiple
#      toward 1.0. Hence the k-fold out-of-fold predictions rather than fitted values.
#   2. The frame must be UNTRUNCATED. 01's model_df drops next-season totals below 35/25/15 points,
#      which removes precisely the bust outcomes a floor is supposed to represent. Players with no
#      following season left the league, which is a realized 0, not missing data.
build_analog_table <- function(position, holdout_date, feature_cols, best_params, best_nrounds) {
  pos_rows <- combined_bo %>%
    filter(
      (position == "WR" & Pos %in% c("WR", "TE")) |
        (position == "QB" & Pos == "QB") |
        (position == "RB" & Pos %in% c("RB", "FB"))
    )

  # points_next_year is already the realized following-season total, and since 01 stopped filling it
  # with 0 it is NA exactly for players who have no next season - i.e. who left the league. That is a
  # realized outcome of 0, not missing data, so coalesce rather than drop.
  analog <- pos_rows %>%
    filter(Year < holdout_date, G > 0) %>%
    mutate(outcome = coalesce(points_next_year, 0))

  X <- data.matrix(analog[, feature_cols, drop = FALSE])

  # The expectation each multiple is measured against must sit on the SAME scale as the point model's
  # prediction, or `point_pred * multiple` is systematically off. So the folds train on the TRUNCATED
  # frame (production's filter, mean target ~112) while predicting EVERY analog row including the
  # busts and the out-of-league zeros. Training on the untruncated frame instead put oof_fitted on a
  # ~2x lower scale (mean ~55), which inflated every arm's band and - because pinball in an inflated
  # regime quietly rewards narrower bands - handed an unearned edge to whichever method was tightest.
  #
  # Rows outside the truncated set were never in any fold's training data, so every fold is genuinely
  # out-of-sample for them and their prediction is averaged across all folds.
  in_trunc <- with(analog,
    !is.na(points_next_year) &
      ((Pos == "QB" & points_next_year > 35) |
         (Pos %in% c("RB", "WR") & points_next_year > 25) |
         (Pos == "TE" & points_next_year > 15)))

  fold_params <- list(objective = "reg:squarederror", eval_metric = "rmse", tree_method = "hist",
                      max_depth = round(best_params[["max_depth"]]), eta = best_params[["eta"]],
                      gamma = best_params[["gamma"]],
                      min_child_weight = best_params[["min_child_weight"]],
                      subsample = best_params[["subsample"]],
                      colsample_bytree = best_params[["colsample_bytree"]],
                      lambda = best_params[["lambda"]], alpha = best_params[["alpha"]],
                      seed = RANDOM_STATE)

  set.seed(RANDOM_STATE)
  trunc_idx <- which(in_trunc)
  folds <- sample(rep_len(seq_len(N_FOLDS), length(trunc_idx)))
  oof <- numeric(nrow(analog))
  out_of_trunc_preds <- matrix(NA_real_, nrow = sum(!in_trunc), ncol = N_FOLDS)

  for (f in seq_len(N_FOLDS)) {
    fit_idx <- trunc_idx[folds != f]
    booster <- xgb.train(
      params = fold_params,
      data = xgb.DMatrix(X[fit_idx, , drop = FALSE],
                         label = analog$points_next_year[fit_idx]),
      nrounds = best_nrounds, verbose = 0
    )
    held_idx <- trunc_idx[folds == f]
    oof[held_idx] <- predict(booster, X[held_idx, , drop = FALSE])
    out_of_trunc_preds[, f] <- predict(booster, X[!in_trunc, , drop = FALSE])
  }
  oof[!in_trunc] <- rowMeans(out_of_trunc_preds)

  # A near-zero denominator would otherwise produce an absurd multiple, so floor it and winsorize
  mult <- analog$outcome / pmax(oof, MULT_FLOOR)
  bounds <- quantile(mult, c(0.005, 0.995), na.rm = TRUE)

  analog %>%
    mutate(oof_fitted = oof, multiple = pmin(pmax(mult, bounds[1]), bounds[2]))
}

# Function to score all three conditioning methods for one position group and holdout year
run_case <- function(position, holdout_year, feature_cols, tuned) {
  holdout_date <- as.Date(paste0(holdout_year, "-01-01"))
  next_date    <- as.Date(paste0(holdout_year + 1, "-01-01"))

  actuals <- combined_bo %>%
    filter(Year == next_date) %>%
    select(player_id, actual_points = points)

  pos_pred <- combined_bo %>%
    filter(Year == holdout_date, G > 0) %>%
    filter(
      (position == "WR" & Pos %in% c("WR", "TE")) |
        (position == "QB" & Pos == "QB") |
        (position == "RB" & Pos %in% c("RB", "FB"))
    ) %>%
    left_join(actuals, by = "player_id") %>%
    mutate(played_next = !is.na(actual_points), actual = coalesce(actual_points, 0))

  if (nrow(pos_pred) < 30) return(NULL)

  analog <- build_analog_table(position, holdout_date, feature_cols,
                              tuned$best_params, tuned$best_nrounds)
  if (nrow(analog) < 300) return(NULL)

  X_an <- data.matrix(analog[, feature_cols, drop = FALSE])
  X_te <- data.matrix(pos_pred[, feature_cols, drop = FALSE])
  point_pred <- predict(tuned$model, newdata = X_te)

  # Leaf co-occurrence needs the leaf index matrices from the point model itself. Both this and the
  # trajectory matrix are transposed ONCE here rather than inside the per-player loop - transposing a
  # (n_analogs x n_trees) matrix once per player is the difference between seconds and many minutes.
  leaf_an_t <- t(predict(tuned$model, newdata = X_an, predleaf = TRUE))
  leaf_te <- predict(tuned$model, newdata = X_te, predleaf = TRUE)

  # Trajectory space, standardized on the analog pool so no single column dominates the distance
  traj_center <- colMeans(as.matrix(analog[, TRAJ_COLS]), na.rm = TRUE)
  traj_scale  <- apply(as.matrix(analog[, TRAJ_COLS]), 2, sd, na.rm = TRUE)
  traj_scale[!is.finite(traj_scale) | traj_scale == 0] <- 1
  T_an <- scale(as.matrix(analog[, TRAJ_COLS]), center = traj_center, scale = traj_scale)
  T_te <- scale(as.matrix(pos_pred[, TRAJ_COLS]), center = traj_center, scale = traj_scale)
  T_an[!is.finite(T_an)] <- 0
  T_te[!is.finite(T_te)] <- 0
  T_an_t <- t(T_an)

  # Status-quo pool: multiples binned by projection level (level-only conditioning).
  #
  # The bins are matched on PERCENTILE RANK WITHIN EACH POPULATION, not on absolute fitted value,
  # because the two sides sit on different scales here: point_pred comes from the point model trained
  # on the TRUNCATED frame (mean target ~112) while oof_fitted comes from folds trained on the
  # UNTRUNCATED analog frame including zero outcomes (mean ~55) - a factor of ~2. Matching raw values
  # pushed every holdout player into a bin well above his true level, so global drew multiples from
  # higher-fitted players whose outcome ratios are tighter and more optimistic, and its floor came out
  # far too high (below_q10 = 0.199 against a nominal 0.10). That was an artifact of THIS harness, not
  # of production: in generate_prediction_intervals the OOB pool and point_pred come from models
  # trained on the same frame, so their scales already agree. Rank-matching restores the fair
  # comparison and is what "same projection tier" means anyway.
  an_bin <- ntile(analog$oof_fitted, 10)
  te_bin <- ntile(point_pred, 10)

  out <- vector("list", 0)
  for (j in seq_len(nrow(pos_pred))) {
    base <- list(player_id = pos_pred$player_id[j], Pos = pos_pred$Pos[j],
                 actual = pos_pred$actual[j], played_next = pos_pred$played_next[j],
                 pred = point_pred[j])

    # global: equal weight over the level-matched pool
    pool_g <- analog$multiple[an_bin == te_bin[j]]
    q_g <- weighted_quantile(point_pred[j] * pool_g, rep(1, length(pool_g)), PROBS)
    out[[length(out) + 1]] <- c(base, list(method = "global", K = NA_integer_),
                                setNames(as.list(q_g), c("q05","q10","q50","q90","q95")))

    s_leaf <- colMeans(leaf_an_t == leaf_te[j, ])   # share of trees in which the analog shares a leaf
    d_traj <- sqrt(colSums((T_an_t - T_te[j, ])^2))

    for (k in K_GRID) {
      top_l <- order(s_leaf, decreasing = TRUE)[seq_len(min(k, length(s_leaf)))]
      q_l <- weighted_quantile(point_pred[j] * analog$multiple[top_l], s_leaf[top_l], PROBS)
      out[[length(out) + 1]] <- c(base, list(method = "leaf", K = k),
                                  setNames(as.list(q_l), c("q05","q10","q50","q90","q95")))

      # Gaussian kernel with the local Kth distance as bandwidth, so weighting adapts to density
      top_t <- order(d_traj)[seq_len(min(k, length(d_traj)))]
      h <- max(d_traj[top_t[length(top_t)]], 1e-6)
      w_t <- exp(-(d_traj[top_t] / h)^2)
      q_t <- weighted_quantile(point_pred[j] * analog$multiple[top_t], w_t, PROBS)
      out[[length(out) + 1]] <- c(base, list(method = "traj", K = k),
                                  setNames(as.list(q_t), c("q05","q10","q50","q90","q95")))
    }
  }

  bind_rows(lapply(out, as.data.frame)) %>%
    mutate(model_group = position, holdout = holdout_year)
}

# Tuning once per position, on the earliest holdout's training frame, then reusing the parameters
# across every holdout year. The conditioning layer is what is under test here, not the tuning.
tune_position <- function(position, feature_cols) {
  cat("\n=== tuning", position, "(once, reused across holdout years) ===\n")
  first_date <- as.Date(paste0(min(BAKEOFF_YEARS), "-01-01"))
  bo_model_df <- combined_bo %>%
    filter(!is.na(points_next_year), G > 0) %>%
    filter(
      (Pos == "QB" & points_next_year > 35) |
        (Pos %in% c("RB", "WR") & points_next_year > 25) |
        (Pos == "TE" & points_next_year > 15)
    ) %>%
    filter(Year < first_date)

  train_position_model(bo_model_df, position, feature_cols,
                       init_points = BO_INIT_POINTS, n_iter = BO_N_ITER,
                       random_state = RANDOM_STATE)
}

positions <- list(QB = qb_features, RB = rb_features, WR = wr_features)
results <- list()
for (position in names(positions)) {
  tuned <- tune_position(position, positions[[position]])
  for (yr in BAKEOFF_YEARS) {
    cat("  scoring", position, "holdout", yr, "\n")
    results[[length(results) + 1]] <- run_case(position, yr, tuned$features, tuned)
  }
}
bakeoff <- bind_rows(results)

score <- function(df) df %>% summarise(
  n = n(),
  cover_80 = mean(actual >= q10 & actual <= q90),
  cover_90 = mean(actual >= q05 & actual <= q95),
  below_q10 = mean(actual < q10),
  above_q90 = mean(actual > q90),
  pin05 = pinball(actual, q05, 0.05),
  pin50 = pinball(actual, q50, 0.50),
  pin95 = pinball(actual, q95, 0.95),
  pin_avg = (pin05 + pin50 + pin95) / 3,
  width80 = mean(q90 - q10),
  .groups = "drop"
)

cat("\n=== POOLED across holdouts (lower pin_avg wins; cover_80 should be ~0.80) ===\n")
print(as.data.frame(bakeoff %>% group_by(method, K) %>% score() %>%
                      mutate(across(where(is.numeric), ~ round(.x, 3))) %>% arrange(pin_avg)))

cat("\n=== By position (best K per method) ===\n")
print(as.data.frame(bakeoff %>% group_by(model_group, method, K) %>% score() %>%
                      mutate(across(where(is.numeric), ~ round(.x, 3))) %>%
                      arrange(model_group, pin_avg)))

cat("\n=== By holdout year: is the winner stable, or one year of noise? ===\n")
print(as.data.frame(bakeoff %>% group_by(holdout, method, K) %>% score() %>%
                      select(holdout, method, K, cover_80, pin_avg) %>%
                      mutate(across(where(is.numeric), ~ round(.x, 3))) %>%
                      arrange(holdout, pin_avg)))

# The decisive view: who wins each position x year cell on its own. A method can lose the pooled
# number purely because WR contributes most of the rows, so count cells rather than trusting the
# aggregate - that is exactly how the proxy run's "level-only wins everywhere" reading went wrong.
cat("\n=== PER-CELL winners (best K per method within each cell) - READ THIS ONE ===\n")
cell_best <- bakeoff %>%
  group_by(model_group, holdout, method) %>%
  score() %>%
  group_by(model_group, holdout, method) %>%
  summarise(pin_avg = min(pin_avg), .groups = "drop")

cell_winners <- cell_best %>%
  group_by(model_group, holdout) %>%
  summarise(
    winner = method[which.min(pin_avg)],
    global_pin = round(pin_avg[method == "global"], 3),
    best_analog = round(min(pin_avg[method != "global"]), 3),
    global_margin_pct = round(100 * (min(pin_avg[method != "global"]) -
                                       pin_avg[method == "global"]) /
                                pin_avg[method == "global"], 2),
    .groups = "drop"
  )
print(as.data.frame(cell_winners))

cat("\n--- cell wins by method, overall and per position ---\n")
print(table(cell_winners$winner))
print(table(cell_winners$model_group, cell_winners$winner))

cat("\n=== Coverage flatness across projection quartiles (criterion 2) ===\n")
print(as.data.frame(bakeoff %>% filter(pred > 0) %>%
  group_by(method, K, model_group) %>% mutate(level_q = ntile(pred, 4)) %>%
  group_by(method, K, level_q) %>%
  summarise(cover_80 = round(mean(actual >= q10 & actual <= q90), 3), .groups = "drop") %>%
  tidyr::pivot_wider(names_from = level_q, values_from = cover_80, names_prefix = "Q")))

cat(
  "\nDeciding, PER POSITION (the pooled table is dominated by whichever group has the most rows):\n",
  "- A method should only be adopted for a position if it wins MOST of that position's holdout years,\n",
  "  not just the pooled number. Mixed conditioning is fine - the models are already per-position.\n",
  "- If `global` wins a position, its level-only conditioning is adequate there and no analog\n",
  "  machinery should be built for it. That is a real and useful outcome, not a failure.\n",
  "- Margins under ~1% with an inconsistent winner across years are noise. Prefer the simpler method\n",
  "  when the evidence is a tie, and prefer consistency across years over a single large margin.\n",
  "- Cross-check any winner against coverage flatness above and the ceiling-signal test in\n",
  "  tests/interval_coverage_backtest.R section 4 before committing to it.\n"
)
