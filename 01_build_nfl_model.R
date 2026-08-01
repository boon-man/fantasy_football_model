# === IMPORTANT === #
# This script is not designed to let it rip all in one go,
# It is intended to be processed or evaluated in steps to ensure that data is being compiled appropriately
# And that the model is running effectively, making "quality" predictions
# PLEASE DO NOT JUST HIT RUN ALL

# === GLOBAL CONFIGURATION === #
source("00_globals.R")  # Running global variable config script
source("functions.R")   # Loading shared cleaning and nflverse data intake functions
source("evaluate_model.R")  # Loading the model performance diagnostic plots


SKIP_DATA_LOAD <- TRUE  # Set to TRUE after the first refresh has cached data locally
RANDOM_STATE <- 1432   # Seed threaded into train_position_model
UPSIDE_BOTH_BONUS <- 0.05  # Extra credit in upside_index when the WEAKER of the two axes also clears 100
UPSIDE_DEFICIT_WEIGHT <- 0.75  # Penalizer for players with disproportionately low floor or low ceiling relative to peers

# Function to train the XGBoost model for a specific position
#
# Tuning budget parameters, lower these for fast code-testing runs:
#   init_points : random configurations evaluated before the Bayesian search starts
#   n_iter      : Bayesian optimization iterations after initialization
#   max_nrounds : tree count ceiling for CV and the final fit, early stopping decides the actual count
#   random_state: single seed threaded through the train/test split, baseline, tuning, and final
#                 fit. Same value reproduces a run exactly; change it to generate alternate draft
#                 scenarios so production isn't over-exposed to one model realization.
#   log_every   : print a CV-RMSE progress line (current + best-so-far) every Nth tuning
#                 evaluation, to watch how quickly performance stabilizes (default 5)
train_position_model <- function(df, position, feature_cols,
                                 init_points = 10,
                                 n_iter = 20,
                                 max_nrounds = 2000,
                                 random_state = 62820,
                                 log_every = 5) {
  # Filter to relevant position
  pos_df <- df %>%
    filter(
      (position == "WR" & Pos %in% c("WR", "TE")) |
        (position == "QB" & Pos == "QB") |
        (position == "RB" & Pos %in% c("RB", "FB"))
    ) %>%
    filter(!is.na(points_next_year), G > 0)

  # Remove missing features and prep matrices
  X <- pos_df[, feature_cols]
  y <- pos_df$points_next_year

  nzv <- nearZeroVar(X)
  if (length(nzv) > 0) {
    X <- X[, -nzv]
    feature_cols <- feature_cols[-nzv]
  }

  # Train-test split
  set.seed(random_state)
  train_idx <- createDataPartition(y, p = 0.8, list = FALSE)
  X_train <- X[train_idx, ]
  y_train <- y[train_idx]
  X_test <- X[-train_idx, ]
  y_test <- y[-train_idx]

  dtrain <- xgb.DMatrix(data = data.matrix(X_train), label = y_train)

  # === BASELINE MODEL === #
  # An untuned random forest on the same split, the benchmark the tuned model must beat
  # Random forests cannot handle missing values, so the matrices are zero filled
  # to match the pipeline convention before training
  baseline_train <- as.data.frame(data.matrix(X_train))
  baseline_test <- as.data.frame(data.matrix(X_test))
  baseline_train[is.na(baseline_train)] <- 0
  baseline_test[is.na(baseline_test)] <- 0

  set.seed(random_state)
  baseline_model <- ranger(x = baseline_train, y = y_train, num.trees = 500)

  # Scoring the baseline on the holdout split
  baseline_preds <- predict(baseline_model, data = baseline_test)$predictions
  baseline_rmse <- sqrt(mean((baseline_preds - y_test)^2))
  baseline_mae <- mean(abs(baseline_preds - y_test))
  cat("Baseline RMSE for", position, "model:", round(baseline_rmse, 3), "\n")

  # === TUNED MODEL === #
  # Bayesian optimization runs fresh on every call by design: the model is run sparingly, each
  # time with a new RANDOM_STATE, to generate a wholly fresh model realization and a systematically
  # different draft board (controlled variance in which players grade out as premier selections,
  # so the same names don't repeatedly top the board). There is no hyperparameter cache to skip.

  # Counters for the periodic performance log, updated inside xgb_cv_bayes via <<-
  eval_counter <- 0
  best_rmse_so_far <- Inf

  # Define Bayesian optimization function
  # The tree count is not part of the search space, early stopping inside the
  # CV finds the right number of rounds for each candidate configuration
  xgb_cv_bayes <- function(max_depth, eta, gamma, min_child_weight, subsample, colsample_bytree, lambda, alpha) {
    # Reset to random_state on every call so all candidates are scored on identical CV folds
    set.seed(random_state)

    # Count this evaluation so the progress log can fire every log_every iterations
    eval_counter <<- eval_counter + 1

    max_depth <- as.integer(round(max_depth))

    # Validate parameters
    if (anyNA(c(max_depth, eta, gamma, min_child_weight, subsample, lambda, alpha))) {
      return(list(Score = -1e5, Pred = 0))
    }
    if (max_depth <= 0 || !is.finite(eta) || !is.finite(gamma) || !is.finite(subsample)) {
      return(list(Score = -1e5, Pred = 0))
    }

    # Run CV, three folds are sufficient to rank candidate configurations
    cv <- tryCatch({
      xgb.cv(
        data = dtrain,
        nrounds = max_nrounds,
        nfold = 3,
        early_stopping_rounds = 50,
        objective = "reg:squarederror",
        eval_metric = "rmse",
        tree_method = "hist",
        seed = random_state,
        max_depth = max_depth,
        eta = eta,
        gamma = gamma,
        min_child_weight = min_child_weight,
        subsample = subsample,
        colsample_bytree = colsample_bytree,
        lambda = lambda,
        alpha = alpha,
        verbose = 0
      )
    }, error = function(e) NULL)

    # Handle failed CVs
    if (is.null(cv) || is.null(cv$evaluation_log)) {
      return(list(Score = -1e5, Pred = 0))
    }

    best_rmse <- min(cv$evaluation_log$test_rmse_mean, na.rm = TRUE)

    if (!is.finite(best_rmse)) {
      return(list(Score = -1e5, Pred = 0))
    }

    # Track the running best and log progress every log_every evaluations so it is easy to
    # see how quickly CV RMSE stabilizes during the search (current vs best-so-far).
    # Use message() (stderr) not cat() (stdout): BayesianOptimization wraps each evaluation in
    # utils::capture.output(), which redirects stdout and would otherwise swallow the log.
    if (best_rmse < best_rmse_so_far) best_rmse_so_far <<- best_rmse
    if (eval_counter %% log_every == 0) {
      message(sprintf("  [%s] iter %3d | current RMSE: %.3f | best RMSE: %.3f",
                      position, eval_counter, best_rmse, best_rmse_so_far))
    }

    list(Score = -best_rmse, Pred = 0)
  }

  # Run Bayesian Optimization, seeding so the random initial configurations are reproducible per random_state
  set.seed(random_state)
  opt_result <- BayesianOptimization(
    FUN = xgb_cv_bayes,
    bounds = list(
      max_depth = c(2, 8),
      eta = c(0.005, 0.2),
      gamma = c(0, 0.15),
      min_child_weight = c(1, 12),
      subsample = c(0.5, 1.0),
      colsample_bytree = c(0.5, 0.95),
      lambda = c(0, 10),    # L2 regularization
      alpha = c(0, 3)       # L1 regularization
    ),
    init_points = init_points,
    n_iter = n_iter,
    acq = "ucb",          # Or ei depending on strategy
    kappa = 1.75,
    eps = 0.4,
    verbose = FALSE       # Per evaluation RMSE is printed inside the CV function instead
  )

  best_params <- opt_result$Best_Par
  cat("Best CV RMSE for", position, "model:", round(-opt_result$Best_Value, 3), "\n")

  # Train final model with best parameters
  #
  # The tree count is chosen by CV on the TRAINING split only. The previous version early-stopped
  # against the held-out test set and then scored on that same set, which let the model pick the
  # iteration that minimized test error - optimistically biasing the tuned model's reported RMSE.
  # The 80/20 split is unchanged (dtrain is the 80% training data); X_test now stays completely
  # untouched until the single evaluation at the end, so the comparison to the baseline is honest.
  set.seed(random_state)
  cv_final <- xgb.cv(
    data = dtrain,
    nrounds = max_nrounds,
    nfold = 3,
    early_stopping_rounds = 50,
    objective = "reg:squarederror",
    eval_metric = "rmse",
    tree_method = "hist",
    seed = random_state,
    max_depth = round(best_params[["max_depth"]]),
    eta = best_params[["eta"]],
    gamma = best_params[["gamma"]],
    min_child_weight = best_params[["min_child_weight"]],
    subsample = best_params[["subsample"]],
    colsample_bytree = best_params[["colsample_bytree"]],
    lambda = best_params[["lambda"]],
    alpha = best_params[["alpha"]],
    verbose = 0
  )
  # Read the best round straight from the CV log (which.min of mean test RMSE) rather than
  # cv_final$best_iteration, which is NULL/absent in some xgboost versions and yields a
  # length-zero nrounds. This matches the field the tuning loop already relies on.
  best_nrounds <- which.min(cv_final$evaluation_log$test_rmse_mean)
  # Print the CV-chosen tree count against the ceiling - if it sits near max_nrounds the model is
  # being truncated (eta too low / ceiling too tight), not converging on its own
  cat("Best nrounds for", position, "model:", best_nrounds, "of", max_nrounds, "max\n")

  # Fit on the full training split with the CV-chosen tree count - no watchlist, no early stopping,
  # so the test set plays no role in selecting the model
  final_model <- xgb.train(
    data = dtrain,
    nrounds = best_nrounds,
    max_depth = round(best_params[["max_depth"]]),
    eta = best_params[["eta"]],
    gamma = best_params[["gamma"]],
    min_child_weight = best_params[["min_child_weight"]],
    subsample = best_params[["subsample"]],
    colsample_bytree = best_params[["colsample_bytree"]],
    lambda = best_params[["lambda"]],
    alpha = best_params[["alpha"]],
    objective = "reg:squarederror",
    eval_metric = "rmse",
    tree_method = "hist",
    seed = random_state,
    verbose = 0
  )

  # Evaluate once on the untouched held-out test split
  preds <- predict(final_model, newdata = data.matrix(X_test))
  rmse <- sqrt(mean((preds - y_test)^2))
  mae <- mean(abs(preds - y_test))

  list(
    model = final_model,
    features = feature_cols,
    best_nrounds = best_nrounds,   # CV-chosen tree count, reused by generate_prediction_intervals
    rmse = rmse,
    mae = mae,
    baseline_rmse = baseline_rmse,
    baseline_mae = baseline_mae,
    predictions = data.frame(
      Player = pos_df$Player[-train_idx],
      Year = pos_df$Year[-train_idx],
      Actual = y_test,
      Predicted = preds
    ),
    best_params = best_params
  )
}

# Function to plot out model feature importance
plot_feature_importance <- function(model, feature_names, top_n = 10) {
  # Get importance scores from xgboost
  importance <- xgb.importance(model = model, feature_names = feature_names)

  # Tidy and select top_n
  plot_df <- importance %>%
    arrange(desc(Gain)) %>%
    slice_head(n = top_n)

  # Plot
  ggplot(plot_df, aes(x = reorder(Feature, Gain), y = Gain)) +
    geom_col(fill = "#4682B4") +
    coord_flip() +
    labs(
      title = "Top Feature Importances (XGBoost Gain)",
      x = "Feature",
      y = "Gain"
    ) +
    theme_minimal(base_size = 13)
}

# Function to take the trained model and use it to predict upcoming season results
predict_next_year <- function(model_object, pred_df) {
  # Use the trained feature set
  feature_cols <- model_object$features

  # Add missing features with 0s
  missing_features <- setdiff(feature_cols, colnames(pred_df))
  if (length(missing_features) > 0) {
    pred_df[missing_features] <- 0
  }

  # Ensure correct feature order and format
  X_pred <- pred_df[, feature_cols, drop = FALSE]
  X_pred_matrix <- data.matrix(X_pred)

  # Predict using the xgboost booster model
  predicted_points <- predict(model_object$model, newdata = X_pred_matrix)

  # Return dataframe with predictions, keeping player_id so same-named players stay distinct
  pred_df %>%
    select(player_id, Player, Year, Pos) %>%
    mutate(
      Predicted = predicted_points,
      Pred_Year = as.Date(paste0(as.numeric(format(Year, "%Y")) + 1, "-01-01"))
    ) %>%
    arrange(desc(Predicted))
}

# Function to learn each player's lower and upper outcome spread from his features
#
# Quantile regression at 0.05 / 0.50 / 0.95. These only set the SHAPE of the simulated outcome
# distribution - how far it reaches down versus up - and are never reported directly.
#
# Trained on the UNTRUNCATED frame: model_df's 35/25/15 points filter is a filter on the OUTCOME, so
# it removes the bust seasons a floor exists to represent. A missing next season is a realized ZERO.
#
# Parameters
# ----------
# model_object : list
#     A trained point model from train_position_model. Supplies the pruned feature set and tuned
#     hyperparameters, so no second Bayesian search is needed for what is only a shape parameter.
# alphas : numeric
#     Quantile levels, ascending. Predictions are sorted row-wise afterwards because the fits are
#     independent and cross for roughly 1 row in 2000.
#
# Returns
# -------
# list(alphas, models, features) - consumed by predict_shape().
#
# Notes
# -----
# nrounds is re-selected per alpha by CV, since the tree count minimising squared error is not the one
# minimising pinball loss at 0.05. `quantile_alpha` must be passed INSIDE `params` - as a top-level
# xgb.train argument it is silently ignored and every alpha returns the same model.
#
# Known quirk: points_next_year is lead(points, 1) over rows that exist only for seasons PLAYED, so a
# player who misses a full year has his outcome read from two years ahead.
train_shape_models <- function(model_object, train_df, position,
                               alphas = c(0.05, 0.50, 0.95),
                               max_nrounds = 1200,
                               random_state = 62820) {
  feature_cols <- model_object$features
  best_params <- model_object$best_params

  # Untruncated: keep the busts and treat a missing next season as a realized zero
  pos_df <- train_df %>%
    filter(
      (position == "WR" & Pos %in% c("WR", "TE")) |
        (position == "QB" & Pos == "QB") |
        (position == "RB" & Pos %in% c("RB", "FB"))
    ) %>%
    filter(G > 0) %>%
    mutate(outcome = coalesce(points_next_year, 0))

  dtrain <- xgb.DMatrix(data = data.matrix(pos_df[, feature_cols, drop = FALSE]),
                        label = pos_df$outcome)

  base_params <- list(
    objective = "reg:quantileerror",
    eval_metric = "quantile",
    tree_method = "hist",
    max_depth = round(best_params[["max_depth"]]),
    eta = best_params[["eta"]],
    gamma = best_params[["gamma"]],
    min_child_weight = best_params[["min_child_weight"]],
    subsample = best_params[["subsample"]],
    colsample_bytree = best_params[["colsample_bytree"]],
    lambda = best_params[["lambda"]],
    alpha = best_params[["alpha"]],
    seed = random_state
  )

  models <- lapply(alphas, function(a) {
    params_a <- c(base_params, list(quantile_alpha = a))
    set.seed(random_state)
    cv <- xgb.cv(params = params_a, data = dtrain, nrounds = max_nrounds, nfold = 3,
                 early_stopping_rounds = 40, verbose = 0)
    metric_col <- grep("^test_.*mean$", names(cv$evaluation_log), value = TRUE)[1]
    nr <- which.min(cv$evaluation_log[[metric_col]])
    cat("  shape q", a, " (", position, ") - nrounds: ", nr, "\n", sep = "")
    xgb.train(params = params_a, data = dtrain, nrounds = nr, verbose = 0)
  })

  list(alphas = alphas, models = models, features = feature_cols)
}

# Function to turn the shape models into each player's RELATIVE lower and upper half-width
#
# Each half is returned as a FRACTION of the shape model's own median, never as raw points: the shape
# models and the point model are trained on different populations, so their raw scales do not match and
# mixing them puts the band in the wrong place. The level comes from `Predicted` alone.
#
# Mirrors predict_next_year's missing-feature handling so a column absent from the prediction frame is
# filled with zeros rather than erroring. Row-wise sorting enforces q05 <= q50 <= q95.
predict_shape <- function(shape_models, pred_df) {
  feature_cols <- shape_models$features
  missing_features <- setdiff(feature_cols, colnames(pred_df))
  if (length(missing_features) > 0) {
    pred_df[missing_features] <- 0
  }
  X <- data.matrix(pred_df[, feature_cols, drop = FALSE])

  q <- vapply(shape_models$models, function(m) predict(m, newdata = X), numeric(nrow(X)))
  q <- t(apply(q, 1, sort))

  # A season total cannot be negative, but the regression will predict q05 < 0. Clamping keeps
  # r_lo <= 1, hence Floor >= 0.
  q <- pmax(q, 0)

  # Guards against a tiny denominator: a player whose q50 rounds to nothing would otherwise post an
  # enormous ceiling room and sweep the top of the board.
  q50_safe <- pmax(q[, 2], 1)
  list(r_lo = pmin((q[, 2] - q[, 1]) / q50_safe, 1),
       r_hi = pmin((q[, 3] - q[, 2]) / q50_safe, 4))
}

# Function to simulate possible player outcomes and turn them into floor/ceiling prediction intervals
#
# A player's simulated outcome law is
#   Predicted + epi_weight * epistemic deviation + his own asymmetrically-scaled outcome noise
#
# Each of n_bootstrap iterations resamples PLAYERS with replacement (a cluster bootstrap, so whole
# careers move together) and refits at the point model's fixed tree count. The spread of those refit
# predictions is the epistemic term; their out-of-bag residuals pool into the outcome SHAPE, which
# shape_models then stretch per player via his own half-widths.
#
# The mixture is evaluated EXACTLY, not sampled: the shape is discretized onto an equal-weight quantile
# grid and the reported percentiles come from the full n_bootstrap x n_grid cross-product. So a given
# random_state always reproduces the same bands, while a different one is a genuinely different refit -
# attr(result, "simulate_draws") emits draws for functionals the stored percentiles cannot answer.
#
# Parameters
# ----------
# n_bootstrap : integer
#     Cluster-bootstrap refits. Drives BOTH the epistemic spread and the size of the OOB pool.
# n_grid : integer
#     Points used to discretize the outcome shape. Quantiles are exact to O(1/n_grid).
# epi_weight : numeric
#     Scales the epistemic refit spread. It is SYMMETRIC, so it dilutes the per-player asymmetry the
#     shape models provide; 0 drops it entirely.
# min_lo_frac : numeric
#     Floors the lower half-width at this fraction of the upper one, so a player whose q05 sits at his
#     q50 still gets a two-sided band.
# random_state : numeric
#     Seeds the bootstrap resampling only. The reported quantiles are deterministic given the refits.
#
# Returns
# -------
# data.frame keyed on player_id with pred_mean (exactly `Predicted`), the percentile set,
# Floor/Ceiling, width/upside/downside, and the three level-neutral indices from add_upside_indices.
generate_prediction_intervals <- function(model_object, shape_models, train_df, pred_df, position,
                                           n_bootstrap = 30,
                                           random_state = 62820,
                                           n_grid = 2000,
                                           epi_weight = 1,
                                           min_lo_frac = 0.15,
                                           report_probs = c(0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95),
                                           floor_from = "pred_p20",
                                           ceiling_from = "pred_p80") {
  # Reuse the exact feature set, tuned hyperparameters, and CV-chosen tree count from the point model
  feature_cols <- model_object$features
  best_params <- model_object$best_params
  fixed_nrounds <- model_object$best_nrounds

  # Filter training data to the position group, mirroring train_position_model
  pos_df <- train_df %>%
    filter(
      (position == "WR" & Pos %in% c("WR", "TE")) |
        (position == "QB" & Pos == "QB") |
        (position == "RB" & Pos %in% c("RB", "FB"))
    ) %>%
    filter(!is.na(points_next_year), G > 0)

  # Build the training matrix, target, and the player grouping vector used for bootstrapping
  X_tr <- data.matrix(pos_df[, feature_cols, drop = FALSE])
  y_tr <- pos_df$points_next_year
  group_ids <- pos_df$player_id
  unique_players <- unique(group_ids)

  # Build the prediction matrix once, reusing predict_next_year's missing-feature handling
  pred_pos <- pred_df
  missing_features <- setdiff(feature_cols, colnames(pred_pos))
  if (length(missing_features) > 0) {
    pred_pos[missing_features] <- 0
  }
  X_pred <- data.matrix(pred_pos[, feature_cols, drop = FALSE])
  n_pred <- nrow(X_pred)

  # Band centre = the same full-data prediction carried downstream as `Predicted`, so the intervals
  # bracket the number on the draft sheet rather than the bootstrap mean
  point_pred <- predict(model_object$model, newdata = X_pred)

  # --- Pass 1: cluster-bootstrap refits at a FIXED tree count (no early stopping) ---
  # base_all[b, ] holds each refit's eval-player predictions (the epistemic spread); out-of-bag
  # (fitted, residual) pairs accumulate into one global pool that becomes the outcome shape.
  base_all <- matrix(NA_real_, nrow = n_bootstrap, ncol = n_pred)
  pool_fitted <- vector("list", n_bootstrap)
  pool_resid  <- vector("list", n_bootstrap)

  for (b in seq_len(n_bootstrap)) {
    set.seed(random_state + b)

    # Sample players with replacement, replicating each drawn player's rows by his draw count so a
    # duplicate draw actually counts twice
    boot_players <- sample(unique_players, size = length(unique_players), replace = TRUE)
    draw_counts <- table(boot_players)
    in_bag_idx <- unlist(
      lapply(names(draw_counts), function(pid) {
        rep(which(group_ids == pid), times = draw_counts[[pid]])
      }),
      use.names = FALSE
    )
    # Out-of-bag players (never drawn) feed the global residual pool
    oob_idx <- which(group_ids %in% setdiff(unique_players, boot_players))

    # Fixed tree count, no early stopping, so the OOB rows are never used for round selection and stay
    # honest for the residual pool
    dtrain <- xgb.DMatrix(data = X_tr[in_bag_idx, , drop = FALSE], label = y_tr[in_bag_idx])
    booster <- xgb.train(
      data = dtrain,
      nrounds = fixed_nrounds,
      max_depth = round(best_params[["max_depth"]]),
      eta = best_params[["eta"]],
      gamma = best_params[["gamma"]],
      min_child_weight = best_params[["min_child_weight"]],
      subsample = best_params[["subsample"]],
      colsample_bytree = best_params[["colsample_bytree"]],
      lambda = best_params[["lambda"]],
      alpha = best_params[["alpha"]],
      objective = "reg:squarederror",
      eval_metric = "rmse",
      tree_method = "hist",
      seed = random_state + b,
      verbose = 0
    )

    base_all[b, ] <- predict(booster, newdata = X_pred)

    # Accumulate this iteration's OOB fitted/residual pairs into the global pool
    if (length(oob_idx) > 0) {
      oob_preds <- predict(booster, newdata = X_tr[oob_idx, , drop = FALSE])
      pool_fitted[[b]] <- oob_preds
      pool_resid[[b]]  <- y_tr[oob_idx] - oob_preds
    }
  }

  # --- One global outcome SHAPE, stretched per player by his own learned half-widths ---
  # The pooled OOB residuals supply only the shape - three fitted quantiles cannot describe a whole
  # distribution, and the empirical tails are the realistic part. shape_models supply the two
  # half-widths, which is what gives each player his own asymmetry rather than his tier's.
  g_resid <- unlist(pool_resid, use.names = FALSE)
  global_centered <- g_resid - mean(g_resid)

  # Rescale each half of the residual pool by its own spread, giving a unit shape whose lower half
  # spans -1 and upper half +1. Re-multiplying by a player's d_lo / d_hi then reproduces his
  # conditional q05 / q50 / q95 while keeping the real empirical tail form.
  unit_shape <- function(r) {
    r <- r - median(r)
    lo <- median(r) - quantile(r, 0.05, names = FALSE)
    hi <- quantile(r, 0.95, names = FALSE) - median(r)
    if (!is.finite(lo) || lo <= 0) lo <- 1
    if (!is.finite(hi) || hi <= 0) hi <- 1
    ifelse(r < 0, r / lo, r / hi)
  }

  # Summarize the shape on an equal-weight grid at MIDPOINT probabilities (i - 0.5)/n_grid, so its
  # quantile function matches the pool's to O(1/n_grid) with no sampling involved
  grid_probs <- (seq_len(n_grid) - 0.5) / n_grid
  to_grid <- function(r) {
    if (length(r) == 0) return(NULL)
    as.numeric(quantile(r, probs = grid_probs, names = FALSE, type = 7))
  }
  z_grid <- to_grid(unit_shape(global_centered))

  # Relative half-widths scaled onto the player's own projection, which puts both halves on the same
  # scale as the number they are measured against. A fringe player gets r_lo ~ 1 so his floor reaches
  # zero; a stud gets r_lo ~ 0.76 so his lands near a quarter of his projection. Asymmetry is unchanged
  # by the rescaling, since d_hi / d_lo = r_hi / r_lo.
  shape <- predict_shape(shape_models, pred_pos)
  d_hi <- pmax(shape$r_hi * point_pred, 1e-6)
  d_lo <- pmax(shape$r_lo * point_pred, min_lo_frac * d_hi)

  # --- Pass 2: exact quantiles of the predictive mixture, no sampling ---
  # The mixture over the n_bootstrap epistemic shifts of the outcome grid IS the full cross-product
  # (n_bootstrap x n_grid equal-weight points), so quantile() on it is exact - no draw noise, and two
  # players with identical inputs get identical bands.
  ensemble_mean <- colMeans(base_all)
  probs_out <- sort(unique(report_probs))
  prob_names <- sprintf("pred_p%02d", round(probs_out * 100))
  q_mat <- matrix(NA_real_, nrow = length(probs_out), ncol = n_pred,
                  dimnames = list(prob_names, NULL))

  # Per-player outcome grid: the shared shape stretched DOWN by d_lo and UP by d_hi, anchored on its
  # MEDIAN. Resolved once here so the quantile pass and simulate_draws() cannot disagree.
  #
  # Median-anchoring is load-bearing: the MEAN of an asymmetrically-stretched grid depends on both
  # half-widths, so subtracting it would let a large d_hi drag the floor down and re-couple the two
  # edges. Anchoring on the median keeps q05 a function of d_lo alone and q95 of d_hi alone.
  # Consequence: `Predicted` is the MEDIAN of the simulated law, so its mean sits above it when the
  # player is right-skewed.
  player_grid <- function(j) {
    if (is.null(z_grid)) return(NULL)
    g <- ifelse(z_grid < 0, z_grid * d_lo[j], z_grid * d_hi[j])
    g - median(g)
  }

  for (j in seq_len(n_pred)) {
    # The refit spread is symmetric, so it dilutes the shape models' asymmetry - epi_weight scales it
    epi_dev <- epi_weight * (base_all[, j] - ensemble_mean[j])
    grid_j <- player_grid(j)
    vals <- if (is.null(grid_j)) {
      point_pred[j] + epi_dev
    } else {
      point_pred[j] + as.numeric(outer(epi_dev, grid_j, "+"))
    }
    # The epistemic term can still push draws under zero even with the shape ratios clamped
    q_mat[, j] <- quantile(pmax(vals, 0), probs = probs_out, names = FALSE, type = 7)
  }

  # Draws from the same law the quantiles above describe, for functionals the stored percentiles cannot
  # answer - P(top-12 finish), P(bust below X). Not used by reporting.
  simulate_draws <- function(j, n_draws = 10000, seed = random_state) {
    set.seed(seed + j)
    epi_dev <- epi_weight * (base_all[, j] - ensemble_mean[j])
    grid_j <- player_grid(j)
    noise <- if (is.null(grid_j)) 0 else sample(grid_j, n_draws, replace = TRUE)
    pmax(point_pred[j] + sample(epi_dev, n_draws, replace = TRUE) + noise, 0)
  }

  bands <- data.frame(
    player_id = pred_pos$player_id,
    Player = pred_pos$Player,
    Pos = pred_pos$Pos,
    # The band is median-anchored on this, so pred_mean is exactly `Predicted` and also the simulated
    # law's median. Named pred_mean for the downstream contract, not because it is the law's mean.
    pred_mean = point_pred,
    stringsAsFactors = FALSE
  )
  bands <- cbind(bands, as.data.frame(t(q_mat)))   # one column per reported percentile

  attr(bands, "simulate_draws") <- simulate_draws
  add_upside_indices(bands, floor_from = floor_from, ceiling_from = ceiling_from)
}

# Standardize a vector to mean 100 / sd 15 (IQ/wRC+-style); flat 100 if it has no spread.
# These three helpers are top-level, not nested, so the diagnostics in tests/ can reach them -
# import_from_script there keeps only top-level `name <- function(...)` expressions.
index_100 <- function(x) {
  s <- sd(x, na.rm = TRUE)
  if (is.na(s) || s == 0) rep(100, length(x)) else 100 + 15 * (x - mean(x, na.rm = TRUE)) / s
}

# Level-adjust a signal into a studentized residual against the projection level, equalizing BOTH its
# average and its spread by level so no projection tier is favored in the centre or the tail. A
# mean-only detrend leaves the higher-variance tier over-populating the top. smooth() fits
# value-vs-level (loess, linear fallback for tiny pools); centre = residual, scale = smooth of
# |residual|. Vectors, not data-mask columns, so loess behaves inside mutate.
level_adjust <- function(x, level) {
  smooth <- function(y) {
    df <- data.frame(y = y, level = level)
    fit <- tryCatch(
      if (sum(is.finite(y) & is.finite(level)) >= 10) {
        loess(y ~ level, data = df, span = 0.75, na.action = na.exclude)
      } else {
        lm(y ~ level, data = df, na.action = na.exclude)
      },
      error = function(e) lm(y ~ level, data = df, na.action = na.exclude)
    )
    predict(fit, newdata = df)
  }
  resid <- x - smooth(x)                    # center: above/below typical for the level
  local_scale <- smooth(abs(resid))         # scale: typical |residual| at the level (~ conditional sd)
  # Floor the denominator at HALF the typical |residual|. The |residual| smooth collapses toward zero
  # at the sparse top of the projection range, where an ordinary residual would otherwise studentize
  # into an enormous value and inflate the position's sd. Load-bearing - do not loosen.
  floor_scale <- 0.50 * median(abs(resid), na.rm = TRUE)
  if (!is.finite(floor_scale) || floor_scale <= 0) floor_scale <- 1
  local_scale <- ifelse(is.finite(local_scale) & local_scale > floor_scale, local_scale, floor_scale)
  resid / local_scale
}

# Standardize the upside composite around the NEUTRAL point (100) rather than the sample mean.
#
# index_100 would put 100 at the mean of whatever it is given, and an OR-style combination has a mean
# well above 100 when its inputs are centred there - so a player above 100 on both axes could still land
# below 100 overall, which contradicts what the score is supposed to mean. Anchoring on 100 makes the
# input and output thresholds agree: both axes above 100 always scores above 100, and vice versa.
#
# Consequence: 100 means "exactly neutral on both axes", NOT "typical player". The median player scores
# above 100, because most players are above neutral on at least one axis.
index_at_neutral <- function(x, neutral = 100) {
  s <- sd(x, na.rm = TRUE)
  if (is.na(s) || s == 0) return(rep(neutral, length(x)))
  neutral + 15 * (x - neutral) / s
}

# Put an index on a tail-equalized scale, holding `neutral` fixed: rank -> normal score.
#
# ceiling_index and floor_index are both mean 100 / sd 15, but NOT the same SHAPE - ceiling_index is
# strongly left-skewed (-1.8 QB / -2.2 RB / -6.1 WR) while floor_index is right-skewed (+0.9 / +1.8 /
# +0.4). floor_index therefore owns the long UPPER tail, so adding raw deviations silently weights
# floor far heavier than ceiling (a raw sum comes out .77 correlated with floor_index and only .25
# with ceiling_index at WR). Mapping each axis through its own ECDF makes a +2sd ceiling exactly as
# rare as a +2sd floor, so the sum in add_upside_indices weighs them evenly.
#
# The qnorm(p0) subtraction is load-bearing. A plain rank-normal score puts its neutral point at the
# axis's MEDIAN, which is not 100 on a skewed axis (only ~30% of WR ceiling_index values sit below
# 100) - which breaks the "above 100 on both axes always scores above 100" guarantee for a handful of
# players. Anchoring on where 100 actually falls keeps the published axis columns and the composite
# agreeing about what neutral means.
rank_normal_at <- function(x, neutral = 100) {
  n <- sum(!is.na(x))
  if (n < 3) return(x)
  p  <- (rank(x, na.last = "keep") - 0.5) / n
  p0 <- mean(x < neutral, na.rm = TRUE)          # where `neutral` sits on this axis
  p0 <- min(max(p0, 1e-4), 1 - 1e-4)             # guard qnorm against an all-above / all-below axis
  neutral + 15 * (qnorm(p) - qnorm(p0))
}

# Function to attach the three level-normalized upside signals to a table of band edges
#
# Takes only pred_mean / pred_p05 / pred_p95, so it is indifferent to how the edges were produced.
#
# Parameters
# ----------
# bands_df : data.frame
#     Must carry pred_mean and the two percentile columns named below. Indices are standardized WITHIN
#     the frame passed in, so this must be called PER MODEL GROUP (QB / RB / WR+TE) exactly as
#     generate_prediction_intervals does - pooling positions would change every published index.
# floor_from, ceiling_from : character
#     Which percentile columns become Floor and Ceiling. p20/p80 by default rather than p05/p95: at the
#     5th percentile most players' floors pin near zero, which compresses floor_share and costs
#     floor_index its ability to separate them. Extreme quantiles are also the least precisely
#     estimated. Set to "pred_p05"/"pred_p95" for the wider reading.
#
# Returns
# -------
# bands_df plus Floor, Ceiling, pred_width, pred_upside, pred_downside, and the three indices.
add_upside_indices <- function(bands_df, floor_from = "pred_p20", ceiling_from = "pred_p80") {
  bands_df %>%
    mutate(
      Floor = .data[[floor_from]],
      Ceiling = .data[[ceiling_from]],
      pred_width = Ceiling - Floor,
      pred_upside = Ceiling - pred_mean,    # ceiling distance above the projection
      pred_downside = pred_mean - Floor,    # floor distance below the projection

      # Scale-free band edges relative to the projection. Intermediates only - each is detrended vs
      # level and standardized into an index below, then dropped.
      #   ceiling_room: how far ABOVE the projection the ceiling sits
      #   floor_share:  how CLOSE to the projection the floor sits
      ceiling_room = if_else(pred_mean > 0, Ceiling / pred_mean, NA_real_),
      floor_share  = if_else(pred_mean > 0, Floor   / pred_mean, NA_real_),

      # level_adjust makes each ratio level-NEUTRAL, then index_100 standardizes within position to
      # mean 100 / sd 15, so each answers "unusually high for a player projected this much?"
      ceiling_index = index_100(level_adjust(ceiling_room, pred_mean)),
      floor_index   = index_100(level_adjust(floor_share,  pred_mean)),

      # Single sortable score: a DEFICIT-WEIGHTED SUM of the two axes, on the tail-equalized scale.
      # Surpluses above 100 count in full, so a genuine spike on either axis still flags the player;
      # deficits count at UPSIDE_DEFICIT_WEIGHT, so a weak axis costs something without cancelling a
      # spike; and the weaker axis earns UPSIDE_BOTH_BONUS extra once it too clears 100.
      #
      # This replaced a pmax() over the two axes, which ignored the weaker one entirely unless it beat
      # 100 - Josh Allen scored 104 on a ceiling_index of 20. A plain sum (ceiling + floor - 200) over-
      # corrects in the other direction: the axes anti-correlate in the tails, so it cancelled exactly
      # the players it should flag (Tyreek Hill 149/61 fell from 170 to 107, Malik Nabers 136/47 to 87)
      # and degenerated into a floor sort. Damping the deficit keeps both specialist types on the board.
      #
      # Monotone increasing in BOTH axes (slope 1 above neutral, UPSIDE_DEFICIT_WEIGHT below), so a
      # player cannot be beaten by someone worse on both, and above 100 on both always scores above 100.
      #
      # Anchored at 100 rather than standardized to the mean - see index_at_neutral. The composite is
      # deliberately NOT level-adjusted: level_adjust is a relative transform, so it can reorder players
      # at different projections and would break the guarantee above. That costs some level neutrality,
      # which is the accepted trade for a score that means what it says.
      ceiling_n = rank_normal_at(ceiling_index),
      floor_n   = rank_normal_at(floor_index),
      upside_index = index_at_neutral(
        100 +
          pmax(ceiling_n - 100, 0) + pmax(floor_n - 100, 0) +
          UPSIDE_DEFICIT_WEIGHT * (pmin(ceiling_n - 100, 0) + pmin(floor_n - 100, 0)) +
          UPSIDE_BOTH_BONUS * pmax(pmin(ceiling_n, floor_n) - 100, 0)
      )
    ) %>%
    select(-ceiling_room, -floor_share, -ceiling_n, -floor_n)
}

# Function to display the anticipated "career trajectory" of players, combining historical results with forecasted performance
plot_predicted_trajectories <- function(combined_df, pred_df, pos_group = "QB", sample_n = 8, tier = 1) {
  # Dynamically create prediction year as date
  pred_year <- as.Date(paste0(PRED_YEAR, "-01-01"))
  eval_year <- as.Date(paste0(EVAL_YEAR, "-01-01"))

  # Historical data
  hist_df <- combined_df %>%
    filter(Pos == pos_group) %>%
    select(Player, Year, points)

  # Predicted values
  preds <- pred_df %>%
    filter(Pos == pos_group) %>%
    select(Player, Pred_Year, Predicted) %>%
    rename(Year = Pred_Year, points = Predicted)

  # Combine both
  full_df <- bind_rows(hist_df, preds)

  # Tier = a contiguous slice of players ranked by predicted value, sample_n players per tier.
  # tier 1 -> ranks 1..sample_n, tier 2 -> ranks (sample_n+1)..(2*sample_n), and so on. This keeps
  # each plot to a similar-value band (the y-axis isn't distorted by mixing a star with a deep
  # backup) and is deterministic, so a given tier always shows the same players.
  ranked_players <- pred_df %>%
    filter(Pos == pos_group) %>%
    arrange(desc(Predicted)) %>%
    mutate(rank = row_number())

  start_rank <- (tier - 1) * sample_n + 1
  end_rank <- tier * sample_n

  sampled_players <- ranked_players %>%
    filter(rank >= start_rank, rank <= end_rank) %>%
    pull(Player)

  if (length(sampled_players) == 0) {
    stop(sprintf("No %s players in tier %d (ranks %d-%d); only %d ranked players available.",
                 pos_group, tier, start_rank, end_rank, nrow(ranked_players)))
  }

  plot_df <- full_df %>%
    filter(Player %in% sampled_players) %>%
    # Order players by predicted rank (highest first) so the color gradient and legend run from the
    # top-predicted player to the bottom-predicted player
    mutate(Player = factor(Player, levels = sampled_players))

  # Split each trajectory into a solid historical leg and a dashed eval->prediction leg.
  # Both include the eval-year point, so the dashed projection connects seamlessly to the line.
  hist_lines <- plot_df %>% filter(Year <= eval_year)
  pred_lines <- plot_df %>% filter(Year >= eval_year)

  # 5. Labels at final year (projection year)
  label_df <- plot_df %>%
    group_by(Player) %>%
    slice_max(Year, n = 1, with_ties = FALSE) %>%
    ungroup()

  # Hiroshige gradient: map the player ordering (highest predicted -> lowest) onto the MetBrewer
  # "Hiroshige" palette, interpolated continuously across however many players the tier holds, so the
  # top-predicted player anchors one end of the ramp and the bottom-predicted player the other.
  gradient_colors <- setNames(
    rev(met.brewer("Hiroshige", n = length(sampled_players), type = "continuous")),
    sampled_players
  )

  ggplot(plot_df, aes(x = Year, y = points, color = Player, group = Player)) +
    # Solid historical trajectory, then a dashed leg into the prediction year
    geom_line(data = hist_lines, linewidth = 0.7, alpha = 0.7) +
    geom_line(data = pred_lines, linewidth = 0.7, alpha = 0.7, linetype = "dashed") +
    geom_text_repel(
      data = label_df,
      aes(label = Player),
      size = 4,
      hjust = 0,
      nudge_x = 0.05,
      direction = "y",
      segment.color = "#cccccc",
      segment.size = 0.2,
      family = "sans",
      force = 1,
      max.overlaps = Inf
    ) +
    scale_x_date(expand = expansion(mult = c(0.01, 0.2)), date_breaks = "1 year", date_labels = "%Y") +
    coord_cartesian(clip = "off") +
    scale_color_manual(values = gradient_colors) +
    labs(
      title = paste0(pos_group, " Career Trajectories + ", format(pred_year, "%Y"),
                     " Predictions (Tier ", tier, ": ranks ", start_rank, "-", end_rank, ")"),
      x = "Season",
      y = "Fantasy Points"
    ) +
    theme_minimal(base_size = 13) +
    theme(
      plot.title = element_text(colour = "#262626", size = 16, face = "bold", hjust = 0.5),
      axis.title = element_text(colour = "#262626", size = 14),
      axis.text = element_text(colour = "#262626", size = 12),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      legend.position = "none",
      panel.grid.minor = element_blank(),
      # Faint yearly vertical guides, plus light horizontal guides for reading point values
      panel.grid.major.x = element_line(color = "#E6E6E6", linewidth = 0.3),
      panel.grid.major.y = element_line(color = "#EFEFEF", linewidth = 0.3),
      # Dark "L" shaped axes along the left and bottom
      axis.line.x = element_line(color = "#4D4D4D", linewidth = 0.5),
      axis.line.y = element_line(color = "#4D4D4D", linewidth = 0.5)
    )
}

# Slope (bump) chart of projected rank movement for a position: last season's positional rank
# (by eval-year fantasy points) vs the model's predicted rank (by Predicted next-year points).
# Each player is a line between the two rank columns - risers slope toward a better (lower) rank,
# droppers toward a worse one. Players are tiered by predicted rank: tier 1 = the top
# players_per_tier predicted, tier 2 = the next block, and so on. Only players ranked in both
# seasons can move, so rookies (no last-season rank) and departures (no prediction) are excluded.
# Names use repelling so labels stay legible even where outliers squish the axis.
plot_rank_movement <- function(combined_df, pred_df, pos_group = "QB", tier = 1, players_per_tier = 20) {
  eval_year <- as.Date(paste0(EVAL_YEAR, "-01-01"))

  # Last season's positional rank from actual eval-year fantasy points (rank 1 = most points)
  last_rank <- combined_df %>%
    filter(Pos == pos_group, Year == eval_year, points > 0) %>%
    distinct(Player, points) %>%
    mutate(last_rank = dense_rank(desc(points))) %>%
    select(Player, last_rank)

  # Predicted positional rank from the model's next-year point forecast
  pred_rank <- pred_df %>%
    filter(Pos == pos_group) %>%
    mutate(pred_rank = dense_rank(desc(Predicted))) %>%
    select(Player, pred_rank)

  # Keep only players ranked in both seasons, then score the move (positive = rose up the board)
  movers <- inner_join(last_rank, pred_rank, by = "Player") %>%
    mutate(
      rank_change = last_rank - pred_rank,
      direction = case_when(
        rank_change > 0 ~ "Riser",
        rank_change < 0 ~ "Dropoff",
        TRUE ~ "Flat"
      )
    )

  # Slice the requested tier: a contiguous block of players_per_tier players ordered by predicted rank
  start_rank <- (tier - 1) * players_per_tier + 1
  end_rank <- tier * players_per_tier
  movers_sel <- movers %>%
    filter(pred_rank >= start_rank, pred_rank <= end_rank)

  if (nrow(movers_sel) == 0) {
    stop(sprintf("No %s players in tier %d (predicted ranks %d-%d).",
                 pos_group, tier, start_rank, end_rank))
  }

  # Long form: two rows per player (one per rank column) to draw the connecting slope line
  slope_df <- movers_sel %>%
    select(Player, direction, last_rank, pred_rank) %>%
    pivot_longer(c(last_rank, pred_rank), names_to = "stage", values_to = "rank") %>%
    mutate(x = if_else(stage == "last_rank", 1, 2))

  ggplot(slope_df, aes(x = x, y = rank, group = Player, color = direction)) +
    # Reference line under each rank column
    geom_vline(xintercept = c(1, 2), color = "#D9D9D9", linewidth = 0.4) +
    geom_line(linewidth = 0.45, alpha = 0.85) +
    geom_point(size = 1.5) +
    # Player + rank labels, repelled vertically within their column so squished outliers stay legible.
    # direction = "y" keeps each label in its own column; nudge_x pushes it off the dots.
    geom_text_repel(
      data = filter(slope_df, x == 1),
      aes(label = paste0(Player, " (", rank, ")")),
      hjust = 1, nudge_x = -0.05, direction = "y",
      size = 3.2, segment.color = "#cccccc", segment.size = 0.2,
      min.segment.length = 0, box.padding = 0.2, max.overlaps = Inf, show.legend = FALSE
    ) +
    geom_text_repel(
      data = filter(slope_df, x == 2),
      aes(label = paste0("(", rank, ") ", Player)),
      hjust = 0, nudge_x = 0.05, direction = "y",
      size = 3.2, segment.color = "#cccccc", segment.size = 0.2,
      min.segment.length = 0, box.padding = 0.2, max.overlaps = Inf, show.legend = FALSE
    ) +
    # Rank 1 sits at the top; ranks are read off the player labels rather than the y-axis
    scale_y_reverse() +
    scale_x_continuous(
      breaks = c(1, 2), labels = c("Last Season", "Predicted"),
      limits = c(0.4, 2.6)
    ) +
    scale_color_manual(
      values = c(Riser = "#418161", Dropoff = "#892938", Flat = "#666666")
    ) +
    labs(
      title = paste0(pos_group, " Projected Rank Movement — Last Season vs ", PRED_YEAR, " Prediction"),
      subtitle = paste0("Tier ", tier, ": predicted ranks ", start_rank, "-", end_rank),
      x = NULL, y = NULL, color = NULL
    ) +
    theme_minimal(base_size = 13) +
    theme(
      plot.title = element_text(colour = "#262626", size = 15, face = "bold"),
      plot.subtitle = element_text(colour = "#595959", size = 11),
      axis.text.x = element_text(colour = "#262626", size = 12, face = "bold"),
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      legend.position = "top",
      panel.grid = element_blank()
    )
}

# === DATA IMPORT === #
# Player data is sourced from the nflverse ecosystem via nflreadr, replacing the retired PFR scraper
# See build_player_season_stats in functions.R for column mapping notes and substitution decisions
# Building the output path with the year range baked into the filename
stats_path <- sprintf("data/player_stats_final_%d_%d.csv", START_YEAR, EVAL_YEAR)

if (SKIP_DATA_LOAD) {
  # Loading the prepared dataset directly from the data directory
  player_stats_final <- fread(stats_path)
} else {
  # Refreshing the full player season dataset from nflverse and caching it for future runs
  player_stats_final <- build_player_season_stats(START_YEAR, EVAL_YEAR)
  fwrite(player_stats_final, stats_path)
}

#### Full dataset containing each player & statistical category
# The legacy traded player cleanup and duplicate name fixes are no longer needed,
# nflverse season summaries arrive pre-aggregated and keyed by the GSIS player id
combined <-
  player_stats_final %>%
  mutate(points = (receiving_td * 6) + (receiving_yds * .1) + (Rec * PPR_MULT) +
           (rush_td * 6) + (rush_yds * .1) +
           (passing_yards * .04) + (passing_td * 4) -
           (rush_fbl * 2) - (passing_int * 2)) %>%
  arrange(player_id, Year) %>%
  group_by(player_id) %>%
  mutate(Pos = last(Pos)) %>% # Each player's most recent position will be used for their historical performance eval
  ungroup() %>%
  mutate(Year = as.Date(as.yearmon(Year))) %>%
  filter(Pos %in% c('WR', 'TE', 'RB', 'QB'))

# Trailing index of a player's best season up to (and including) each row.
# Used so "years since peak" only ever looks backward; a whole-career which.max()
# would leak future seasons (including next year's target) into the feature.
running_argmax <- function(x) {
  best_i <- 1L
  best_v <- -Inf
  out <- integer(length(x))
  for (i in seq_along(x)) {
    if (!is.na(x[i]) && x[i] > best_v) {
      best_v <- x[i]
      best_i <- i
    }
    out[i] <- best_i
  }
  out
}

# Coefficient of variation (sd / mean) for a trailing window - used as a realized consistency
# measure. Returns NA when the window has no spread to report (single value or zero mean) so the
# downstream NA->0 cleanup can take over.
coef_var <- function(x) {
  m <- mean(x, na.rm = TRUE)
  if (length(x) < 2 || is.na(m) || m == 0) return(NA_real_)
  sd(x, na.rm = TRUE) / m
}

# OLS slope of a trailing window against its position index (1, 2, 3, ...). Captures multi-year
# trajectory direction more stably than a single-year delta. NA until at least two points exist.
trailing_slope <- function(x) {
  if (length(x) < 2) return(NA_real_)
  t <- seq_along(x)
  cov(t, x) / var(t)
}

# Feature engineering
combined <-
  combined %>%
  arrange(player_id, Year) %>%
  group_by(player_id) %>%
  arrange(Year) %>%
  mutate(
    year_num = year(Year),
    # Identifying when each player entered the league, preferring the roster entry year when available
    rookie_year = coalesce(entry_year, min(year_num)),
    # Flagging veterans whose early career seasons predate the start of the dataset
    missing_pre_2006 = if_else(rookie_year < START_YEAR, 1, 0),
    estimated_rookie_year = rookie_year,
    # How many years of a player's career are missing from the data?
    num_missing_years = pmax(START_YEAR - rookie_year, 0),

    # Draft capital that boosts highly drafted players early then fades out, replacing raw
    # draft_number so pedigree stops influencing predictions past the rookie-contract window.
    # Base value is exponential by pick (1st overall best, undrafted ~0); a linear decay drops
    # it 25% per league year and pins it to 0 once a player is 4+ years in.
    years_in_league = pmax(0, year_num - estimated_rookie_year),
    draft_capital = exp(-(draft_number - 1) / 40) * pmax(0, 1 - 0.25 * years_in_league),

    # Adding an age squared feature, to capture the potential non-linear relationship between age and performance
    age_sq = Age^2,

    # Adding feature that tracks number of teams played for in a player's career
    num_teams_prior = lag(cumsum(!duplicated(Team)), default = 0),

    # Lag-based features
    points_last_year = lag(points, 1),
    age_points_interaction = Age * points_last_year,
    points_delta = points - lag(points, 1),
    points_pct_change = (points - lag(points, 1)) / lag(points, 1),
    consecutive_decline = if_else(points < lag(points, 1) & lag(points, 1) < lag(points, 2), 1, 0),

    # Creating a "productivity score" that will evaluate players based on how they are performing respective to their age
    adjusted_productivity = (points / age_sq),
    adjusted_productivity_trend = adjusted_productivity - lag(adjusted_productivity),
    adjusted_productivity_3yr = rollapplyr(adjusted_productivity, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),

    # TD regression / "TD luck": touchdowns are noisier than yardage and regress year to year, so
    # TDs per yard flags seasons whose scoring outran the underlying usage (a sell-high signal)
    receiving_td_rate = if_else(receiving_yds > 0, receiving_td / receiving_yds, 0),
    rushing_td_rate = if_else(rush_yds > 0, rush_td / rush_yds, 0),

    # Role-trend momentum: a change in opportunity share tends to lead the change in production
    target_share_delta = target_share - lag(target_share, 1),
    wopr_trend = wopr - lag(wopr, 1),

    # Realized consistency (backward-looking complement to the model's forward upside_index):
    # coefficient of variation of points over the trailing three seasons - low = steady/reliable
    points_cv_3yr = rollapplyr(points, width = 3, FUN = coef_var, fill = NA, align = "right", partial = TRUE),

    # Multi-year trajectory: OLS slope of points over the trailing three seasons, a steadier read
    # of momentum than a single-year delta (climbing vs spiking vs fading)
    points_trend_3yr = rollapplyr(points, width = 3, FUN = trailing_slope, fill = NA, align = "right", partial = TRUE),

    # Youth x opportunity: young players already commanding volume are the classic breakout profile
    youth_opportunity = if_else(Age > 0, wopr / Age, 0),

    # Creating a targets + games played feature, to show overall player involvement
    adjusted_targets = (Tgt * G),
    targets_per_game = (Tgt / G),
    touches = rush_att + Rec,
    points_per_target = (points / replace_na(Tgt, 1)),
    points_per_touch = (points / replace_na(touches, 1)),

    # Cumulative stats up to (but not including) current season
    career_games = cumsum(replace_na(G, 0)) - replace_na(G, 0),
    career_adjusted_productivity = cumsum(replace_na(adjusted_productivity, 0)) - replace_na(adjusted_productivity, 0),

    career_touches = cumsum(replace_na(touches, 0)) - replace_na(touches, 0),
    career_rushing_yds = cumsum(replace_na(rush_yds, 0)) - replace_na(rush_yds, 0),
    career_receiving_yds = cumsum(replace_na(receiving_yds, 0)) - replace_na(receiving_yds, 0),
    career_passing_yards = cumsum(replace_na(passing_yards, 0)) - replace_na(passing_yards, 0),

    career_rushing_td = cumsum(replace_na(rush_td, 0)) - replace_na(rush_td, 0),
    career_receiving_td = cumsum(replace_na(receiving_td, 0)) - replace_na(receiving_td, 0),
    career_passing_td = cumsum(replace_na(passing_td, 0)) - replace_na(passing_td, 0),

    # Rolling averages over a trailing three year window for major statistical categories
    avg_games_3yr = rollapplyr(G, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    avg_points_3yr = rollapplyr(points, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    avg_rushing_yds_3yr = rollapplyr(rush_yds, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    avg_receiving_yds_3yr = rollapplyr(receiving_yds, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    avg_passing_yds_3yr = rollapplyr(passing_yards, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    avg_passing_int_3yr = rollapplyr(passing_int, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),

    avg_rushing_td_3yr = rollapplyr(rush_td, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    avg_receiving_td_3yr = rollapplyr(receiving_td, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    avg_passing_td_3yr = rollapplyr(passing_td, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),

    # Adding metrics for QB efficiency stats
    rate_per_attempt = Rate / passing_att,
    td_int_ratio = if_else(passing_int > 0, passing_td / passing_int, NA_real_),
    # Share of fantasy points coming from rushing - isolates the mobile/Konami-code QB archetype
    # (the main driver of fantasy QB1 value), which is very sticky year over year
    rush_pts_share = if_else(points > 0, (rush_yds * 0.1 + rush_td * 6) / points, 0),
    # Pass attempts per game - a volume / every-week-starter proxy (QB fantasy is bimodal on playing time)
    pass_att_per_game = if_else(G > 0, passing_att / G, 0),
    qb_yards = passing_yards + rush_yds,
    qb_yards_3yr = rollapplyr(qb_yards, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    avg_qbr_3yr = rollapplyr(QBR, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    passing_epa_per_att_3yr = rollapplyr(passing_epa_per_att, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    # Season-total passing EPA - a volume-inclusive production signal complementing the per-attempt rate
    passing_epa_3yr = rollapplyr(passing_epa, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    passing_adj_net_yards_att_3yr = rollapplyr(passing_adj_net_yards_att, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    # Trailing 3yr OLS slope of ANY/A - a QB efficiency-trajectory signal (rising vs. fading arm/decisions)
    passing_adj_net_yards_att_trend_3yr = rollapplyr(passing_adj_net_yards_att, width = 3, FUN = trailing_slope, fill = NA, align = "right", partial = TRUE),
    passing_yards_att_3yr = rollapplyr(passing_yards_att, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    passing_cpoe_3yr = rollapplyr(passing_cpoe, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    pacr_3yr = rollapplyr(pacr, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    passing_adot_3yr = rollapplyr(passing_adot, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),

    # Rushing efficiency metrics, EPA based rates replace the legacy PFR success rate
    rush_efficiency = rush_yds * (1 + rush_epa_per_att),
    # Yards gained beyond a baseline expectation per carry, serving as an explosiveness proxy
    explosive_yards_proxy = rush_yds - (rush_att * 4),
    adj_ypa = rush_yds_att * (1 + rush_epa_per_att),
    fbl_per_att = rush_fbl / rush_att,
    yards_from_scrimmage = rush_yds + receiving_yds,
    yards_from_scrimmage_3yr = rollapplyr(yards_from_scrimmage, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),

    # Receiving efficiency metrics
    explosive_catch_rate = if_else(Rec > 0, receiving_1D / Rec, 0),
    explosive_receiving_eff = if_else(receiving_yds > 100, catch_percent * receiving_yards_target, 0),

    # Adding a rolling avg for rushing/receiving efficiency stats
    rush_epa_per_att_3yr = rollapplyr(rush_epa_per_att, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    # Trailing 3yr OLS slope of rushing EPA/att - a rushing-efficiency trajectory signal (athletic gain/decline)
    rush_epa_per_att_trend_3yr = rollapplyr(rush_epa_per_att, width = 3, FUN = trailing_slope, fill = NA, align = "right", partial = TRUE),
    # Season-total rushing EPA - a volume-inclusive production signal complementing the per-attempt rate
    rushing_epa_3yr = rollapplyr(rushing_epa, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    touches_3yr = rollapplyr(touches, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    receiving_epa_per_target_3yr = rollapplyr(receiving_epa_per_target, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    # Season-total receiving EPA - a volume-inclusive production signal complementing the per-target rate
    receiving_epa_3yr = rollapplyr(receiving_epa, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    receiving_yards_target_3yr = rollapplyr(receiving_yards_target, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    # Trailing 3yr OLS slope of yards/target - a receiving-efficiency trajectory signal (separation/YAC gain/decline)
    receiving_yards_target_trend_3yr = rollapplyr(receiving_yards_target, width = 3, FUN = trailing_slope, fill = NA, align = "right", partial = TRUE),
    Tgt_3yr = rollapplyr(Tgt, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    catch_percent_3yr = rollapplyr(catch_percent, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    adjusted_targets_3yr = rollapplyr(adjusted_targets, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    targets_per_game_3yr = rollapplyr(targets_per_game, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    wopr_3yr = rollapplyr(wopr, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    target_share_3yr = rollapplyr(target_share, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    air_yards_share_3yr = rollapplyr(air_yards_share, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    racr_3yr = rollapplyr(racr, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),

    # General career info
    career_total_points = cumsum(replace_na(points, 0)) - replace_na(points, 0),
    seasons_played = row_number() - 1,
    # Seasons elapsed since the player's best year *so far* (trailing argmax, not whole-career
    # which.max) — 0 in a new career-best season, growing as a player moves past their prime.
    years_since_peak = row_number() - running_argmax(points),

    # Per-game efficiency
    avg_points_per_game = if_else(G > 0, points / G, NA_real_),
    avg_points_per_game_3yr = rollapplyr(avg_points_per_game, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),

    # Flagging whether a player is playing above their 3-year average in the recent season (rising star vs fading)
    points_vs_3yr_avg = points - avg_points_3yr,

    # Year-over-year volume/opportunity momentum. Rate deltas (per game) isolate a genuine role
    # change from games missed, unlike the raw points_delta, and are among the stickiest
    # breakout/regression signals. yards_from_scrimmage_delta is a strong RB/WR usage-swing predictor.
    avg_points_per_game_delta = avg_points_per_game - lag(avg_points_per_game, 1),
    touches_delta = touches - lag(touches, 1),
    pass_att_per_game_delta = pass_att_per_game - lag(pass_att_per_game, 1),
    yards_from_scrimmage_delta = yards_from_scrimmage - lag(yards_from_scrimmage, 1),

    # Opportunity vs established baseline: current share minus its trailing 3yr average, the usage
    # analog of points_vs_3yr_avg. A large positive gap flags an anomalous season that tends to
    # mean-revert; a sustained gap confirms a real role change.
    wopr_vs_3yr = wopr - wopr_3yr,
    target_share_vs_3yr = target_share - target_share_3yr,

    # Injury / performance dip flags
    games_last_year = lag(G, 1),
    touches_last_year = lag(touches, 1),
    # Was this player injured in the previous season?
    # Players with a mostly healthy prior season who missed substantial time this season are flagged as injured
    injured_last_year = if_else(games_last_year >= 8 & G < 6, 1, 0),

    # Did this player recover from injury last season?
    prior_injury_flag = lag(injured_last_year, 1),

    # Logged major career statistics to normalize skewed distributions
    log_career_total_points = log1p(career_total_points)

  ) %>%
  # Defining the target variable
  mutate(points_next_year = lead(points, 1)) %>%
  ungroup() %>%
  group_by(Year, Pos) %>%
  mutate(pos_rank = dense_rank(desc(points)), # current year rank
         top_finish_flag = case_when(
           Pos == "QB" & pos_rank <= 12 ~ 1,
           Pos == "RB" & pos_rank <= 24 ~ 1,
           Pos == "WR" & pos_rank <= 36 ~ 1,
           Pos == "TE" & pos_rank <= 12 ~ 1,
           TRUE ~ 0
         )) %>%
  ungroup() %>%
  group_by(player_id) %>%
  arrange(Year) %>%
  mutate(pos_rank_last_year = lag(pos_rank, 1), # prior year rank
         career_top_finish_count = cumsum(top_finish_flag)) %>%
  ungroup() %>%
  # Infinity/NA fixes apply to FEATURE columns only. points_next_year is the target and must stay NA
  # wherever the next season does not exist - the PRED_YEAR rows, and anyone who never played again -
  # or every downstream !is.na(points_next_year) filter becomes a no-op.
  mutate(across(where(is.numeric) & !any_of("points_next_year"), ~ ifelse(is.infinite(.), NA, .))) %>%
  mutate(across(where(is.numeric) & !any_of("points_next_year"), ~ replace_na(., 0)))

# Caching the engineered feature table so the diagnostics in tests/ can rebuild this exact modeling
# frame without re-running the data build
fwrite(combined, paste0("data/combined_features_", as.character(EVAL_YEAR), ".csv"))

# Filtering for players with extremely few points in the next year, these players would not be drafted regardless
# Removing the EVAL_YEAR data from the training set, as it is the evaluation year
model_df <-
  combined %>%
  filter(!is.na(points_next_year), G > 0) %>%
  filter(
    (Pos == "QB" & points_next_year > 35) |
      (Pos %in% c("RB", "WR") & points_next_year > 25) |
      (Pos == "TE" & points_next_year > 15)
  ) %>%
  filter(Year < as.Date(paste0(EVAL_YEAR, "-01-01")))

# The UNTRUNCATED training frame, used only by train_shape_models. model_df's points threshold filters
# on the OUTCOME, so it deletes the bust seasons a floor exists to represent. Same leakage cutoff as
# model_df; missing-next-season rows stay in and become a realized zero inside train_shape_models.
combined_train <-
  combined %>%
  filter(G > 0) %>%
  filter(Year < as.Date(paste0(EVAL_YEAR, "-01-01")))

# Creating the prediction dataframes for the evaluation year.
# NOTE: deliberately no points_next_year filter - by construction these rows have no next season yet,
# so the target is NA for all of them.
pred_df <-
  combined %>%
  filter(Year == as.Date(paste0(EVAL_YEAR, "-01-01")), G > 0)

# Splitting the prediction dataframe into positional groupings
qb_pred_df <- pred_df %>% filter(Pos == "QB")
rb_pred_df <- pred_df %>% filter(Pos %in% c("RB", "FB"))
wr_pred_df <- pred_df %>% filter(Pos %in% c("WR", "TE"))

qb_features <- c(
  "adj_ypa", "adjusted_productivity", "adjusted_productivity_3yr", "adjusted_productivity_trend",
  "Age", "age_points_interaction", "age_sq", "avg_games_3yr",
  "avg_passing_int_3yr", "avg_passing_td_3yr", "avg_passing_yds_3yr", "avg_points_3yr",
  "avg_points_per_game", "avg_points_per_game_3yr", "avg_points_per_game_delta", "avg_qbr_3yr",
  "avg_rushing_td_3yr", "avg_rushing_yds_3yr", "career_adjusted_productivity", "career_games",
  "career_passing_td", "career_passing_yards", "career_rushing_td", "career_rushing_yds",
  "career_top_finish_count", "career_total_points", "consecutive_decline", "draft_capital",
  "estimated_rookie_year", "G", "games_last_year", "injured_last_year",
  "missing_pre_2006", "num_missing_years", "num_teams_prior", "pacr",
  "pacr_3yr", "pass_att_per_game", "pass_att_per_game_delta", "passing_1D",
  "passing_adj_net_yards_att", "passing_adj_net_yards_att_3yr", "passing_adj_net_yards_att_trend_3yr", "passing_adot", "passing_adot_3yr",
  "passing_att", "passing_avg_yards_att", "passing_comp", "passing_comp_pct",
  "passing_cpoe", "passing_cpoe_3yr", "passing_epa_per_att", "passing_epa_per_att_3yr",
  "passing_epa", "passing_epa_3yr", "rushing_epa", "rushing_epa_3yr",
  "passing_int", "passing_int_pct", "passing_net_yards_att", "passing_sack",
  "passing_td", "passing_td_pct", "passing_yards", "passing_yards_att",
  "passing_yards_att_3yr", "passing_yards_comp", "passing_yards_game", "points",
  "points_cv_3yr", "points_delta", "points_last_year", "points_pct_change",
  "points_trend_3yr", "points_vs_3yr_avg", "pos_rank", "pos_rank_last_year",
  "prior_injury_flag", "qb_yards", "qb_yards_3yr", "QBR",
  "Rate", "rate_per_attempt", "rush_1D", "rush_att",
  "rush_attempts_per_game", "rush_efficiency", "rush_epa_per_att", "rush_epa_per_att_3yr",
  "rush_fbl", "rush_pts_share", "rush_td", "rush_yds",
  "rush_yds_att", "rush_yds_game", "rushing_td_rate", "sack_percent",
  "sack_yds", "seasons_played", "td_int_ratio", "Team",
  "top_finish_flag", "wins", "Year", "years_since_peak"
)

rb_features <- c(
  "adj_ypa", "adjusted_productivity", "adjusted_productivity_3yr", "adjusted_productivity_trend",
  "adjusted_targets", "adjusted_targets_3yr", "Age", "age_points_interaction",
  "age_sq", "avg_games_3yr", "avg_points_3yr", "avg_points_per_game",
  "avg_points_per_game_3yr", "avg_points_per_game_delta", "avg_receiving_td_3yr", "avg_receiving_yds_3yr",
  "avg_rushing_td_3yr", "avg_rushing_yds_3yr", "career_adjusted_productivity", "career_games",
  "career_receiving_td", "career_receiving_yds", "career_rushing_td", "career_rushing_yds",
  "career_top_finish_count", "career_total_points", "career_touches", "consecutive_decline",
  "draft_capital", "estimated_rookie_year", "explosive_catch_rate", "explosive_receiving_eff",
  "explosive_yards_proxy", "fbl_per_att", "G", "games_last_year",
  "injured_last_year", "missing_pre_2006", "num_missing_years", "num_teams_prior",
  "points", "points_cv_3yr", "points_delta", "points_last_year",
  "points_pct_change", "points_per_target", "points_per_touch", "points_trend_3yr",
  "points_vs_3yr_avg", "pos_rank", "pos_rank_last_year", "prior_injury_flag",
  "receiving_1D", "receiving_air_yards", "receiving_epa_per_target", "receiving_epa_per_target_3yr",
  "receiving_epa", "receiving_epa_3yr", "rushing_epa", "rushing_epa_3yr",
  "receiving_rec_g", "receiving_td", "receiving_td_rate", "receiving_y_g",
  "receiving_yards_after_catch", "receiving_yards_target", "receiving_yards_target_3yr", "receiving_yds",
  "receiving_yds_rec", "rush_1D", "rush_att", "rush_attempts_per_game",
  "rush_efficiency", "rush_epa_per_att", "rush_epa_per_att_3yr", "rush_epa_per_att_trend_3yr", "rush_fbl",
  "rush_td", "rush_yds", "rush_yds_att", "rush_yds_game",
  "rushing_td_rate", "seasons_played", "target_share", "target_share_3yr",
  "target_share_delta", "target_share_vs_3yr", "targets_per_game", "targets_per_game_3yr",
  "Team", "Tgt_3yr", "top_finish_flag", "touches",
  "touches_3yr", "touches_delta", "touches_last_year", "wopr",
  "wopr_3yr", "wopr_trend", "wopr_vs_3yr", "yards_from_scrimmage",
  "yards_from_scrimmage_3yr", "yards_from_scrimmage_delta", "Year", "years_since_peak",
  "youth_opportunity"
)

wr_features <- c(
  "adjusted_productivity", "adjusted_productivity_3yr", "adjusted_productivity_trend", "adjusted_targets",
  "adjusted_targets_3yr", "Age", "age_points_interaction", "age_sq",
  "air_yards_share", "air_yards_share_3yr", "avg_games_3yr", "avg_points_3yr",
  "avg_points_per_game", "avg_points_per_game_3yr", "avg_points_per_game_delta", "avg_receiving_td_3yr",
  "avg_receiving_yds_3yr", "avg_rushing_td_3yr", "avg_rushing_yds_3yr", "career_adjusted_productivity",
  "career_games", "career_receiving_td", "career_receiving_yds", "career_rushing_yds",
  "career_top_finish_count", "career_total_points", "catch_percent", "catch_percent_3yr",
  "consecutive_decline", "draft_capital", "estimated_rookie_year", "explosive_catch_rate",
  "explosive_receiving_eff", "explosive_yards_proxy", "G", "games_last_year",
  "injured_last_year", "log_career_total_points", "missing_pre_2006", "num_missing_years",
  "num_teams_prior", "points", "points_cv_3yr", "points_delta",
  "points_last_year", "points_pct_change", "points_per_target", "points_trend_3yr",
  "points_vs_3yr_avg", "Pos", "pos_rank", "pos_rank_last_year",
  "prior_injury_flag", "racr", "racr_3yr", "Rec",
  "receiving_1D", "receiving_air_yards", "receiving_epa_per_target", "receiving_epa_per_target_3yr",
  "receiving_epa", "receiving_epa_3yr", "rushing_epa", "rushing_epa_3yr",
  "receiving_rec_g", "receiving_td", "receiving_td_rate", "receiving_y_g",
  "receiving_yards_after_catch", "receiving_yards_target", "receiving_yards_target_3yr", "receiving_yards_target_trend_3yr", "receiving_yds",
  "receiving_yds_rec", "rush_att", "rush_epa_per_att", "rush_epa_per_att_3yr",
  "rush_fbl", "rush_td", "rush_yds", "rush_yds_att",
  "rush_yds_game", "seasons_played", "target_share", "target_share_3yr",
  "target_share_delta", "target_share_vs_3yr", "targets_per_game", "targets_per_game_3yr",
  "Team", "Tgt", "Tgt_3yr", "top_finish_flag",
  "touches", "wopr", "wopr_3yr", "wopr_trend",
  "wopr_vs_3yr", "yards_from_scrimmage", "yards_from_scrimmage_3yr", "yards_from_scrimmage_delta",
  "Year", "years_since_peak", "youth_opportunity"
)

# Creating models and making predictions for each major positional group
qb_model <- train_position_model(model_df, "QB", qb_features, init_points = 12, n_iter = 36, random_state = RANDOM_STATE)
plot_feature_importance(qb_model$model, qb_model$features, top_n = 25) +
  ggtitle("Quarterback Feature Importance")
qb_model_preds <- qb_model[['predictions']] %>%
  mutate(diff = Predicted - Actual) %>%
  arrange(diff)

rb_model <- train_position_model(model_df, "RB", rb_features, init_points = 12, n_iter = 36, random_state = RANDOM_STATE)
plot_feature_importance(rb_model$model, rb_model$features, top_n = 25) +
  ggtitle("Rushing Feature Importance")
rb_model_preds <- rb_model[['predictions']] %>%
  mutate(diff = Predicted - Actual) %>%
  arrange(diff)

# IMPORTANT: TEs will be included in the WR model by default
wr_model <- train_position_model(model_df, "WR", wr_features, init_points = 12, n_iter = 36, random_state = RANDOM_STATE)
plot_feature_importance(wr_model$model, wr_model$features, top_n = 25) +
  ggtitle("Receiving Feature Importance")
wr_model_preds <- wr_model[['predictions']] %>%
  mutate(diff = Predicted - Actual) %>%
  arrange(diff)

# Helper to report holdout performance for one position group,
# comparing the untuned random forest baseline against the tuned XGBoost model
report_holdout_performance <- function(model_object, pos_label) {
  rmse_improvement <- (model_object$baseline_rmse - model_object$rmse) / model_object$baseline_rmse * 100
  mae_improvement <- (model_object$baseline_mae - model_object$mae) / model_object$baseline_mae * 100

  cat(
    pos_label,
    "\n",
    "RMSE - baseline:", round(model_object$baseline_rmse, 2),
    "| model:", round(model_object$rmse, 2),
    paste0("| improvement: ", round(rmse_improvement, 1), "%"),
    "\n",
    "MAE - baseline:", round(model_object$baseline_mae, 2),
    "| model:", round(model_object$mae, 2),
    paste0("| improvement: ", round(mae_improvement, 1), "%"),
    "\n"
  )
}

# Evaluating model performance on the holdout split, baseline vs tuned model
report_holdout_performance(qb_model, "QB   ")
report_holdout_performance(rb_model, "RB   ")
report_holdout_performance(wr_model, "WR/TE")

# Rendering model diagnostics per position, run each plot as needed
# QB diagnostics
plot_actual_vs_pred(qb_model_preds, "QB", overperf_x = 125, underperf_x = 250, overperf_y = 325)
plot_resid_hist(qb_model_preds, "QB", binwidth = 10, band = 75)
plot_decile_calib(qb_model_preds, "QB")

# RB diagnostics
plot_actual_vs_pred(rb_model_preds, "RB", overperf_x = 100)
plot_resid_hist(rb_model_preds, "RB")
plot_decile_calib(rb_model_preds, "RB")

# WR/TE diagnostics
plot_actual_vs_pred(wr_model_preds, "WR/TE", overperf_x = 80)
plot_resid_hist(wr_model_preds, "WR/TE", binwidth = 5)
plot_decile_calib(wr_model_preds, "WR/TE")

# Making player predictions for the upcoming season
qb_preds <- predict_next_year(qb_model, qb_pred_df)
rb_preds <- predict_next_year(rb_model, rb_pred_df)
wr_preds <- predict_next_year(wr_model, wr_pred_df)

# Learning each player's outcome SHAPE - how far his season plausibly reaches down versus up based on historic similarity.
# NOTE: trained on combined_train (untruncated), not model_df, whose points threshold deletes the busts.
qb_shape <- train_shape_models(qb_model, combined_train, "QB", random_state = RANDOM_STATE)
rb_shape <- train_shape_models(rb_model, combined_train, "RB", random_state = RANDOM_STATE)
wr_shape <- train_shape_models(wr_model, combined_train, "WR", random_state = RANDOM_STATE)

# Generating the simulated floor/ceiling intervals per position (reuses each model's tuned params)
# NOTE: this fits n_bootstrap XGBoost models per position, lower n_bootstrap for a fast test run
qb_intervals <- generate_prediction_intervals(qb_model, qb_shape, model_df, qb_pred_df, "QB", random_state = RANDOM_STATE)
rb_intervals <- generate_prediction_intervals(rb_model, rb_shape, model_df, rb_pred_df, "RB", random_state = RANDOM_STATE)
wr_intervals <- generate_prediction_intervals(wr_model, wr_shape, model_df, wr_pred_df, "WR", random_state = RANDOM_STATE)
intervals_all <- bind_rows(qb_intervals, rb_intervals, wr_intervals)

# Visualizing projected rank movement vs last season, 20 players per tier by predicted rank
plot_rank_movement(combined, qb_preds, pos_group = "QB", tier = 1)
plot_rank_movement(combined, rb_preds, pos_group = "RB", tier = 1)
plot_rank_movement(combined, wr_preds, pos_group = "WR", tier = 1)
plot_rank_movement(combined, wr_preds, pos_group = "TE", tier = 1)

# Visualizing predicted player performance trajectories
plot_predicted_trajectories(combined, qb_preds, pos_group = "QB", tier = 1)
plot_predicted_trajectories(combined, rb_preds, pos_group = "RB", tier = 1)
plot_predicted_trajectories(combined, wr_preds, pos_group = "WR", tier = 1)
plot_predicted_trajectories(combined, wr_preds, pos_group = "TE", tier = 1)



# Saving out final combined dataframe. Joining the bootstrap floor/ceiling intervals on the unique
# player_id (not name) so players who share a name + position - e.g. the two Adrian Petersons -
# stay distinct instead of cross-matching. player_id is retained for downstream disambiguation.
final <-
  bind_rows(qb_preds, wr_preds, rb_preds) %>%
  mutate(Player = clean_player_name(Player)) %>% # Cleaning Player Names
  select(player_id, Player, Pos, Predicted) %>%
  left_join(
    intervals_all %>%
      select(player_id, Floor, Ceiling, pred_mean,
             pred_p05, pred_p10, pred_p20, pred_p50, pred_p80, pred_p90, pred_p95, pred_width,
             pred_downside, pred_upside, ceiling_index, floor_index, upside_index),
    by = "player_id"
  )

fwrite(final, paste0("data/model_pred_", as.character(PRED_YEAR), "_", SCORING_TYPE, ".csv"))
