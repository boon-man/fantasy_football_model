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
SKIP_TUNING <- FALSE    # Set to TRUE to reuse cached hyperparameters and skip Bayesian optimization

# TODO: Add in simluated prediction ranges, identify high-ceiling players
# TODO: Add specific prediction/projection blends by position. Model splits QB:50%, RB:40%, WR:60%
# TODO: Fix Career trajectories plot to look better 
# TODO: Add in additional features to improve model performance


# Function to train the XGBoost model for a specific position
#
# Tuning budget parameters, lower these for fast code-testing runs:
#   init_points : random configurations evaluated before the Bayesian search starts
#   n_iter      : Bayesian optimization iterations after initialization
#   max_nrounds : tree count ceiling for CV and the final fit, early stopping decides the actual count
train_position_model <- function(df, position, feature_cols,
                                 skip_tuning = SKIP_TUNING,
                                 init_points = 10,
                                 n_iter = 20,
                                 max_nrounds = 1000) {
  library(dplyr)
  library(xgboost)
  library(rBayesianOptimization)
  library(caret)
  library(ranger)

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
  set.seed(62820)
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

  set.seed(62820)
  baseline_model <- ranger(x = baseline_train, y = y_train, num.trees = 500)

  # Scoring the baseline on the holdout split
  baseline_preds <- predict(baseline_model, data = baseline_test)$predictions
  baseline_rmse <- sqrt(mean((baseline_preds - y_test)^2))
  baseline_mae <- mean(abs(baseline_preds - y_test))
  cat("Baseline RF holdout RMSE for", position, "model:", round(baseline_rmse, 3), "\n")

  # === TUNED MODEL === #
  # Tuned hyperparameters are cached per position so reruns can skip the optimization,
  # mirroring the SKIP_DATA_LOAD pattern, retune once per annual refresh
  params_path <- paste0("data/tuned_params_", position, "_", SCORING_TYPE, ".rds")

  if (skip_tuning && file.exists(params_path)) {
    cat("Loading cached hyperparameters from", params_path, "\n")
    best_params <- readRDS(params_path)
  } else {

    # Define Bayesian optimization function
    # The tree count is not part of the search space, early stopping inside the
    # CV finds the right number of rounds for each candidate configuration
    xgb_cv_bayes <- function(max_depth, eta, gamma, min_child_weight, subsample, colsample_bytree) {
      set.seed(82525)

      max_depth <- as.integer(round(max_depth))

      # Validate parameters
      if (anyNA(c(max_depth, eta, gamma, min_child_weight, subsample))) {
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
          max_depth = max_depth,
          eta = eta,
          gamma = gamma,
          min_child_weight = min_child_weight,
          subsample = subsample,
          colsample_bytree = colsample_bytree,
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

      # Reporting the evaluation metric by name rather than the generic Value label
      cat("  CV RMSE:", round(best_rmse, 3), "\n")

      list(Score = -best_rmse, Pred = 0)
    }

    # Run Bayesian Optimization
    opt_result <- BayesianOptimization(
      FUN = xgb_cv_bayes,
      bounds = list(
        max_depth = c(3, 7),
        eta = c(0.1, 0.3),
        gamma = c(0, 0.1),
        min_child_weight = c(0.05, 0.5),
        subsample = c(0.9, 1.0),
        colsample_bytree = c(0.7, 1.0)
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

    # Caching the winning hyperparameters for future runs
    saveRDS(best_params, params_path)
  }

  # Train final model with best parameters

  dvalid <- xgb.DMatrix(data = data.matrix(X_test), label = y_test)
  watchlist <- list(train = dtrain, eval = dvalid)

  # The tree count is fixed high and early stopping against the holdout decides where to stop
  final_model <- xgb.train(
    data = dtrain,
    nrounds = max_nrounds,
    max_depth = round(best_params[["max_depth"]]),
    eta = best_params[["eta"]],
    gamma = best_params[["gamma"]],
    min_child_weight = best_params[["min_child_weight"]],
    subsample = best_params[["subsample"]],
    colsample_bytree = best_params[["colsample_bytree"]],
    objective = "reg:squarederror",
    eval_metric = "rmse",
    tree_method = "hist",
    early_stopping_rounds = 25,
    watchlist = watchlist,
    verbose = 0
  )

  # Evaluate
  preds <- predict(final_model, newdata = data.matrix(X_test))
  rmse <- sqrt(mean((preds - y_test)^2))
  mae <- mean(abs(preds - y_test))

  list(
    model = final_model,
    features = feature_cols,
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
  library(xgboost)
  library(ggplot2)
  library(dplyr)

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

  # Return dataframe with predictions
  pred_df %>%
    select(Player, Year, Pos) %>%
    mutate(
      Predicted = predicted_points,
      Pred_Year = as.Date(paste0(as.numeric(format(Year, "%Y")) + 1, "-01-01"))
    )
}

# Function to display the anticipated "career trajectory" of players, combining historical results with forecasted performance
plot_predicted_trajectories <- function(combined_df, pred_df, pos_group = "QB", sample_n = 10) {
  library(dplyr)
  library(ggplot2)
  library(ggrepel)

  # Dynamically create prediction year as date
  pred_year <- as.Date(paste0(PRED_YEAR, "-01-01"))
  eval_year <- as.Date(paste0(EVAL_YEAR, "-01-01"))

  # 1. Historical data
  hist_df <- combined_df %>%
    filter(Pos == pos_group) %>%
    select(Player, Year, points)

  # 2. Predicted values
  preds <- pred_df %>%
    filter(Pos == pos_group) %>%
    select(Player, Pred_Year, Predicted) %>%
    rename(Year = Pred_Year, points = Predicted)

  # 3. Combine both
  full_df <- bind_rows(hist_df, preds)

  # 4. Filter players with > 50 pts in the most recent season
  active_players <- hist_df %>%
    filter(Year == eval_year, points > 50) %>%
    distinct(Player) %>%
    pull(Player)

  sampled_players <- sample(active_players, min(sample_n, length(active_players)))

  plot_df <- full_df %>%
    filter(Player %in% sampled_players)

  # 5. Labels at final year (projection year)
  label_df <- plot_df %>%
    group_by(Player) %>%
    slice_max(Year, n = 1, with_ties = FALSE) %>%
    ungroup()

  # 6. Coastal Breeze palette (darker tint)
  coastal_colors <- c(
    "#2C5985",  # Dark steel blue
    "#457C99",  # Dusty blue
    "#639DB8",  # Muted sky blue
    "#87BFD6",  # Cooler blue (formerly sky blue, darkened)
    "#A0C5CF",  # Muted powder blue
    "#507D76",  # Slate green-teal
    "#7F91A6",  # Cool grayish-blue
    "#476072"   # Dark coastal teal
  )
  pastel_colors <- rep(coastal_colors, length.out = length(unique(plot_df$Player)))

  ggplot(plot_df, aes(x = Year, y = points, color = Player, group = Player)) +
    geom_line(linewidth = 0.8, alpha = 0.8) +
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
    scale_x_date(expand = expansion(mult = c(0.01, 0.2))) +
    coord_cartesian(clip = "off") +
    scale_color_manual(values = pastel_colors) +
    labs(
      title = paste("Career Fantasy Point Trajectories +", format(pred_year, "%Y"), "Predictions"),
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
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_line(color = "#FFFFFF", linewidth = 0.3),
      axis.line = element_line(color = "#FFFFFF", linewidth = 0.4)
    )
}

# === DATA IMPORT === #
# Player data is sourced from the nflverse ecosystem via nflreadr, replacing the retired PFR scraper
# See build_player_season_stats in functions.R for column mapping notes and substitution decisions
if (SKIP_DATA_LOAD) {
  # Loading the prepared dataset directly from the data directory
  player_stats_final <- fread("data/player_stats_final.csv")
} else {
  # Refreshing the full player season dataset from nflverse and caching it for future runs
  player_stats_final <- build_player_season_stats(START_YEAR, EVAL_YEAR)
  fwrite(player_stats_final, "data/player_stats_final.csv")
}

#### Full dataset containing each player & statistical category
# The legacy traded player cleanup and duplicate name fixes are no longer needed,
# nflverse season summaries arrive pre-aggregated and keyed by the GSIS player id
combined <-
  player_stats_final |>
  mutate(points = (receiving_td * 6) + (receiving_yds * .1) + (Rec * PPR_MULT) +
           (rush_td * 6) + (rush_yds * .1) +
           (passing_yards * .04) + (passing_td * 4) -
           (rush_fbl * 2) - (passing_int * 2)) |>
  arrange(player_id, Year) |>
  group_by(player_id) |>
  mutate(Pos = last(Pos)) |> # Each player's most recent position will be used for their historical performance eval
  ungroup() |>
  mutate(Year = as.Date(as.yearmon(Year))) |>
  filter(Pos %in% c('WR', 'TE', 'RB', 'QB'))

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
    qb_yards = passing_yards + rush_yds,
    qb_yards_3yr = rollapplyr(qb_yards, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    avg_qbr_3yr = rollapplyr(QBR, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    passing_epa_per_att_3yr = rollapplyr(passing_epa_per_att, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    passing_adj_net_yards_att_3yr = rollapplyr(passing_adj_net_yards_att, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    passing_yards_att_3yr = rollapplyr(passing_yards_att, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),

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
    touches_3yr = rollapplyr(touches, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    receiving_epa_per_target_3yr = rollapplyr(receiving_epa_per_target, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    receiving_yards_target_3yr = rollapplyr(receiving_yards_target, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    Tgt_3yr = rollapplyr(Tgt, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    catch_percent_3yr = rollapplyr(catch_percent, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    adjusted_targets_3yr = rollapplyr(adjusted_targets, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    targets_per_game_3yr = rollapplyr(targets_per_game, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),

    # General career info
    career_total_points = cumsum(replace_na(points, 0)) - replace_na(points, 0),
    seasons_played = row_number() - 1,
    years_since_peak = seasons_played - which.max(points),

    # Per-game efficiency
    avg_points_per_game = if_else(G > 0, points / G, NA_real_),
    avg_points_per_game_3yr = rollapplyr(avg_points_per_game, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),

    # Flagging whether a player is playing above their 3-year average in the recent season (rising star vs fading)
    points_vs_3yr_avg = points - avg_points_3yr,

    # Injury / performance dip flags
    games_last_year = lag(G, 1),
    touches_last_year = lag(touches, 1),
    # Was this player injured in the previous season?
    # Players with a mostly healthy prior season who missed substantial time this season are flagged as injured
    injured_last_year = if_else(games_last_year >= 8 & G < 6, 1, 0),

    # Did this player recover from injury last season?
    prior_injury_flag = lag(injured_last_year, 1),

    # Draft capital serves as the talent pedigree signal in place of the retired award voting data
    # TODO: explore pulling award voting data from an alternative source to restore the star score features

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
  # Only apply infinity and NA fixes to numeric columns
  mutate(across(where(is.numeric), ~ ifelse(is.infinite(.), NA, .))) %>%
  mutate(across(where(is.numeric), ~ replace_na(., 0)))

## TODO: Remove columns with high correlation?

# Filtering for players with extremely few points in the next year, these players would not be drafted regardless
# Removing the EVAL_YEAR data from the training set, as it is the evaluation year
model_df <-
  combined %>%
  filter(!is.na(points_next_year), G > 0) %>%
  filter(
    (Pos == "QB" & points_next_year > 40) |
      (Pos %in% c("RB", "WR") & points_next_year > 25) |
      (Pos == "TE" & points_next_year > 15)
  ) %>%
  filter(Year < as.Date(paste0(EVAL_YEAR, "-01-01")))

# Creating the prediction dataframes for the evaluation year
pred_df <-
  combined %>%
  filter(!is.na(points_next_year), G > 0) %>%
  filter(Year == as.Date(paste0(EVAL_YEAR, "-01-01")))

# Splitting the prediction dataframe into positional groupings
qb_pred_df <- pred_df %>% filter(Pos == "QB")
rb_pred_df <- pred_df %>% filter(Pos %in% c("RB", "FB"))
wr_pred_df <- pred_df %>% filter(Pos %in% c("WR", "TE"))

qb_features <- c(
  "Age", "G", "Year", "QBR", "Rate", "adj_ypa",
  "age_points_interaction", "age_sq",
  "adjusted_productivity", "adjusted_productivity_3yr", "adjusted_productivity_trend",
  "avg_games_3yr",
  "avg_passing_td_3yr", "avg_passing_yds_3yr", "avg_points_3yr", "avg_points_per_game",
  "avg_points_per_game_3yr", "avg_qbr_3yr", "avg_rushing_yds_3yr",
  "career_adjusted_productivity", "career_games",
  "career_passing_td", "career_passing_yards", "career_rushing_td",
  "career_rushing_yds", "career_top_finish_count", "career_total_points",
  "consecutive_decline", "draft_number", "estimated_rookie_year", "games_last_year",
  "injured_last_year", "missing_pre_2006",
  "num_missing_years", "num_teams_prior",
  "qb_yards", "qb_yards_3yr", "passing_1D", "passing_adj_net_yards_att",
  "passing_adj_net_yards_att_3yr", "passing_att", "passing_avg_yards_att",
  "passing_comp", "passing_comp_pct", "passing_epa_per_att", "passing_epa_per_att_3yr",
  "passing_int", "passing_int_pct",
  "passing_net_yards_att", "passing_sack",
  "passing_td", "passing_td_pct", "passing_yards", "passing_yards_att", "passing_yards_att_3yr",
  "passing_yards_comp", "passing_yards_game", "points", "points_delta", "points_last_year",
  "points_pct_change", "points_vs_3yr_avg",
  "pos_rank", "pos_rank_last_year", "prior_injury_flag",
  "rate_per_attempt", "rush_1D", "rush_att", "rush_attempts_per_game",
  "rush_efficiency", "rush_epa_per_att", "rush_epa_per_att_3yr", "rush_fbl",
  "rush_td", "rush_yds", "rush_yds_att", "rush_yds_game",
  "sack_percent", "sack_yds", "seasons_played",
  "Team", "td_int_ratio", "top_finish_flag",
  "wins", "years_since_peak"
)

rb_features <- c(
  "Age", "G", "Year", "Tgt_3yr", "adj_ypa", "adjusted_productivity",
  "adjusted_productivity_3yr", "adjusted_productivity_trend",
  "adjusted_targets", "adjusted_targets_3yr",
  "age_points_interaction", "age_sq", "avg_games_3yr",
  "avg_points_3yr", "avg_points_per_game", "avg_points_per_game_3yr",
  "avg_receiving_td_3yr", "avg_receiving_yds_3yr", "avg_rushing_td_3yr",
  "avg_rushing_yds_3yr", "career_adjusted_productivity",
  "career_games", "career_receiving_td",
  "career_receiving_yds", "career_rushing_td", "career_rushing_yds",
  "career_top_finish_count", "career_total_points", "career_touches",
  "consecutive_decline", "draft_number", "estimated_rookie_year",
  "explosive_catch_rate", "explosive_receiving_eff", "explosive_yards_proxy",
  "fbl_per_att", "games_last_year", "injured_last_year", "missing_pre_2006",
  "num_teams_prior", "num_missing_years",
  "points", "points_delta", "points_last_year", "points_per_target", "points_per_touch",
  "points_pct_change", "points_vs_3yr_avg", "pos_rank", "pos_rank_last_year",
  "prior_injury_flag", "receiving_1D", "receiving_air_yards",
  "receiving_epa_per_target", "receiving_epa_per_target_3yr",
  "receiving_rec_g", "receiving_td", "receiving_yds", "receiving_yds_rec",
  "receiving_y_g", "receiving_yards_after_catch", "receiving_yards_target", "receiving_yards_target_3yr",
  "rush_1D", "rush_att", "rush_attempts_per_game", "rush_efficiency",
  "rush_epa_per_att", "rush_epa_per_att_3yr", "rush_fbl",
  "rush_td", "rush_yds", "rush_yds_att", "rush_yds_game",
  "seasons_played", "targets_per_game", "targets_per_game_3yr",
  "Team", "top_finish_flag", "touches", "touches_last_year", "touches_3yr",
  "yards_from_scrimmage", "yards_from_scrimmage_3yr", "years_since_peak"
)

wr_features <- c(
  "Age", "Pos", "G", "Year", "Rec", "Tgt", "Tgt_3yr", "adjusted_productivity",
  "adjusted_productivity_3yr", "adjusted_targets", "adjusted_targets_3yr", "adjusted_productivity_trend",
  "age_points_interaction", "age_sq", "avg_games_3yr",
  "avg_points_3yr", "avg_points_per_game", "avg_points_per_game_3yr",
  "avg_receiving_td_3yr", "avg_receiving_yds_3yr", "avg_rushing_td_3yr",
  "avg_rushing_yds_3yr", "career_adjusted_productivity",
  "career_games", "career_receiving_td",
  "career_receiving_yds", "career_rushing_yds", "career_top_finish_count",
  "career_total_points", "catch_percent", "catch_percent_3yr", "consecutive_decline",
  "draft_number", "estimated_rookie_year",
  "explosive_catch_rate", "explosive_receiving_eff", "explosive_yards_proxy",
  "games_last_year", "injured_last_year", "log_career_total_points",
  "missing_pre_2006", "num_teams_prior", "num_missing_years",
  "points", "points_delta", "points_last_year", "points_per_target",
  "points_pct_change", "points_vs_3yr_avg", "prior_injury_flag",
  "pos_rank", "pos_rank_last_year", "receiving_1D", "receiving_air_yards",
  "receiving_epa_per_target", "receiving_epa_per_target_3yr",
  "receiving_rec_g", "receiving_td", "receiving_yds", "receiving_yds_rec",
  "receiving_y_g", "receiving_yards_after_catch", "receiving_yards_target", "receiving_yards_target_3yr",
  "rush_att", "rush_epa_per_att", "rush_epa_per_att_3yr", "rush_fbl",
  "rush_td", "rush_yds", "rush_yds_att", "rush_yds_game",
  "seasons_played", "targets_per_game", "targets_per_game_3yr",
  "Team", "top_finish_flag", "touches", "yards_from_scrimmage",
  "yards_from_scrimmage_3yr", "years_since_peak"
)

# Creating models and making predictions for each major positional group
qb_model <- train_position_model(model_df, "QB", qb_features, init_points = 3, n_iter = 3)
plot_feature_importance(qb_model$model, qb_model$features, top_n = 20) +
  ggtitle("Quarterback Feature Importance")
qb_model_preds <- qb_model[['predictions']] %>%
  mutate(diff = Predicted - Actual)

rb_model <- train_position_model(model_df, "RB", rb_features, init_points = 3, n_iter = 3)
plot_feature_importance(rb_model$model, rb_model$features, top_n = 20) +
  ggtitle("Rushing Feature Importance")
rb_model_preds <- rb_model[['predictions']] %>%
  mutate(diff = Predicted - Actual)

# IMPORTANT: TEs will be included in the WR model by default
wr_model <- train_position_model(model_df, "WR", wr_features, init_points = 3, n_iter = 3) 
plot_feature_importance(wr_model$model, wr_model$features, top_n = 20) +
  ggtitle("Receiving Feature Importance")
wr_model_preds <- wr_model[['predictions']] %>%
  mutate(diff = Predicted - Actual)

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
plot_actual_vs_pred(qb_model_preds, "QB")
plot_resid_vs_pred(qb_model_preds, "QB")
plot_resid_hist(qb_model_preds, "QB")
plot_decile_calib(qb_model_preds, "QB")

# RB diagnostics
plot_actual_vs_pred(rb_model_preds, "RB")
plot_resid_vs_pred(rb_model_preds, "RB")
plot_resid_hist(rb_model_preds, "RB")
plot_decile_calib(rb_model_preds, "RB")

# WR/TE diagnostics
plot_actual_vs_pred(wr_model_preds, "WR/TE")
plot_resid_vs_pred(wr_model_preds, "WR/TE")
plot_resid_hist(wr_model_preds, "WR/TE")
plot_decile_calib(wr_model_preds, "WR/TE")

# Making player predictions for the upcoming season
qb_preds <- predict_next_year(qb_model, qb_pred_df)
rb_preds <- predict_next_year(rb_model, rb_pred_df)
wr_preds <- predict_next_year(wr_model, wr_pred_df)

# Visualizing predicted player performance trajectories
plot_predicted_trajectories(combined, qb_preds, pos_group = "QB", sample_n = 8)
plot_predicted_trajectories(combined, rb_preds, pos_group = "RB", sample_n = 8)
plot_predicted_trajectories(combined, wr_preds, pos_group = "WR", sample_n = 8)
plot_predicted_trajectories(combined, wr_preds, pos_group = "TE", sample_n = 8)

# Saving out final combined dataframe
final <-
  bind_rows(qb_preds, wr_preds, rb_preds) %>%
  mutate(Player = clean_player_name(Player)) %>% # Cleaning Player Names
  select(Player, Pos, Predicted)

fwrite(final, paste0("data/model_pred_", as.character(PRED_YEAR), "_", SCORING_TYPE, ".csv"))
