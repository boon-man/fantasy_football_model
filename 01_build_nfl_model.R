

# === IMPORTANT === #
# This script is not designed to let it rip all in one go,
# It is intended to be processed or evaluated in steps to ensure that data is being compiled appropriately
# And that the model is running effectively, making "quality" predictions
# PLEASE DO NOT JUST HIT RUN ALL

# === GLOBAL CONFIGURATION === #
source("00_globals.R") # Running global variable config script
SKIP_DATA_LOAD <- TRUE  # Set to TRUE to skip data loading and use pre-saved data

# Function to scrape data from Pro Football Reference
scrapeData = function(urlprefix, urlend, startyr, endyr, stat) {
  master <- data.frame()
  for (i in startyr:endyr) {
    Sys.sleep(5)
    cat('Loading Year', i, '\n')
    URL <- paste(urlprefix, i, urlend, sep = "")
    table <-
      read_html(URL) %>%
      html_node('table') %>%
      html_table()
    
    table$Year <- i
    master <- rbind(table, master)
  }
  assign(quo_name(enquo(stat)), master, envir=.GlobalEnv)
  return('Complete')
}

# Function to filter out split season stat rows, when players were traded or released to join a new team mid-season.
clean_traded_players <- function(df) {
  df %>%
    group_by(Player, Year) %>%
    mutate(has_combined_row = any(Team %in% c("2TM", "3TM", "4TM"))) %>%
    filter(
      (has_combined_row & Team %in% c("2TM", "3TM", "4TM")) |
        (!has_combined_row)
    ) %>%
    ungroup()
}

# Function to expand the Awards data column on player datasets
expand_awards <- function(df, player_col = "Player", year_col = "Year", awards_col = "Awards") {
  require(dplyr)
  require(tidyr)
  require(stringr)
  
  df %>%
    filter(!is.na(.data[[awards_col]]) & .data[[awards_col]] != "") %>%
    mutate(Award_List = str_split(.data[[awards_col]], ",")) %>%
    unnest(Award_List) %>%
    mutate(
      Award_List = str_squish(Award_List),
      Award_Type = str_extract(Award_List, "^[^0-9\\-]+"),
      Award_Type = str_replace_all(Award_Type, " ", "_"),
      Award_Rank = as.numeric(str_extract(Award_List, "\\d+$")),
      Award_Rank = ifelse(is.na(Award_Rank), 1, Award_Rank)
    ) %>%
    select(all_of(c(player_col, year_col)), Award_Type, Award_Rank) %>%
    pivot_wider(
      names_from = Award_Type,
      values_from = Award_Rank,
      values_fill = list(Award_Rank = 0)
    )
}

# Function for cleaning up player name columns for joining projection data together
clean_player_name <- function(name) {
  name %>%
    tolower() %>%
    str_remove_all("\\b(jr|sr|ii|iii|iv|v)\\b\\.?") %>%  # remove suffixes
    str_replace_all("['’\\.\\-]", "") %>%               # remove apostrophes, periods, hyphens
    str_replace_all("[^a-z ]", " ") %>%                 # remove any remaining non-letter chars
    str_squish() %>%
    str_to_title()
}

# Function to train the XGBoost model for a specific position
train_position_model <- function(df, position, feature_cols) {
  library(dplyr)
  library(xgboost)
  library(rBayesianOptimization)
  library(caret)
  
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
  
  # Define Bayesian optimization function
  xgb_cv_bayes <- function(nrounds, max_depth, eta, gamma, min_child_weight, subsample, colsample_bytree) {
    set.seed(82525)
    
    nrounds <- as.integer(round(nrounds))
    max_depth <- as.integer(round(max_depth))
    
    # Validate parameters
    if (anyNA(c(nrounds, max_depth, eta, gamma, min_child_weight, subsample))) {
      return(list(Score = -1e5, Pred = 0))
    }
    if (nrounds <= 0 || max_depth <= 0) {
      return(list(Score = -1e5, Pred = 0))
    }
    
    # Run CV
    cv <- tryCatch({
      xgb.cv(
        data = dtrain,
        nrounds = nrounds,
        nfold = 5,
        early_stopping_rounds = 50,
        objective = "reg:squarederror",
        eval_metric = "mae",
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
    
    if (!is.finite(eta) || !is.finite(gamma) || !is.finite(subsample)) {
      return(list(Score = -1e5, Pred = 0))
    }
    
    best_mae <- min(cv$evaluation_log$test_mae_mean, na.rm = TRUE)
    
    if (!is.finite(best_mae)) {
      return(list(Score = -1e5, Pred = 0))
    }
    
    list(Score = -best_mae, Pred = 0)
  }
  
  # Run Bayesian Optimization
  opt_result <- BayesianOptimization(
    FUN = xgb_cv_bayes,
    bounds = list(
      nrounds = c(150, 750),
      max_depth = c(3, 7),
      eta = c(0.1, 0.3),
      gamma = c(0, 0.1),
      min_child_weight = c(0.05, 0.5),
      subsample = c(0.9, 1.0),
      colsample_bytree = c(0.7, 1.0)  
    ),
    init_points = 20,
    n_iter = 30,
    acq = "ucb",          # Or ei depending on strategy
    kappa = 1.75,
    eps = 0.4,
    verbose = TRUE
  )
  
  # Train final model with best parameters
  best_params <- opt_result$Best_Par
  
  dvalid <- xgb.DMatrix(data = data.matrix(X_test), label = y_test)
  watchlist <- list(train = dtrain, eval = dvalid)
  
  final_model <- xgb.train(
    data = dtrain,
    nrounds = round(best_params[["nrounds"]]),
    max_depth = round(best_params[["max_depth"]]),
    eta = best_params[["eta"]],
    gamma = best_params[["gamma"]],
    min_child_weight = best_params[["min_child_weight"]],
    subsample = best_params[["subsample"]],
    colsample_bytree = best_params[["colsample_bytree"]],
    objective = "reg:squarederror",
    eval_metric = "mae",
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

# Function to plot out actual vs predicted points
plot_actual_vs_predicted <- function(pred_df) {
  ggplot(pred_df, aes(x = Actual, y = Predicted)) +
    geom_point(alpha = 0.6) +
    geom_smooth(linewidth = 0.6, alpha = 0.85, method = "lm", se = FALSE, color = "#FFC461", linetype = "solid") +  # Regression line
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#619cff") +
    labs(
      title = "Predicted vs. Actual Fantasy Points",
      x = "Actual Points (Next Year)",
      y = "Predicted Points"
    ) +
    theme_minimal()
}

# Function to plot prediction error distribution
plot_prediction_error_distribution <- function(pred_df) {
  pred_df <- pred_df %>%
    mutate(error = Predicted - Actual)
  
  ggplot(pred_df, aes(x = error)) +
    geom_histogram(
      aes(y = after_stat(density)),           # Normalize histogram to match density curve
      binwidth = 5,
      fill = "#619cff",
      color = "white",
      alpha = 0.8
    ) +
    geom_density(color = "#FFC461", linewidth = 0.7, alpha = 0.9) +  # Add smoothed density line
    labs(
      title = "Distribution of Prediction Errors",
      x = "Prediction Error (Predicted - Actual)",
      y = "Density"
    ) +
    theme_minimal()
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
    geom_line(linewidth = 0.9) +
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
      panel.background = element_rect(fill = "#ECDFCF", color = NA),
      plot.background = element_rect(fill = "#ECDFCF", color = NA),
      legend.position = "none",
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_line(color = "#FFFFFF", linewidth = 0.3),
      axis.line = element_line(color = "#FFFFFF", linewidth = 0.4) 
    )
}

# Importing data
if (SKIP_DATA_LOAD) {
  # Load cleaned data directly from "data/" directory
  receiving_final <- fread("data/receiving_final.csv")
  rushing_final <- fread("data/rushing_final.csv")
  passing_final <- fread("data/passing_final.csv")
  
} else {
  # Scrape and process data if SKIP_DATA_LOAD is FALSE
  
  scrapeData("https://www.pro-football-reference.com/years/", "/receiving.htm", START_YEAR, EVAL_YEAR, 'receiving')
  scrapeData("https://www.pro-football-reference.com/years/", "/rushing.htm", START_YEAR, EVAL_YEAR, 'rushing')
  scrapeData("https://www.pro-football-reference.com/years/", "/passing.htm", START_YEAR, EVAL_YEAR, 'passing')
  
  # --- Receiving Cleaning ---
  colnames(receiving) <- as.character(receiving[1, ])
  receiving <- receiving[-1, ]
  colnames(receiving)[which(names(receiving) == "Ctch%")] <- "catch_percent"
  receiving$Player <- gsub("[^[:alnum:][:space:]]", "", receiving$Player)
  receiving$Player <- str_squish(receiving$Player)
  receiving$catch_percent <- gsub("%", "", receiving$catch_percent)
  receiving_strings <- c('Player', 'Team', 'Pos', 'Awards')
  
  receiving_final <- receiving %>%
    rename(Year = `2024`) %>%
    mutate_at(vars(-one_of(receiving_strings)), list(as.numeric)) %>%
    replace(is.na(.), 0) %>%
    select(-c(Rk, Fmb)) %>%
    rename(
      receiving_yds = Yds,
      receiving_yds_rec = `Y/R`,
      receiving_td = TD,
      receiving_1D = `1D`,
      receiving_success_pct = `Succ%`,
      receiving_rec_g = `R/G`,
      receiving_long = Lng,
      receiving_y_g = `Y/G`,
      receiving_yards_target = `Y/Tgt`
    ) %>%
    clean_traded_players()
  
  # --- Rushing Cleaning ---
  colnames(rushing) <- as.character(rushing[1, ])
  rushing <- rushing[-1, ]
  rushing$Player <- gsub("[^[:alnum:][:space:]]", "", rushing$Player)
  rushing$Player <- str_squish(rushing$Player)
  rushing_strings <- c('Player', 'Team', 'Pos', 'Awards')
  
  rushing <- rushing %>%
    rename(Year = `2024`) %>%
    mutate_at(vars(-one_of(rushing_strings)), list(as.numeric)) %>%
    replace(is.na(.), 0) %>%
    rename(
      success_pct = `Succ%`,
      long = Lng,
      yds_att = `Y/A`,
      yds_game = `Y/G`,
      attempts_per_game = `A/G`
    )
  
  rushing_final <- rushing %>%
    rename(
      rush_att = Att,
      rush_yds = Yds,
      rush_td = TD,
      rush_1D = `1D`,
      rush_success_pct = success_pct,
      rush_long = long,
      rush_yds_att = yds_att,
      rush_yds_game = yds_game,
      rush_attempts_per_game = attempts_per_game,
      rush_fbl = `Fmb`
    ) %>%
    select(-c(Rk)) %>%
    clean_traded_players()
  
  # --- Passing Cleaning ---
  passing$Player <- gsub("[^[:alnum:][:space:]]", "", passing$Player)
  passing$Player <- str_squish(passing$Player)
  names(passing)[27] <- "sack_yds"
  passing <- passing %>%
    mutate(wins = ifelse(is.na(QBrec) | QBrec == "", 0, as.numeric(str_extract(QBrec, "^[0-9]+"))))
  
  passing_strings <- c('Player', 'Team', 'Pos', 'Awards')
  
  passing_final <- passing %>%
    select(-QBrec) %>%
    mutate_at(vars(-one_of(passing_strings)), list(as.numeric)) %>%
    replace(is.na(.), 0) %>%
    rename(
      passing_comp = Cmp,
      passing_att = Att,
      passing_comp_pct = `Cmp%`,
      passing_yards = Yds,
      passing_td = TD,
      passing_td_pct = `TD%`,
      passing_int = Int,
      passing_int_pct = `Int%`,
      passing_1D = `1D`,
      passing_success_pct = `Succ%`,
      passing_long = Lng,
      passing_yards_att = `Y/A`,
      passing_avg_yards_att = `AY/A`,
      passing_yards_comp = `Y/C`,
      passing_yards_game = `Y/G`,
      passing_sack = Sk,
      sack_percent = `Sk%`,
      passing_net_yards_att = `NY/A`,
      passing_adj_net_yards_att = `ANY/A`,
      passing_comebacks = `4QC`
    ) %>%
    select(-c(Rk)) %>%
    clean_traded_players()
  
  # removing intermediate data files
  rm(receiving, rushing, passing)
  
  # Save cleaned data
  fwrite(receiving_final, "data/receiving_final.csv")
  fwrite(rushing_final, "data/rushing_final.csv")
  fwrite(passing_final, "data/passing_final.csv")
}

#### Full dataset containing each player & statistical category
combined <-
  receiving_final %>%
  full_join(rushing_final, by = c('Player', 'Team', 'Year', 'Age', 'G', 'GS', 'Pos', 'Awards')) %>%
  full_join(passing_final, by = c('Player', 'Team', 'Year', 'Age', 'G', 'GS', 'Pos', 'Awards')) %>%
  select(Player, Year, Pos, everything()) %>%
  replace(is.na(.), 0) %>%
  mutate(points = (receiving_td * 6) + (receiving_yds * .1) + (Rec * PPR_MULT) +
           (rush_td * 6) + (rush_yds * .1) +
           (passing_yards * .04) + (passing_td * 4) -
           (rush_fbl * 2) - (passing_int * 2)) %>%
  arrange(Player, Year) %>%
  group_by(Player) %>%
  mutate(Pos = last(Pos)) %>% # Each player's most recent position will be used for their historical performance eval
  ungroup() %>%
  mutate(Year = as.Date(as.yearmon(Year))) %>%
  filter(Pos %in% c('WR', 'TE', 'RB', 'QB'))
#  mutate(Player = clean_player_name(Player)) # Cleaning Player Names

# Removing duplicate player rows again after creating combined dataset
# There are wonky circumstances where Christian McCaffery completed a pass with one team during a season but not another that he was traded to, distorting a join
combined <- 
  clean_traded_players(combined) %>%
  select(-contains("has_combined"))

# Removes records of a Mike Williams that played in the mid-2000s. They hardly played and distort the rest of Mike Williams data
combined <- combined %>%
  filter(!(Player == "Mike Williams" & Year < as.Date("2010-01-01")))

# Distinguishing Marvin Harrison Jr from Marvin Harrison Sr in the dataset
# combined <-
#   combined %>%
#   mutate(Player = if_else(Player == "Marvin Harrison" & year(Year) >= 2024,
#                  "Marvin Harrison Jr",
#                  Player))


# Creating the awards dataframe
awards_df <- expand_awards(combined)

# Joining player awards rows onto the combined dataset
combined <-
  combined %>%
  left_join(awards_df, by = c("Player", "Year")) %>%
  select(-Awards) %>%
  mutate(across(c(AP_MVP, AP_OPoY, AP_CPoY, AP_ORoY),
                ~ ifelse(. == 0, NA, 1 / .))) %>%  # Invert rank, necessary as columns represent final vote ranking (e.g., MVP has value of 1, 2nd place for MVP has value of 2)
  replace(is.na(.), 0)

# Feature engineering
combined <- 
  combined %>%
  arrange(Player, Year) %>%
  group_by(Player) %>%
  arrange(Year) %>%
  mutate(
    year_num = year(Year),
    # Identifying a player's rookie year
    rookie_year = min(year_num),
    # Data prior to 2006 is not available, this will make the model incorrectly assume a player like Peyton Manning was a 30 year old rookie
    missing_pre_2006 = if_else(rookie_year <= 2006 & Age > 23, 1, 0),
    # Estimating the rookie year for players that played prior to 2006
    estimated_rookie_year = if_else(
      missing_pre_2006 == 1,
      year_num - (Age - 22),  # Assume they were ~22 in their real rookie year
      rookie_year
    ),
    # How many years of a player's career are missing from the data?
    num_missing_years = 
      if_else(
        missing_pre_2006 == 1,
        2006 - estimated_rookie_year,
        0
    ),
    
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
    career_games_started = cumsum(replace_na(GS, 0)) - replace_na(GS, 0),
    career_adjusted_productivity = cumsum(replace_na(adjusted_productivity, 0)) - replace_na(adjusted_productivity, 0),
    
    career_touches = cumsum(replace_na(touches, 0)) - replace_na(touches, 0),
    career_rushing_yds = cumsum(replace_na(rush_yds, 0)) - replace_na(rush_yds, 0),
    career_receiving_yds = cumsum(replace_na(receiving_yds, 0)) - replace_na(receiving_yds, 0),
    career_passing_yards = cumsum(replace_na(passing_yards, 0)) - replace_na(passing_yards, 0),
    
    career_rushing_td = cumsum(replace_na(rush_td, 0)) - replace_na(rush_td, 0),
    career_receiving_td = cumsum(replace_na(receiving_td, 0)) - replace_na(receiving_td, 0),
    career_passing_td = cumsum(replace_na(passing_td, 0)) - replace_na(passing_td, 0),
    
    # Rolling averages for the last 3 years for major statistical categories
    avg_games_3yr = rollapplyr(G, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    avg_games_started_3yr = rollapplyr(GS, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
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
    passing_success_pct_3yr = rollapplyr(passing_success_pct, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    passing_adj_net_yards_att_3yr = rollapplyr(passing_adj_net_yards_att, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    passing_yards_att_3yr = rollapplyr(passing_yards_att, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    
    # Rushing efficiency metrics
    rush_efficiency = rush_yds * (1 + rush_success_pct),
    explosive_yards_proxy = rush_yds - (rush_success_pct * rush_att * 4),
    adj_ypa = rush_yds_att * (1 + rush_success_pct),
    fbl_per_att = rush_fbl / rush_att,
    yards_from_scrimmage = rush_yds + receiving_yds,
    yards_from_scrimmage_3yr = rollapplyr(yards_from_scrimmage, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    
    # Receiving efficiency metrics
    explosive_catch_rate = if_else(Rec > 0, receiving_1D / Rec, 0),
    explosive_receiving_eff = if_else(receiving_yds > 100, catch_percent * receiving_yards_target, 0),
    
    # Adding a rolling avg for rushing/receiving efficiency stats
    rush_success_pct_3yr = rollapplyr(rush_success_pct, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    touches_3yr = rollapplyr(touches, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
    receiving_success_pct_3yr = rollapplyr(receiving_success_pct, width = 3, FUN = mean, fill = NA, align = "right", partial = TRUE),
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
    # If a player played 8 or more games last year but less than 8 this year, they are considered injured
    injured_last_year = if_else(games_last_year >= 8 & G < 6, 1, 0),
    
    # Did this player recover from injury last season?
    prior_injury_flag = lag(injured_last_year, 1),
    
    # Creating a star score based on performance metrics
    star_score = (PB + AP + AP_MVP * 2 + AP_OPoY * 1.5 + AP_CPoY + AP_ORoY),
    age_star_score_interaction = Age * star_score,
    
    # Was the player highly rated in the previous season?
    most_recent_star_score = lag(star_score, 1),
    
    # Cumulative star score for the player's career
    career_star_score = cumsum(star_score) - star_score,
    
    # Logged major career statistics to normalize skewed distributions
    log_career_total_points = log1p(career_total_points),
    log_career_star_score = log1p(career_star_score)
    
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
  group_by(Player) %>%
  arrange(Year) %>%
  mutate(pos_rank_last_year = lag(pos_rank, 1), # prior year rank
         career_top_finish_count = cumsum(top_finish_flag)) %>%  
  ungroup() %>%
  # Only apply infinity and NA fixes to numeric columns
  mutate(across(where(is.numeric), ~ ifelse(is.infinite(.), NA, .))) %>% 
  mutate(across(where(is.numeric), ~ replace_na(., 0)))

## TODO: Remove columns with high correlation?
#check <- combined %>% 
#  select(Player, Year, Pos, Age, G, receiving_yds, rush_yds, passing_yards, seasons_played, career_total_points, rookie_year, missing_pre_2006, pos_rank, pos_rank_last_year)

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
  "Age", "G", "GS", "Year", "QBR", "Rate", "adj_ypa", 
  "age_points_interaction", "age_sq", "age_star_score_interaction",
  "adjusted_productivity", "adjusted_productivity_3yr", "adjusted_productivity_trend",
  "avg_games_3yr", "avg_games_started_3yr",
  "avg_passing_td_3yr", "avg_passing_yds_3yr", "avg_points_3yr", "avg_points_per_game",
  "avg_points_per_game_3yr", "avg_qbr_3yr", "avg_rushing_yds_3yr", 
  "career_adjusted_productivity", "career_games",
  "career_games_started", "career_passing_td", "career_passing_yards", "career_rushing_td",
  "career_rushing_yds", "career_star_score", "career_top_finish_count", "career_total_points", 
  "consecutive_decline", "estimated_rookie_year", "GWD", "games_last_year",
  "injured_last_year", "missing_pre_2006",
  "most_recent_star_score", "num_missing_years", "num_teams_prior",
  "qb_yards", "qb_yards_3yr", "passing_1D", "passing_adj_net_yards_att",
  "passing_adj_net_yards_att_3yr", "passing_att", "passing_avg_yards_att", "passing_comebacks",
  "passing_comp", "passing_comp_pct", "passing_int", "passing_int_pct", "passing_long",
  "passing_net_yards_att", "passing_sack", "passing_success_pct", "passing_success_pct_3yr",
  "passing_td", "passing_td_pct", "passing_yards", "passing_yards_att", "passing_yards_att_3yr", 
  "passing_yards_comp", "passing_yards_game", "points", "points_delta", "points_last_year", 
  "points_pct_change", "points_vs_3yr_avg",
  "pos_rank", "pos_rank_last_year", "prior_injury_flag",
  "rate_per_attempt", "rush_1D", "rush_att", "rush_attempts_per_game",
  "rush_efficiency", "rush_fbl", "rush_long", "rush_success_pct", "rush_success_pct_3yr", 
  "rush_td", "rush_yds", "rush_yds_att", "rush_yds_game", 
  "sack_percent", "sack_yds", "seasons_played", 
  "star_score", "Team", "td_int_ratio", "top_finish_flag",
  "wins", "years_since_peak"
)

rb_features <- c(
  "Age", "G", "GS", "Year", "Tgt_3yr", "adj_ypa", "adjusted_productivity",
  "adjusted_productivity_3yr", "adjusted_productivity_trend", 
  "adjusted_targets", "adjusted_targets_3yr",
  "age_points_interaction", "age_sq", "age_star_score_interaction", "avg_games_3yr",
  "avg_games_started_3yr", "avg_points_3yr", "avg_points_per_game", "avg_points_per_game_3yr",
  "avg_receiving_td_3yr", "avg_receiving_yds_3yr", "avg_rushing_td_3yr",
  "avg_rushing_yds_3yr", "career_adjusted_productivity",
  "career_games", "career_games_started", "career_receiving_td",
  "career_receiving_yds", "career_rushing_td", "career_rushing_yds", "career_star_score",
  "career_top_finish_count", "career_total_points", "career_touches",
  "consecutive_decline", "estimated_rookie_year",
  "explosive_catch_rate", "explosive_receiving_eff", "explosive_yards_proxy",
  "fbl_per_att", "games_last_year", "injured_last_year", "missing_pre_2006",
  "most_recent_star_score", "num_teams_prior", "num_missing_years",
  "points", "points_delta", "points_last_year", "points_per_target", "points_per_touch",
  "points_pct_change", "points_vs_3yr_avg", "pos_rank", "pos_rank_last_year",
  "prior_injury_flag", "receiving_1D",
  "receiving_long", "receiving_rec_g", "receiving_success_pct",
  "receiving_success_pct_3yr", "receiving_td", "receiving_yds", "receiving_yds_rec",
  "receiving_y_g", "receiving_yards_target", "receiving_yards_target_3yr", "rush_1D",
  "rush_att", "rush_attempts_per_game", "rush_efficiency", "rush_fbl", "rush_long", "rush_success_pct",
  "rush_success_pct_3yr", "rush_td", "rush_yds", "rush_yds_att", "rush_yds_game",
  "seasons_played", "star_score", "targets_per_game", "targets_per_game_3yr",
  "Team", "top_finish_flag", "touches", "touches_last_year", "touches_3yr",
  "yards_from_scrimmage", "yards_from_scrimmage_3yr", "years_since_peak"
)

wr_features <- c(
  "Age", "Pos", "G", "GS", "Year", "Rec", "Tgt", "Tgt_3yr", "adjusted_productivity",
  "adjusted_productivity_3yr", "adjusted_targets", "adjusted_targets_3yr", "adjusted_productivity_trend",
  "age_points_interaction", "age_sq", "age_star_score_interaction", "avg_games_3yr",
  "avg_games_started_3yr", "avg_points_3yr", "avg_points_per_game", "avg_points_per_game_3yr",
  "avg_receiving_td_3yr", "avg_receiving_yds_3yr", "avg_rushing_td_3yr",
  "avg_rushing_yds_3yr", "career_adjusted_productivity", 
  "career_games", "career_games_started", "career_receiving_td",
  "career_receiving_yds", "career_rushing_yds", "career_star_score", "career_top_finish_count",
  "career_total_points", "catch_percent", "catch_percent_3yr", "consecutive_decline",
  "estimated_rookie_year",
  "explosive_catch_rate", "explosive_receiving_eff", "explosive_yards_proxy",
  "games_last_year", "injured_last_year", "log_career_star_score", "log_career_total_points",
  "missing_pre_2006", "most_recent_star_score", "num_teams_prior", "num_missing_years",
  "points", "points_delta", "points_last_year", "points_per_target",
  "points_pct_change", "points_vs_3yr_avg", "prior_injury_flag", 
  "pos_rank", "pos_rank_last_year", "receiving_1D",
  "receiving_long", "receiving_rec_g", "receiving_success_pct",
  "receiving_success_pct_3yr", "receiving_td", "receiving_yds", "receiving_yds_rec",
  "receiving_y_g", "receiving_yards_target", "receiving_yards_target_3yr",
  "rush_att", "rush_fbl", "rush_long", "rush_success_pct",
  "rush_success_pct_3yr", "rush_td", "rush_yds", "rush_yds_att", "rush_yds_game",
  "seasons_played", "star_score", "targets_per_game", "targets_per_game_3yr",
  "Team", "top_finish_flag", "touches", "yards_from_scrimmage", 
  "yards_from_scrimmage_3yr", "years_since_peak"
)

# Creating models and making predictions for each major positional group
qb_model <- train_position_model(model_df, "QB", qb_features)
plot_feature_importance(qb_model$model, qb_model$features, top_n = 20) +
  ggtitle("Quarterback Feature Importance")
qb_model_preds <- qb_model[['predictions']] %>%
  mutate(diff = Predicted - Actual)

rb_model <- train_position_model(model_df, "RB", rb_features)
plot_feature_importance(rb_model$model, rb_model$features, top_n = 20) +
  ggtitle("Rushing Feature Importance")
rb_model_preds <- rb_model[['predictions']] %>%
  mutate(diff = Predicted - Actual)

wr_model <- train_position_model(model_df, "WR", wr_features) # IMPORTANT: TEs will be included in the WR model by default
plot_feature_importance(wr_model$model, wr_model$features, top_n = 20) +
  ggtitle("Receiving Feature Importance")
wr_model_preds <- wr_model[['predictions']] %>%
  mutate(diff = Predicted - Actual)

# Evaluating model performance
print(qb_model$rmse)
print(qb_model$mae)

print(rb_model$rmse)
print(rb_model$mae)

print(wr_model$rmse)
print(wr_model$mae)

# Plotting actual vs predicted for each position
plot_actual_vs_predicted(qb_model_preds) +
  ggtitle("Quarterback Predictions")
plot_prediction_error_distribution(qb_model_preds) +
  ggtitle("Quarterback Prediction Error")

plot_actual_vs_predicted(rb_model_preds) +
  ggtitle("Rushing Predictions")
plot_prediction_error_distribution(rb_model_preds) +
  ggtitle("Rushing Prediction Error")

plot_actual_vs_predicted(wr_model_preds) +
  ggtitle("Receiving Predictions")
plot_prediction_error_distribution(wr_model_preds) +
  ggtitle("Receiving Prediction Error")

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
