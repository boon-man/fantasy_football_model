#############################################################################
### Optimizing Model & Expert Projection Blend Weights
library(tidyverse)
library(data.table)
library(rvest)
library(nflreadr)

source("00_globals.R")
source("functions.R")

# Loading actual 2025 data
receiving <- read_csv("data/receiving_2025.csv", col_names = TRUE)
rushing <- read_csv("data/rushing_2025.csv", col_names = TRUE)
passing <- read_csv("data/passing_2025.csv", col_names = TRUE)

# --- Receiving Cleaning ---
# Dropping column at the end of the dataframe
receiving <- receiving[, -ncol(receiving)]
receiving$Year <- 2025
colnames(receiving)[which(names(receiving) == "Ctch%")] <- "catch_percent"
receiving$Player <- gsub("[^[:alnum:][:space:]]", "", receiving$Player)
receiving$Player <- str_squish(receiving$Player)
receiving$catch_percent <- gsub("%", "", receiving$catch_percent)
receiving_strings <- c('Player', 'Team', 'Pos', 'Awards')

receiving_final <- receiving %>%
  mutate(across(where(is.character), as.character)) %>%
  mutate(across(!all_of(receiving_strings), as.numeric)) %>%
  mutate(across(everything(), ~replace(., is.na(.), 0))) %>%
  select(-c(Rk, Fmb)) %>%
  rename(
    receiving_yds = Yds,
    receiving_td = TD,
    receiving_1D = `1D`,
    receiving_rec_g = `R/G`,
    receiving_long = Lng,
    receiving_y_g = `Y/G`
  ) %>%
  clean_traded_players()

# --- Rushing Cleaning ---
rushing <- rushing[, -ncol(rushing)]
rushing$Year <- 2025
rushing$Player <- gsub("[^[:alnum:][:space:]]", "", rushing$Player)
rushing$Player <- str_squish(rushing$Player)
rushing_strings <- c('Player', 'Team', 'Pos', 'Awards')

rushing_final <- rushing %>%
  mutate(across(where(is.character), as.character)) %>%
  mutate(across(!all_of(rushing_strings), as.numeric)) %>%
  mutate(across(everything(), ~replace(., is.na(.), 0))) %>%
  rename(
    rush_att = Att,
    rush_yds = Yds,
    rush_td = TD,
    rush_1D = `1D`,
    rush_long = Lng,
    rush_yds_att = `Y/A`,
    rush_yds_game = `Y/G`,
    rush_fbl = `Fmb`
  ) %>%
  select(-c(Rk)) %>%
  clean_traded_players()

# --- Passing Cleaning ---
passing <- passing[, -ncol(passing)]
passing$Year <- 2025
passing$Player <- gsub("[^[:alnum:][:space:]]", "", passing$Player)
passing$Player <- str_squish(passing$Player)
names(passing)[12] <- "Yds"
names(passing)[27] <- "sack_yds"

passing_strings <- c('Player', 'Team', 'Pos', 'Awards')

passing_final <- passing %>%
  select(-QBrec) %>%
  mutate(across(where(is.character), as.character)) %>%
  mutate(across(!all_of(passing_strings), as.numeric)) %>%
  mutate(across(everything(), ~replace(., is.na(.), 0))) %>%
  rename(
    passing_comp = Cmp,
    passing_att = Att,
    passing_yards = Yds,
    passing_td = TD,
    passing_int = Int,
    passing_1D = `1D`,
    passing_long = Lng,
    passing_yards_att = `Y/A`,
    passing_avg_yards_att = `AY/A`,
    passing_yards_comp = `Y/C`,
    passing_yards_game = `Y/G`,
    passing_sack = Sk,
    sack_percent = `Sk%`
  ) %>%
  select(-c(Rk)) %>%
  clean_traded_players()

# --- Combine Actual 2025 Results ---
actual_2025 <-
  receiving_final %>%
  full_join(rushing_final, by = c('Player', 'Team', 'Year', 'Age', 'G', 'GS', 'Pos', 'Awards')) %>%
  full_join(passing_final, by = c('Player', 'Team', 'Year', 'Age', 'G', 'GS', 'Pos', 'Awards')) %>%
  select(Player, Pos, everything()) %>%
  replace(is.na(.), 0) %>%
  mutate(
    actual_points = (receiving_td * 6) + (receiving_yds * 0.1) + (Rec * PPR_MULT) +
      (rush_td * 6) + (rush_yds * 0.1) +
      (passing_yards * 0.04) + (passing_td * 4) -
      (rush_fbl * 2) - (passing_int * 2)
  ) %>%
  select(Player, Pos, actual_points) %>%
  mutate(Player = clean_player_name(Player))

# --- Load Projections ---
projections <- 
  read_csv(paste0("data/blended_proj_", as.character(EVAL_YEAR), "_", SCORING_TYPE, ".csv")) %>%
  select(Player, Pos, Model_Prediction, FantasyPros_Prediction)

# --- Merge Actuals with Projections ---
blend_analysis <- 
  projections %>%
  left_join(actual_2025, by = c("Player", "Pos")) %>%
  filter(!is.na(actual_points)) %>%
  drop_na(Model_Prediction, FantasyPros_Prediction, actual_points)

# --- Grid Search for Optimal Weights ---
weight_grid <- expand_grid(
  model_weight = seq(0, 1, by = 0.05),
  expert_weight = seq(0, 1, by = 0.05)
) %>%
  filter(model_weight + expert_weight == 1) %>%
  mutate(dampening = 1)  # Start without dampening factor

# Function to calculate blend and error metrics
calculate_blend_error <- function(df, model_w, expert_w, damp = 1) {
  df %>%
    mutate(
      blended_pred = model_w * Model_Prediction + expert_w * (FantasyPros_Prediction * damp),
      error = blended_pred - actual_points,
      abs_error = abs(error),
      squared_error = error^2
    ) %>%
    summarise(
      mae = mean(abs_error, na.rm = TRUE),
      rmse = sqrt(mean(squared_error, na.rm = TRUE)),
      .groups = "drop"
    )
}

# Test all weight combinations
optimization_results <- 
  weight_grid %>%
  rowwise() %>%
  mutate(
    metrics = list(calculate_blend_error(blend_analysis, model_weight, expert_weight, dampening))
  ) %>%
  unnest(metrics) %>%
  arrange(mae)

# --- Results Summary ---
best_blend <- optimization_results %>%
  arrange(rmse) %>%
  head(1)

optimal_rmse_results <-
  optimization_results %>%
  arrange(rmse) %>%
  head(10)

optimal_rmse_results

# --- Visualize Results ---
ggplot(optimization_results, aes(x = model_weight, y = rmse)) +
  geom_point(size = 2, alpha = 0.6, color = "#4682B4") +
  geom_point(data = best_blend, aes(x = model_weight, y = rmse), 
             color = "#FF6B6B", size = 4) +
  labs(
    title = "RMSE Across Model-Expert Weight Combinations",
    x = "Model Weight",
    y = "Root Mean Squared Error",
    subtitle = paste0("Optimal: ", round(best_blend$model_weight, 2), 
                      " model / ", round(best_blend$expert_weight, 2), " expert")
  ) +
  theme_minimal()

# --- Apply Optimal Blend to Test ---
final_blend <- 
  blend_analysis %>%
  mutate(
    final_prediction = best_blend$model_weight * Model_Prediction + 
                       best_blend$expert_weight * FantasyPros_Prediction
  ) %>%
  select(Player, Pos, Model_Prediction, FantasyPros_Prediction, 
         final_prediction, actual_points) %>%
  mutate(
    model_error = abs(Model_Prediction - actual_points),
    expert_error = abs(FantasyPros_Prediction - actual_points),
    blend_error = abs(final_prediction - actual_points)
  )

fwrite(final_blend, "tests/blend_optimization_results.csv")

final_blend %>%
  summarise(
    Model_MAE = mean(model_error),
    Expert_MAE = mean(expert_error),
    Blend_MAE = mean(blend_error)
  )

# ~~~~~ Position-Specific Blend Optimization ~~~~~

# --- Merge and Create Analysis Datasets ---
blend_analysis_receiving <- 
  actual_2025 %>%
  filter(Pos %in% c("WR", "TE")) %>%
  left_join(projections, by = c("Player", "Pos")) %>%
  drop_na(Model_Prediction, FantasyPros_Prediction, actual_points) %>%
  rename(actual_points = actual_points)

blend_analysis_rushing <- 
  actual_2025 %>%
  filter(Pos == "RB") %>%
  left_join(projections, by = c("Player", "Pos")) %>%
  drop_na(Model_Prediction, FantasyPros_Prediction, actual_points)

blend_analysis_passing <- 
  actual_2025 %>%
  filter(Pos == "QB") %>%
  left_join(projections, by = c("Player", "Pos")) %>%
  drop_na(Model_Prediction, FantasyPros_Prediction, actual_points)

# --- Grid Search for Optimal Weights ---
weight_grid <- expand_grid(
  model_weight = seq(0, 1, by = 0.05),
  expert_weight = seq(0, 1, by = 0.05)
) %>%
  filter(model_weight + expert_weight == 1) %>%
  mutate(dampening = 1)

# Function to calculate blend and error metrics
calculate_blend_error <- function(df, model_w, expert_w, damp = 1) {
  df %>%
    mutate(
      blended_pred = model_w * Model_Prediction + expert_w * (FantasyPros_Prediction * damp),
      error = blended_pred - actual_points,
      abs_error = abs(error),
      squared_error = error^2
    ) %>%
    summarise(
      mae = mean(abs_error, na.rm = TRUE),
      rmse = sqrt(mean(squared_error, na.rm = TRUE)),
      .groups = "drop"
    )
}

# --- Optimize for Each Position ---
optimization_receiving <- 
  weight_grid %>%
  rowwise() %>%
  mutate(
    metrics = list(calculate_blend_error(blend_analysis_receiving, model_weight, expert_weight, dampening))
  ) %>%
  unnest(metrics) %>%
  arrange(rmse) %>%
  mutate(position = "Receiving")

optimization_rushing <- 
  weight_grid %>%
  rowwise() %>%
  mutate(
    metrics = list(calculate_blend_error(blend_analysis_rushing, model_weight, expert_weight, dampening))
  ) %>%
  unnest(metrics) %>%
  arrange(rmse) %>%
  mutate(position = "Rushing")

optimization_passing <- 
  weight_grid %>%
  rowwise() %>%
  mutate(
    metrics = list(calculate_blend_error(blend_analysis_passing, model_weight, expert_weight, dampening))
  ) %>%
  unnest(metrics) %>%
  arrange(rmse) %>%
  mutate(position = "Passing")

# --- Results Summary ---
best_blend_receiving <- optimization_receiving %>% head(1)
best_blend_rushing <- optimization_rushing %>% head(1)
best_blend_passing <- optimization_passing %>% head(1)

optimal_results <- bind_rows(
  optimization_receiving %>% head(10),
  optimization_rushing %>% head(10),
  optimization_passing %>% head(10)
)

optimal_results

# --- Visualize Results by Position ---
ggplot(bind_rows(optimization_receiving, optimization_rushing, optimization_passing), 
       aes(x = model_weight, y = rmse, color = position)) +
  geom_point(size = 2, alpha = 0.6) +
  geom_point(data = bind_rows(best_blend_receiving, best_blend_rushing, best_blend_passing),
             aes(x = model_weight, y = rmse, color = position), size = 4) +
  facet_wrap(~position) +
  labs(
    title = "RMSE Across Model-Expert Weight Combinations by Position",
    x = "Model Weight",
    y = "Root Mean Squared Error",
    color = "Position"
  ) +
  theme_minimal()

### Final Results
# Optimal Weights by Position:
# Passing: 0.5 model / 0.5 expert
# Receiving: 0.6 model / 0.4 expert
# Rushing: 0.4 model / 0.6 expert