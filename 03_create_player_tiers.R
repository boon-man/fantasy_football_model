##############################################################################
### Overall Player Tiering
# First, an overall relative value tier will be created for each player regardless of position
# Next, we will create relative value tiers for each player respective to their position
library(tidyverse)
library(data.table)

PRED_YEAR <- 2025

player_df <- read_csv(paste0("data/blended_proj_", as.character(PRED_YEAR), ".csv"))

## SELECT EITHER UNDERDOG FANTASY OR ESPN ROSTER CUTOFF SUGGESTIONS
## Player pool size should be roughly 115% of total league roster spots available
# Underdog Fantasy Player Cutoffs
QB_CUTOFF <- 30
RB_CUTOFF <- 84
WR_CUTOFF <- 102
TE_CUTOFF <- 28

# # ESPN Draft Cutoffs
# QB_CUTOFF <- 27
# RB_CUTOFF <- 60
# WR_CUTOFF <- 80
# TE_CUTOFF <- 26

# Function for estimating optimal K value
plot_wss_elbow <- function(player_df, pos = NULL, max_k = 25) {
  library(dplyr)
  library(ggplot2)
  
  # Optionally filter by position
  df_attrs <- player_df %>%
    { if (!is.null(pos)) filter(., Pos == pos) else . } %>%
    select(Relative_Value)
  
  # Calculate WSS
  wss <- numeric(max_k)
  for (i in 1:max_k) {
    km.out <- kmeans(df_attrs, centers = i, nstart = 20)
    wss[i] <- km.out$tot.withinss
  }
  
  # Plot elbow chart
  elbow_df <- data.frame(k = 1:max_k, wss = wss)
  ggplot(elbow_df, aes(x = k, y = wss)) +
    geom_line(color = "#3A6C99", linewidth = .75) +
    geom_point(color = "#4682B4", size = 1.5) +
    labs(
      title = paste("Elbow Plot for", pos, "Cluster WSS"),
      x = "Number of Clusters (k)",
      y = "Within-Cluster Sum of Squares"
    ) +
    theme_minimal()
}

# Function for creating player positional tiers
assign_positional_tiers <- function(player_df, pos, k = 5) {
  
  # Filtering total df to the position group
  pos_df <-
    player_df %>%
    filter(Pos == pos)
  
  # Filter to the positional group and extract attribute for clustering
  pos_attrs <- 
    pos_df %>%
    select(Relative_Value)

  # Run k-means clustering
  set.seed(123)  # for reproducibility
  kmeans_result <- kmeans(pos_attrs, centers = k, nstart = 20)
  
  # Assign tier back to dataframe, adjust tier labels to correspond to player value
  pos_df <- 
    pos_df %>%
    mutate(Pos_Tier = kmeans_result$cluster) %>%
    group_by(Pos_Tier) %>%
    mutate(tier_avg = mean(Relative_Value)) %>%
    ungroup() %>%
    mutate(Pos_Tier = dense_rank(desc(tier_avg))) %>%
    arrange(desc(Relative_Value)) %>%
    select(-tier_avg)
  
  return(pos_df)
}

########################## Overall Tiers #######################################
# Creating a "Relative Value" metric based on how valuable a player relative to the rest of the player pool above the positional cutoffs
# Ranking each player by their Final Projection value
# Filtering data so that only the top N players by position group (defined by above cutoffs) are included in the final rankings
total_df <-
  player_df %>%
  group_by(Pos) %>%
  mutate(FP_Pos_Ranking = dense_rank(desc(FantasyPros_Prediction)),
         Pos_Ranking = dense_rank(desc(Final_Projection))) %>%
  ungroup() %>%
  filter(Pos == "QB" & Pos_Ranking <= QB_CUTOFF |
           Pos == "RB" & Pos_Ranking <= RB_CUTOFF |
           Pos == "WR" & Pos_Ranking <= WR_CUTOFF |
           Pos == "TE" & Pos_Ranking <= TE_CUTOFF) %>%
  group_by(Pos) %>%
  arrange(Pos_Ranking, .by_group = TRUE) %>%
  mutate(Relative_Value = ((Final_Projection - mean(Final_Projection)) / sd(Final_Projection)) * Final_Projection) %>%
  mutate(Relative_Value = sort(Relative_Value, decreasing = TRUE)) %>% # Ensuring that Relative Values are sorted correctly
  ungroup() %>%
  mutate(Overall_Ranking = dense_rank(desc(Relative_Value))) %>%
  arrange(Pos, desc(Relative_Value))

plot_wss_elbow(total_df, max_k = 25)

total_attrs <-
  total_df %>%
  select(Relative_Value)

#### Obtaining clusters with optimal K value
kmeans_attrs <- kmeans(total_attrs, centers = 10, nstart = 20)

total_df$Tier <- kmeans_attrs$cluster

final_df <- 
  total_df %>%
  select(Player, FP_Pos_Ranking, Pos_Ranking, Overall_Ranking, Relative_Value, Tier)

# Creating overall tiering dataset
tier_df <- 
  player_df %>% 
  inner_join(final_df, by = 'Player') %>%
  group_by(Tier) %>%
  mutate(tier_avg = mean(Relative_Value)) %>%
  ungroup() %>%
  mutate(Tier = dense_rank(desc(tier_avg))) %>%
  arrange(Pos, desc(Relative_Value)) %>%
  select(-tier_avg)

########################## Positional Tiers ####################################
# Assign positional tiers for each position

## Quarterbacks
plot_wss_elbow(tier_df, pos = "QB", max_k = 15)

qb_df <- assign_positional_tiers(tier_df, pos = "QB", k = 6)

## Runningbacks
plot_wss_elbow(tier_df, pos = "RB", max_k = 15)

rb_df <- assign_positional_tiers(tier_df, pos = "RB", k = 6)

## Wide Receivers
plot_wss_elbow(tier_df, pos = "WR", max_k = 15)

wr_df <- assign_positional_tiers(tier_df, pos = "WR", k = 7)

## Tight Ends
plot_wss_elbow(tier_df, pos = "TE", max_k = 15)

te_df <- assign_positional_tiers(tier_df, pos = "TE", k = 6)

# Combine all positional tiers into a single dataframe
final_df <- bind_rows(list(qb_df, rb_df, wr_df, te_df)) %>%
  arrange(desc(Relative_Value), Tier, Pos_Tier)


########################## MISSION COMPLETE ####################################
# Save the final dataframe with tiers
fwrite(final_df, paste0("data/final_projections_", as.character(PRED_YEAR), ".csv"))
