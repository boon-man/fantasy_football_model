##############################################################################
### Overall Player Tiering
# First, an overall relative value tier will be created for each player regardless of position
# Next, we will create relative value tiers for each player respective to their position

# Reading in global variables if not already set
# 00_globals.R loads all package dependencies for the pipeline
source("00_globals.R")

## SELECT EITHER UNDERDOG FANTASY OR ESPN ROSTER CUTOFF SUGGESTIONS
## Player pool size should be roughly 115% of total league roster spots available
# Underdog Fantasy Player Cutoffs
QB_CUTOFF <- 30
RB_CUTOFF <- 82
WR_CUTOFF <- 104
TE_CUTOFF <- 28

# Draftkings Fantasy Player Cutoffs
# QB_CUTOFF <- 32
# RB_CUTOFF <- 90
# WR_CUTOFF <- 118
# TE_CUTOFF <- 36

# # ESPN Draft Cutoffs
# QB_CUTOFF <- 24
# RB_CUTOFF <- 51
# WR_CUTOFF <- 64
# TE_CUTOFF <- 25

# # Yahoo Fantasy Player Cutoffs
# QB_CUTOFF <- 26
# RB_CUTOFF <- 45
# WR_CUTOFF <- 55
# TE_CUTOFF <- 25

# Setting the VORP cutoff to identify how a "replacement" player performs at the position
VORP_CUTOFF <- 0.66

# Defining the Dampening variable to adjust final relative value scores
# z-scores get thrown off due to large expert projections
# Generally, WRs are going to be the most valuable position by ADP
# rankings should roughly reflect expected positional scarcity when drafting
QB_DAMP <- 0.6
RB_DAMP <- 1.1
TE_DAMP <- 0.9
WR_DAMP <- 1.2

# Function for estimating optimal K value
plot_wss_elbow <- function(player_df, pos = NULL, max_k = 25) {
  # Optionally filter by position
  df_attrs <- player_df %>%
    { if (!is.null(pos)) filter(., Pos == pos) else . } %>%
    select(Relative_Value)
  
  # Calculate WSS
  wss <- numeric(max_k)
  for (i in 1:max_k) {
    km.out <- kmeans(df_attrs, centers = i, nstart = 50)
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
  kmeans_result <- kmeans(pos_attrs, centers = k, nstart = 50)
  
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

# Reading in the player projection's data
player_df <- read_csv(paste0("data/blended_proj_", as.character(PRED_YEAR), "_", SCORING_TYPE, ".csv"))

# Trimming down the player pool to the positional cutoff points
player_df <-
  player_df %>%
  group_by(Pos) %>%
  mutate(FP_Pos_Ranking = dense_rank(desc(FantasyPros_Prediction)),
         Pos_Ranking = dense_rank(desc(Final_Projection))) %>%
  ungroup() %>%
  filter(Pos == "QB" & Pos_Ranking <= QB_CUTOFF |
           Pos == "RB" & Pos_Ranking <= RB_CUTOFF |
           Pos == "WR" & Pos_Ranking <= WR_CUTOFF |
           Pos == "TE" & Pos_Ranking <= TE_CUTOFF)

# Creating positional value over replacement (VORP) tiers
# What does the "replacement" point value for each positional group?
# These replacement values will be used to determine the relative value of each player
## Replacement level by position based on defined cutoffs
replacement_points <- 
  player_df %>%
  filter(
    (Pos == "QB" & Pos_Ranking == round(QB_CUTOFF * VORP_CUTOFF)) |
      (Pos == "RB" & Pos_Ranking == round(RB_CUTOFF * VORP_CUTOFF)) |
      (Pos == "WR" & Pos_Ranking == round(WR_CUTOFF * VORP_CUTOFF)) |
      (Pos == "TE" & Pos_Ranking == round(TE_CUTOFF * VORP_CUTOFF))
  ) %>%
  select(Pos, Replacement_Value = Final_Projection)


########################## Overall Tiers #######################################
# Creating a "Relative Value" metric based on how valuable a player relative to the rest of the player pool above the positional cutoffs
# Ranking each player by their Final Projection value
# Filtering data so that only the top N players by position group (defined by above cutoffs) are included in the final rankings
total_df <-
  player_df %>%
  left_join(replacement_points, by = "Pos") %>%
  group_by(Pos) %>%
  arrange(Pos_Ranking, .by_group = TRUE) %>%
  mutate(
    # Z-score scaled projection
    z_score_value = ((Final_Projection - mean(Final_Projection)) / sd(Final_Projection)) * Final_Projection,
    
    # VORP = projection - replacement-level projection
    vorp = (Final_Projection - Replacement_Value),
    
    # Combine Z and VORP with weights, multiplying VORP by 1.3 to increase distribution for value estimation accuracy (1.3 was selected after regression analysis, see estimate_vorp_zscore..)
    Relative_Value = 0.5 * z_score_value + 0.5 * (vorp * 1.3)
  ) %>%
  # Adjusting the Relative Value based on position, QBs are generally less valuable in best ball fantasy football, so we reduce their value slightly
  mutate(
    Pos_Adjustment = case_when(
      Pos == "QB" ~ QB_DAMP,   # Reduce QB value
      Pos == "TE" ~ TE_DAMP,   # Reduce TE value
      Pos == "RB" ~ RB_DAMP,   # Reduce RB value
      Pos == "WR" ~ WR_DAMP,   # Increase WR value
      TRUE ~ 1
    ),
    Relative_Value = Relative_Value * Pos_Adjustment
  ) %>%
  mutate(Relative_Value = sort(Relative_Value, decreasing = TRUE)) %>% # Forcing player rankings to be in direct value order, z-scoring can get funky sometimes
  ungroup() %>%
  mutate(Overall_Ranking = dense_rank(desc(Relative_Value))) %>%
  arrange(Pos, desc(Relative_Value))

plot_wss_elbow(total_df, max_k = 25)

total_attrs <-
  total_df %>%
  select(Relative_Value)

#### Obtaining clusters with optimal K value
kmeans_attrs <- kmeans(total_attrs, centers = 9, nstart = 50)

total_df$Tier <- kmeans_attrs$cluster

# Build the tier dataset straight from total_df, which already carries every needed column.
# This replaces a name-based self-join that collided for same-named players (e.g. the two
# Adrian Petersons). Re-rank by cluster mean so Tier 1 is the most valuable group.
tier_df <-
  total_df %>%
  select(player_id, Player, Pos, Model_Prediction, FantasyPros_Prediction, Final_Projection,
         Floor, Ceiling, upside_index, ceiling_index, floor_index,   # model's own range/upside, carried to the final sheet
         FP_Pos_Ranking, Pos_Ranking, Overall_Ranking, Relative_Value, Tier) %>%
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

qb_df <- assign_positional_tiers(tier_df, pos = "QB", k = 8)

## Runningbacks
plot_wss_elbow(tier_df, pos = "RB", max_k = 15)

rb_df <- assign_positional_tiers(tier_df, pos = "RB", k = 7)

## Wide Receivers
plot_wss_elbow(tier_df, pos = "WR", max_k = 15)

wr_df <- assign_positional_tiers(tier_df, pos = "WR", k = 6)

## Tight Ends
plot_wss_elbow(tier_df, pos = "TE", max_k = 15)

te_df <- assign_positional_tiers(tier_df, pos = "TE", k = 6)

# Combine all positional tiers into a single dataframe
final_df <- bind_rows(list(qb_df, rb_df, wr_df, te_df)) %>%
  arrange(desc(Relative_Value), Tier, Pos_Tier) %>%
  select(player_id, Player, Pos, Model_Prediction, FantasyPros_Prediction,
         Final_Projection, Floor, Ceiling, ceiling_index, floor_index, upside_index,
         FP_Pos_Ranking, Pos_Ranking, Overall_Ranking,
         Relative_Value, Tier, Pos_Tier)

# TODO: create a scatterplot of player projections, with color grouping by position tier?

########################## MISSION COMPLETE ####################################
# Save the final dataframe with tiers
fwrite(final_df, paste0("data/final_projections_", as.character(PRED_YEAR), "_", SCORING_TYPE, ".csv"))

