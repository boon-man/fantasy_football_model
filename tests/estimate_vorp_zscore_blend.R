##############################################################################
### Estimating weights to set for Relative Value & Z-Score Blending
### ADP source is now the Underdog Fantasy flat export (data/underdog_adp_rankings.csv),
### since FantasyPros removed free access to its ADP rankings.
library(readr)
library(data.table)
library(dplyr)
library(tidyr)
library(stringr)

PRED_YEAR <- 2026
SCORING_TYPE <- "HALF"

player_df <- read_csv(paste0("data/blended_proj_", as.character(PRED_YEAR), "_", SCORING_TYPE, ".csv"))

## SELECT EITHER UNDERDOG FANTASY OR ESPN ROSTER CUTOFF SUGGESTIONS
## Player pool size should be roughly 115% of total league roster spots available
# Underdog Fantasy Player Cutoffs
QB_CUTOFF <- 30
RB_CUTOFF <- 84
WR_CUTOFF <- 102
TE_CUTOFF <- 28

# Defining the Dampening variable to adjust final relative value scores
# z-scores get thrown off due to large expert projections
QB_DAMP <- 0.6
RB_DAMP <- 1.1
TE_DAMP <- 0.9
WR_DAMP <- 1.2

# Setting the VORP cutoff to identify how a "replacement" player performs at the position
VORP_CUTOFF <- 0.66

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

# Reading each player's best ball ADP from the Underdog Fantasy flat export.
# (FantasyPros removed free ADP access, so we swapped the scrape for data/underdog_adp_rankings.csv.)
# Underdog ships one row per player with separate name columns and a clean position in slotName,
# so we assemble the same Player / Pos / ADP shape the downstream regression join expects.
read_underdog_adp <- function(path = "data/underdog_adp_rankings.csv") {
  read_csv(path, show_col_types = FALSE) %>%
    transmute(
      Player = clean_player_name(paste(firstName, lastName)),  # match the join key format
      Pos = slotName,                                          # QB / RB / WR / TE, no rank suffix
      ADP = as.numeric(adp)
    ) %>%
    drop_na(ADP)
}

# Reading in ADP data
adp_df <- read_underdog_adp()

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
    
    # Combine Z and VORP with weights, multiplying VORP by 2 to increase distribution
    Relative_Value = 0.5 * z_score_value + 0.5 * vorp
  ) %>%
  select(Player, Pos, z_score_value, vorp)

# Merging ADP data with player projections to create regression dataframe
reg_df <- 
  total_df %>%
  left_join(adp_df, by = c("Player", "Pos")) %>%
  mutate(Value = -ADP) %>%
  drop_na(Value) # Some player names don't join on correctly, don't want to deal with cleaning

# Run linear regression
value_model <- lm(Value ~ z_score_value + vorp, data = reg_df)

# View weights
summary(value_model)

# Z Score coefficient: -.23
# VORP coefficient: -2.33
# VORP coefficient 2026: 1.30
