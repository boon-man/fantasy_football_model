##############################################################################
### Estimating weights to set for Relative Value & Z-Score Blending
library(rvest)
library(data.table)
library(rvest)
library(dplyr)
library(stringr)
library(janitor)

PRED_YEAR <- 2025

player_df <- read_csv(paste0("data/blended_proj_", as.character(PRED_YEAR), ".csv"))

## SELECT EITHER UNDERDOG FANTASY OR ESPN ROSTER CUTOFF SUGGESTIONS
## Player pool size should be roughly 115% of total league roster spots available
# Underdog Fantasy Player Cutoffs
QB_CUTOFF <- 30
RB_CUTOFF <- 84
WR_CUTOFF <- 102
TE_CUTOFF <- 28

# Defining the Dampening variable to adjust final relative value scores
# z-scores get thrown off due to large expert projections
QB_DAMP <- 0.55
TE_DAMP <- 0.9

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

# Scraping FantasyPros projections each player's best ball ADP
scrape_fp_projections <- function(position) {
  url <- paste0("https://www.fantasypros.com/nfl/adp/best-ball-overall.php")
  
  # Read and parse the HTML table
  adp_page <- read_html(url)
  
  adp_df <- adp_page %>%
    html_element("table") %>%
    html_table(fill = TRUE) %>%
    clean_names() %>%
    rename(
      Rank = rank,
      Player_Team_Bye = player_team_bye,
      Pos = pos,
      ADP = avg
    ) %>%
    mutate(
      # Extract player name from "Player Team (Bye)"
      Player = str_trim(str_remove(Player_Team_Bye, "\\s\\(.*\\)$")),  # remove (Bye)
      Player = str_remove(Player, "\\s+[A-Z]{2,3}$"),                  # remove team code (e.g., "Cin")
      Player = clean_player_name(Player),  # clean player name
      Pos = str_remove_all(Pos, "\\d+"),  # remove any numbers from position
      ADP = as.numeric(ADP),
      Rank = as.integer(Rank)
    ) %>%
    select(Player, Pos, ADP)
  
  return(adp_df)
}

# Reading in ADP data
adp_df <- 
  scrape_fp_projections() 

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
