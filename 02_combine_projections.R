##############################################################################
### Combining my predictions with FantasyPros projection data
library(rvest)
library(dplyr)
library(stringr)
library(tidyverse)
library(data.table)

# Defining Variables
EVAL_YEAR <- 2024
PRED_YEAR <- 2025
positions <- c("rb", "qb", "wr", "te")

# Blended projection: 64% model, 36% FantasyPros
## Personal bias making the blend ensure that Justin Jefferson is a top 4 receiver
PRED_WEIGHT <- 0.64 # my model prediction weight
PROJ_WEIGHT <- 0.36 # fp projection weight
FANTASYPROS_WEIGHT <- 0.81 # Decreasing FantasyPros projections, they are fairly aggressive relative to model predictions

# Reading in player prediction dataframe
player_df <- read_csv(paste0("data/model_pred_", as.character(PRED_YEAR), ".csv"))

# Scraping FantasyPros projections for each position
scrape_fp_projections <- function(position) {
  url <- paste0("https://www.fantasypros.com/nfl/projections/", position, ".php?week=draft&scoring=HALF&week=draft")
  
  page <- read_html(url)
  
  table <- page %>%
    html_node("table") %>%
    html_table(fill = TRUE)
  
  table <- table %>%
    rename_with(~ str_trim(.x)) %>%
    rename(Player = 1, Projected_Points = ncol(.)) %>%
    mutate(
      Player = str_remove(Player, "\\s+[A-Z]{2,3}$"),  # Remove team abbrev
      Player = str_squish(Player),
      Pos = toupper(position)
    ) %>%
    select(Player, Pos, Projected_Points)
  
  return(table)
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

# Plotting predicted player trajectories
plot_predicted_trajectories <- function(combined_df, pred_df, pos_group = "QB", sample_n = 10, pred_year = as.Date("2025-01-01")) {
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
  
  # 2. Predicted values with hardcoded prediction year
  preds <- pred_df %>%
    filter(Pos == pos_group) %>%
    mutate(Year = pred_year, points = Final_Projection) %>%
    select(Player, Year, points)
  
  # 3. Combine both
  full_df <- bind_rows(hist_df, preds) %>%
    group_by(Player) %>%
    filter(n() > 1) %>%
    ungroup()
  
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
      plot.title = element_text(colour = "#808080", size = 16, face = "bold", hjust = 0.5),
      axis.title = element_text(colour = "#808080", size = 14),
      axis.text = element_text(colour = "#808080", size = 12),
      panel.background = element_rect(fill = "#f7e8d7", color = NA),
      plot.background = element_rect(fill = "#f7e8d7", color = NA),
      legend.position = "none",
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_line(color = "#FFFFFF", linewidth = 0.4),
      axis.line = element_line(color = "#FFFFFF", size = 0.4) 
    )
}


# Pulling in fantasypros predictions
# Dropping inadvertent string rows from the projections column
fp_projections <- 
  bind_rows(lapply(positions, scrape_fp_projections)) %>%
  mutate(Projected_Points = as.numeric(str_remove(Projected_Points, "[A-Za-z]"))) %>%
  drop_na(Projected_Points)


# Apply cleaning to both datasets
## TODO: make sure the updates to the combined dataset to fix player names in the previous step worked out
player_df_cleaned <-
  player_df %>%
  mutate(Player_clean = clean_player_name(Player))

fp_cleaned <- 
  fp_projections %>%
  mutate(Player_clean = clean_player_name(Player))

# Join using cleaned names, keep original names for reference
combined_projections <- 
  player_df_cleaned %>%
  full_join(fp_cleaned, by = c("Player_clean", "Pos")) %>%
  select(Player = Player.x, Pos, Predicted, FantasyPros_Player = Player.y, Projected_Points)

# Blending the combined projections into a final blended_projection step
combine_projections <- 
  combined_projections %>% # Marvin Harrison's prediction is warped due to sharing a name with his father, 
  mutate(Predicted = if_else(Player == "Marvin Harrison", NA_real_, Predicted)) %>%
  mutate(
    # Ensure numeric columns
    Predicted = as.numeric(Predicted),
    Projected_Points = as.numeric(Projected_Points),
    
    # Creating a final blended projection using weights and FantasyPros adjustment
    final_projection = case_when(
      !is.na(Predicted) & !is.na(Projected_Points) ~
        PRED_WEIGHT * Predicted + PROJ_WEIGHT * (FANTASYPROS_WEIGHT * Projected_Points),
      !is.na(Predicted) ~ Predicted,
      !is.na(Projected_Points) ~ FANTASYPROS_WEIGHT * Projected_Points,
      TRUE ~ NA_real_
    )
  )

final_df <- 
  combine_projections %>%
  mutate(Player = coalesce(Player, FantasyPros_Player)) %>%
  select(Player, Pos, 
         Model_Prediction = Predicted,
         FantasyPros_Prediction = Projected_Points,
         Final_Projection = final_projection) %>%
  arrange(Pos, desc(Final_Projection)) %>%
  filter(!is.na(FantasyPros_Prediction)) # Removing players without a FantasyPros Projection, these players are out of the league or retired


# Let's see if we cooked here
plot_predicted_trajectories(combined, final_df, pos_group = "QB", sample_n = 8)
plot_predicted_trajectories(combined, final_df, pos_group = "RB", sample_n = 8)
plot_predicted_trajectories(combined, final_df, pos_group = "WR", sample_n = 8)
plot_predicted_trajectories(combined, final_df, pos_group = "TE", sample_n = 8)

fwrite(final_df, paste0("data/blended_proj_", as.character(PRED_YEAR), ".csv"))
