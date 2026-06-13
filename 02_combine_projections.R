##############################################################################
### Combining my predictions with FantasyPros projection data

# Defining Variables, ensuring global variables are read in
# 00_globals.R loads all package dependencies for the pipeline
source("00_globals.R")
positions <- c("rb", "qb", "wr", "te")

## I'm gonna favor my model a bit
PRED_WEIGHT    <- 0.62   # Weight on my model's prediction
PROJ_WEIGHT    <- 0.38   # Weight on FantasyPros projection
PROJ_DAMP        <- 0.95  # Dampening factor for expert projections, they are pretty aggressive
QB_PENALTY_FACTOR <- 0.85  # Penalty if only expert projection is available (rookies, players that were injured all of last season)
SKILL_PENALTY_FACTOR <- 0.9  # Penalty if only expert projection is available (rookies, players that were injured all of last season)

# Scraping FantasyPros projections for each position
scrape_fp_projections <- function(position, scoring = "HALF") {
  url <- paste0("https://www.fantasypros.com/nfl/projections/", position, ".php?week=draft&scoring=", scoring, "&week=draft")
  
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

# Reading in player prediction dataframe
player_df <- read_csv(paste0("data/model_pred_", as.character(PRED_YEAR), "_", SCORING_TYPE, ".csv"))

# Pulling in fantasypros predictions
# Dropping inadvertent string rows from the projections column
fp_projections <- 
  bind_rows(lapply(positions, function(pos) scrape_fp_projections(pos, scoring = SCORING_TYPE))) %>%
  mutate(Projected_Points = as.numeric(str_remove(Projected_Points, "[A-Za-z]"))) %>%
  drop_na(Projected_Points)


# Apply cleaning to both datasets
player_df_cleaned <-
  player_df %>%
  mutate(Player_clean = clean_player_name(Player))

fp_cleaned <- 
  fp_projections %>%
  mutate(Player_clean = clean_player_name(Player))

# Join using cleaned names (FantasyPros ships no player_id), but keep the model's unique
# player_id so same-named players stay distinct. When two same-named, same-position model
# players (e.g. the two Adrian Petersons) both match a single FantasyPros row, the expert
# projection is kept only on the highest-projected (active) player; the other falls back to a
# model-only blend and is dropped later by the no-FantasyPros filter when it has no projection.
combined_projections <-
  player_df_cleaned %>%
  full_join(fp_cleaned, by = c("Player_clean", "Pos"), relationship = "many-to-many") %>%
  group_by(Player_clean, Pos) %>%
  mutate(
    is_fp_primary = is.na(Predicted) | row_number(desc(Predicted)) == 1,
    Projected_Points = if_else(is_fp_primary, Projected_Points, NA_real_),
    Player.y = if_else(is_fp_primary, Player.y, NA_character_)
  ) %>%
  ungroup() %>%
  # Floor/Ceiling/implied_upside are carried through untouched - they stay the model's own
  # expectation (not manually adjusted, not blended), shown beside the blended Final_Projection
  select(player_id, Player = Player.x, Pos, Predicted, FantasyPros_Player = Player.y, Projected_Points,
         Floor, Ceiling, implied_upside) %>%
  mutate(PosGroup = if_else(Pos == "QB", "QB", "SKILL")) # Grouping QBs separately from skill positions for blending

# Adjusting model predictions for players that are expected to miss games during the upcoming season
combined_projections <- combined_projections %>%
  mutate(
    Predicted = case_when(
      Player == "Alexander Mattison" ~ Predicted * 0.5, # Banished to bench warming
      Player == "Antonio Gibson" ~ Predicted * 0.5, # Banished to bench warming
      Player == "Austin Ekeler" ~ Predicted * 0.8, # Banished to bench warming
      Player == "Brandon Aiyuk" ~ Predicted * 0.6, # Injury
      Player == "Brian Robinson" ~ Predicted * 0.85, # Team hates him?
      Player == "Bucky Irving" ~ Predicted * 1.25, # Now a starter
      Player == "Cam Akers" ~ Predicted * 0.25, # Banished to bench warming
      Player == "Chase Brown" ~ Predicted * 1.2, # Now a starter
      Player == "Chris Godwin" ~ Predicted * 0.75, # Recovering from injury
      Player == "Christian Watson" ~ Predicted * 0.40, # Torn ACL
      Player == "Clyde Edwardshelaire" ~ Predicted * 0.33, # PTSD?
      Player == "Cole Kmet" ~ Predicted * 0.8, # Now a backup
      Player == "Cordarrelle Patterson" ~ Predicted * 0.33, # Cut, washed
      Player == "Darnell Mooney" ~ Predicted * 0.85, # Injury
      Player == "Dameon Pierce" ~ Predicted * 0.50, # Banished to bench warming
      Player == "Deandre Hopkins" ~ Predicted * 0.75, # Washed, now a slot receiver
      Player == "Emmanuel Wilson" ~ Predicted * 0.5, # Banished to bench warming
      Player == "Jakobi Meyers" ~ Predicted * 0.9, # Hates his team
      Player == "Jayden Reed" ~ Predicted * 0.75, # Injury
      Player == "Jaylen Warren" ~ Predicted * 1.25, # Now a projected starter
      Player == "Joe Mixon" ~ Predicted * 0.6,       # Broken foot
      Player == "Jonnu Smith" ~ Predicted * 0.8, # Aaron Rodgers
      Player == "Jordan Addison" ~ Predicted * 0.85, # DUI
      Player == "Jordan Mason" ~ Predicted * 1.25, # Splitting starting duties
      Player == "Keenan Allen" ~ Predicted * 0.8, # No longer a starter
      Player == "Michael Carter" ~ Predicted * 0.25, # Banished to bench warming
      Player == "Michael Pittman" ~ Predicted * 0.85, # Anthony Richardson
      Player == "Mike Williams" ~ Predicted * 0.00,    # Retiring
      Player == "Najee Harris" ~ Predicted * 0.7,     # Blew his damn eyes off
      Player == "Nick Chubb" ~ Predicted * 0.9,       # Broken foot
      Player == "Pierre Strong" ~ Predicted * 0.25, # Banished to bench warming
      Player == "Rachaad White" ~ Predicted * 0.8, # Now a backup
      Player == "Rashee Rice" ~ Predicted * 0.62,     # DUI
      Player == "Ricky Pearsall" ~ Predicted * 1.35, # Was shot, only WR left on team
      Player == "Russell Wilson" ~ Predicted * 0.8, # Model needs to chill out with Russ a bit
      Player == "Sam Darnold" ~ Predicted * 0.85, # Model needs to chill out with Darnold a bit
      Player == "Stefon Diggs" ~ Predicted * 0.9,     # Torn ACL
      Player == "Tank Dell" ~ Predicted * 0.10,        # Torn ACL
      Player == "Tyrone Tracy" ~ Predicted * 1.2, # Now a starter
      Player == "Xavier Worthy" ~ Predicted * 1.35,    # Teammate has DUI
      TRUE ~ Predicted
    )
  )


## Blending the combined projections into a final blended_projection step
blended_df <- 
  combined_projections %>%
  mutate(
    final_projection = case_when(
      !is.na(Predicted) & !is.na(Projected_Points) ~
        PRED_WEIGHT * Predicted +
        (PROJ_WEIGHT * Projected_Points * PROJ_DAMP),
      
      !is.na(Predicted) ~ Predicted,
      
      is.na(Predicted) & !is.na(Projected_Points) ~
        PROJ_DAMP * Projected_Points *
        if_else(PosGroup == "QB", QB_PENALTY_FACTOR, SKILL_PENALTY_FACTOR),
      
      TRUE ~ NA_real_
    )
  )

final_df <- 
  blended_df %>%
  mutate(Player = coalesce(Player, FantasyPros_Player)) %>%
  select(player_id, Player, Pos,    # player_id is NA for FantasyPros-only rookies (no model row)
         Model_Prediction = Predicted,
         FantasyPros_Prediction = Projected_Points,
         Final_Projection = final_projection,
         Floor, Ceiling, implied_upside) %>%  # model's own range, carried through untouched
  arrange(Pos, desc(Final_Projection)) %>%
  filter(!is.na(FantasyPros_Prediction)) # Removing players without a FantasyPros Projection, these players are out of the league or retired


# Let's see if we cooked here
plot_predicted_trajectories(combined, final_df, pos_group = "QB", sample_n = 8)
plot_predicted_trajectories(combined, final_df, pos_group = "RB", sample_n = 8)
plot_predicted_trajectories(combined, final_df, pos_group = "WR", sample_n = 8)
plot_predicted_trajectories(combined, final_df, pos_group = "TE", sample_n = 8)

fwrite(final_df, paste0("data/blended_proj_", as.character(PRED_YEAR), "_", SCORING_TYPE, ".csv"))
