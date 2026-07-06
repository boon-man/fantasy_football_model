##############################################################################
### Combining my predictions with FantasyPros projection data

# Defining Variables, ensuring global variables are read in
# 00_globals.R loads all package dependencies for the pipeline
source("00_globals.R")
source("evaluate_model.R")  # Loading the model performance diagnostic plots
positions <- c("rb", "qb", "wr", "te")

## Per-position weight on my model's prediction vs the FantasyPros projection.
## Keyed by Pos and looked up per row; the projection weight is the complement (1 - pred weight)
## so the pair always sums to 1. QB and receivers lean on the model, RB leans on expert consensus.
PRED_WEIGHTS <- c(QB = 0.50, RB = 0.40, WR = 0.60, TE = 0.60)
PROJ_DAMP        <- 0.95  # Dampening factor for expert projections, they are pretty aggressive
QB_PENALTY_FACTOR <- 0.85  # Penalty if only expert projection is available (rookies, players that were injured all of last season)
SKILL_PENALTY_FACTOR <- 0.9  # Penalty if only expert projection is available (rookies, players that were injured all of last season)

# Scraping FantasyPros projections for each position.
# FantasyPros intermittently answers bare user-agents (and rapid-fire requests) with a
# truncated ~10-row "preview" of the projections table, so we send a browser user-agent and
# re-request with a short back-off until the full position list comes back (min_rows guard).
scrape_fp_projections <- function(position, scoring = "HALF", min_rows = 20, max_tries = 4) {
  url <- paste0("https://www.fantasypros.com/nfl/projections/", position, ".php?week=draft&scoring=", scoring, "&week=draft")

  # One request -> parsed projections table. Reading the response text through a browser
  # user-agent avoids the stripped-down table rvest's default agent sometimes receives.
  fetch_once <- function() {
    resp <- GET(
      url,
      user_agent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
      add_headers("Accept-Language" = "en-US,en;q=0.9")
    )
    read_html(content(resp, as = "text", encoding = "UTF-8")) %>%
      html_node("table") %>%
      html_table(fill = TRUE)
  }

  # Retry while the table looks truncated (the preview table only carries ~10 players)
  table <- fetch_once()
  tries <- 1
  while (nrow(table) < min_rows && tries < max_tries) {
    Sys.sleep(3)
    table <- fetch_once()
    tries <- tries + 1
  }
  if (nrow(table) < min_rows) {
    warning(sprintf("FantasyPros %s returned only %d rows after %d tries", position, nrow(table), tries))
  }

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

# Dumbbell chart comparing my model prediction against the FantasyPros projection for the
# top_n players in a position group (ranked by the blended Final_Projection). Each player is a
# row with two dots - model vs expert - joined by a connector, so the gap between the two reads
# at a glance. Colors come from NFL_COLOR_PALETTE (defined in evaluate_model.R, present in the
# session after 01 has been run).
plot_pred_vs_proj_dumbbell <- function(projection_df, pos_group = "QB", top_n = 30) {
  # Rank the position group by the blended projection and keep the top N players
  ranked <-
    projection_df %>%
    filter(Pos == pos_group,
           !is.na(Model_Prediction),
           !is.na(FantasyPros_Prediction)) %>%
    arrange(desc(Final_Projection)) %>%
    slice_head(n = top_n) %>%
    # Order the y-axis so the highest projected player sits at the top of the chart
    mutate(Player = factor(Player, levels = rev(Player)))

  # Long form drives the two colored dots, the wide ranked frame anchors the connector endpoints
  points_long <-
    ranked %>%
    select(Player, Model = Model_Prediction, FantasyPros = FantasyPros_Prediction) %>%
    pivot_longer(c(Model, FantasyPros), names_to = "Source", values_to = "Points") %>%
    mutate(Source = factor(Source, levels = c("Model", "FantasyPros")))

  ggplot() +
    # Dashed horizontal separators every 10 players, counted from the top of the ranking.
    # y is discrete (1..top_n, highest player on top), so minor gridlines won't render -
    # we draw the lines explicitly at the .5 boundary between each block of 10.
    geom_hline(
      yintercept = top_n - seq(5, top_n - 1, by = 5) + 0.5,
      linetype = "dashed", color = "#00000F", linewidth = 0.4, alpha = 0.25
    ) +
    # Connector between the model and expert estimate for each player
    geom_segment(
      data = ranked,
      aes(y = Player, yend = Player, x = Model_Prediction, xend = FantasyPros_Prediction),
      color = "#C2C2C2", linewidth = 1
    ) +
    # The two estimate dots, colored by source
    geom_point(
      data = points_long,
      aes(y = Player, x = Points, color = Source),
      size = 3
    ) +
    # Black tick marking the blended Final_Projection that sits between the two estimates
    geom_point(
      data = ranked,
      aes(y = Player, x = Final_Projection, shape = "Blended Projection"),
      color = "#000000", size = 1
    ) +
    scale_color_manual(
      values = c(Model = NFL_COLOR_PALETTE[1], FantasyPros = NFL_COLOR_PALETTE[3]),
      breaks = c("Model", "FantasyPros"),
      labels = c("Model Prediction", "FantasyPros Projection")
    ) +
    scale_shape_manual(values = c("Blended Projection" = 18)) +
    # Major gridlines every 100 points, minor lines added at the 50-point marks between them
    scale_x_continuous(breaks = seq(0, 1000, by = 100), minor_breaks = seq(0, 1000, by = 50)) +
    labs(
      title = paste0("Model vs FantasyPros — Top ", top_n, " ", pos_group, "s"),
      x = "Projected Fantasy Points",
      y = NULL,
      color = NULL,
      shape = NULL
    ) +
    theme_minimal(base_size = 13) +
    theme(
      plot.title = element_text(colour = "#262626", size = 16, face = "bold"),
      plot.subtitle = element_text(colour = "#595959", size = 11),
      axis.text = element_text(colour = "#262626"),
      legend.position = "top",
      panel.grid.minor.y = element_blank(),
      panel.grid.major.y = element_blank(),
      panel.grid.major.x = element_line(color = "#E6E6E6", linewidth = 0.5),
      # Thinner minor vertical gridlines at the 50-point marks between the majors
      panel.grid.minor.x = element_line(color = "#EFEFEF", linewidth = 0.3)
    )
}


# Reading in player prediction dataframe
player_df <- read_csv(paste0("data/model_pred_", as.character(PRED_YEAR), "_", SCORING_TYPE, ".csv"))

# Pulling in fantasypros predictions
# Pausing between positions so FantasyPros doesn't throttle the back-to-back requests down
# to its truncated preview table, then dropping the header/string rows from the points column
fp_projections <-
  bind_rows(lapply(positions, function(pos) {
    out <- scrape_fp_projections(pos, scoring = SCORING_TYPE)
    Sys.sleep(3)
    out
  })) %>%
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
  # Floor/Ceiling/upside_index/ceiling_index/floor_index are carried through untouched - they stay the
  # model's own expectation (not manually adjusted, not blended), shown beside the blended Final_Projection
  select(player_id, Player = Player.x, Pos, Predicted, FantasyPros_Player = Player.y, Projected_Points,
         Floor, Ceiling, upside_index, ceiling_index, floor_index) %>%
  mutate(PosGroup = if_else(Pos == "QB", "QB", "SKILL")) # Grouping QBs separately from skill positions for blending

# Adjusting model predictions for players that are expected to miss games during the upcoming season
combined_projections <- combined_projections %>%
  mutate(
    Predicted = case_when(
      Player == "Alexander Mattison" ~ Predicted * 1.0, # Banished to bench warming
      Player == "Xavier Worthy" ~ Predicted * 1.00,    # Teammate has DUI
      TRUE ~ Predicted
    )
  )


## Blending the combined projections into a final blended_projection step
blended_df <-
  combined_projections %>%
  mutate(
    # Look up each player's model weight by position, projection weight is the complement
    pred_w = PRED_WEIGHTS[Pos],
    proj_w = 1 - pred_w,
    final_projection = case_when(
      !is.na(Predicted) & !is.na(Projected_Points) ~
        pred_w * Predicted +
        (proj_w * Projected_Points * PROJ_DAMP),

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
         Floor, Ceiling, ceiling_index, floor_index, upside_index) %>%  # model's own range, carried through untouched
  arrange(Pos, desc(Final_Projection)) %>%
  filter(!is.na(FantasyPros_Prediction)) # Removing players without a FantasyPros Projection, these players are out of the league or retired


# Let's see if we cooked here - model prediction vs FantasyPros projection per position
plot_pred_vs_proj_dumbbell(final_df, pos_group = "QB", top_n = 30)
plot_pred_vs_proj_dumbbell(final_df, pos_group = "RB", top_n = 30)
plot_pred_vs_proj_dumbbell(final_df, pos_group = "WR", top_n = 30)
plot_pred_vs_proj_dumbbell(final_df, pos_group = "TE", top_n = 30)

fwrite(final_df, paste0("data/blended_proj_", as.character(PRED_YEAR), "_", SCORING_TYPE, ".csv"))


      # Player == "Alexander Mattison" ~ Predicted * 0.5, # Banished to bench warming
      # Player == "Antonio Gibson" ~ Predicted * 0.5, # Banished to bench warming
      # Player == "Austin Ekeler" ~ Predicted * 0.8, # Banished to bench warming
      # Player == "Brandon Aiyuk" ~ Predicted * 0.6, # Injury
      # Player == "Brian Robinson" ~ Predicted * 0.85, # Team hates him?
      # Player == "Bucky Irving" ~ Predicted * 1.25, # Now a starter
      # Player == "Cam Akers" ~ Predicted * 0.25, # Banished to bench warming
      # Player == "Chase Brown" ~ Predicted * 1.2, # Now a starter
      # Player == "Chris Godwin" ~ Predicted * 0.75, # Recovering from injury
      # Player == "Christian Watson" ~ Predicted * 0.40, # Torn ACL
      # Player == "Clyde Edwardshelaire" ~ Predicted * 0.33, # PTSD?
      # Player == "Cole Kmet" ~ Predicted * 0.8, # Now a backup
      # Player == "Cordarrelle Patterson" ~ Predicted * 0.33, # Cut, washed
      # Player == "Darnell Mooney" ~ Predicted * 0.85, # Injury
      # Player == "Dameon Pierce" ~ Predicted * 0.50, # Banished to bench warming
      # Player == "Deandre Hopkins" ~ Predicted * 0.75, # Washed, now a slot receiver
      # Player == "Emmanuel Wilson" ~ Predicted * 0.5, # Banished to bench warming
      # Player == "Jakobi Meyers" ~ Predicted * 0.9, # Hates his team
      # Player == "Jayden Reed" ~ Predicted * 0.75, # Injury
      # Player == "Jaylen Warren" ~ Predicted * 1.25, # Now a projected starter
      # Player == "Joe Mixon" ~ Predicted * 0.6,       # Broken foot
      # Player == "Jonnu Smith" ~ Predicted * 0.8, # Aaron Rodgers
      # Player == "Jordan Addison" ~ Predicted * 0.85, # DUI
      # Player == "Jordan Mason" ~ Predicted * 1.25, # Splitting starting duties
      # Player == "Keenan Allen" ~ Predicted * 0.8, # No longer a starter
      # Player == "Michael Carter" ~ Predicted * 0.25, # Banished to bench warming
      # Player == "Michael Pittman" ~ Predicted * 0.85, # Anthony Richardson
      # Player == "Mike Williams" ~ Predicted * 0.00,    # Retiring
      # Player == "Najee Harris" ~ Predicted * 0.7,     # Blew his damn eyes off
      # Player == "Nick Chubb" ~ Predicted * 0.9,       # Broken foot
      # Player == "Pierre Strong" ~ Predicted * 0.25, # Banished to bench warming
      # Player == "Rachaad White" ~ Predicted * 0.8, # Now a backup
      # Player == "Rashee Rice" ~ Predicted * 0.62,     # DUI
      # Player == "Ricky Pearsall" ~ Predicted * 1.35, # Was shot, only WR left on team
      # Player == "Russell Wilson" ~ Predicted * 0.8, # Model needs to chill out with Russ a bit
      # Player == "Sam Darnold" ~ Predicted * 0.85, # Model needs to chill out with Darnold a bit
      # Player == "Stefon Diggs" ~ Predicted * 0.9,     # Torn ACL
      # Player == "Tank Dell" ~ Predicted * 0.10,        # Torn ACL
      # Player == "Tyrone Tracy" ~ Predicted * 1.2, # Now a starter
      # Player == "Xavier Worthy" ~ Predicted * 1.35,    # Teammate has DUI