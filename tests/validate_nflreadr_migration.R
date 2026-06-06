##############################################################################
### Validation for the nflreadr migration
### Compares the new nflverse sourced dataset against the legacy PFR csvs
### Run interactively from the project root after the first full data refresh
library(tidyverse)
library(data.table)

source("00_globals.R")
source("functions.R")

# Loading both generations of data
new_stats <- fread("data/player_stats_final.csv")
legacy_receiving <- fread("data/receiving_final.csv")
legacy_rushing <- fread("data/rushing_final.csv")
legacy_passing <- fread("data/passing_final.csv")

# === SEASON COVERAGE === #
# Row counts by season, coverage should be stable across the full year range
new_stats |>
  count(Year) |>
  print(n = Inf)

# === PLAYER LEVEL COMPARISONS === #
# Joining legacy and new data on cleaned player names within each season
# Differences should be near zero for high volume players

# Helper to prepare a legacy stat table for comparison joins
prep_legacy <- function(df, stat_col) {
  df |>
    mutate(Player_clean = clean_player_name(Player)) |>
    select(Player_clean, Year, legacy_value = all_of(stat_col))
}

# Helper to compare a single stat between the legacy and new datasets
compare_stat <- function(legacy_df, stat_col, volume_floor) {
  legacy_prepped <- prep_legacy(legacy_df, stat_col)

  new_prepped <-
    new_stats |>
    mutate(Player_clean = clean_player_name(Player)) |>
    select(Player_clean, Year, new_value = all_of(stat_col))

  comparison <-
    legacy_prepped |>
    inner_join(new_prepped, by = c("Player_clean", "Year")) |>
    filter(legacy_value >= volume_floor) |>
    mutate(abs_diff = abs(legacy_value - new_value))

  # Summarizing overall agreement
  comparison |>
    summarise(
      players_compared = n(),
      exact_matches = sum(abs_diff == 0),
      mean_abs_diff = mean(abs_diff),
      max_abs_diff = max(abs_diff)
    ) |>
    print()

  # Surfacing the largest mismatches for manual review
  comparison |>
    arrange(desc(abs_diff)) |>
    head(10) |>
    print()
}

# Receiving yards agreement for regular contributors
cat("=== RECEIVING YARDS ===\n")
compare_stat(legacy_receiving, "receiving_yds", volume_floor = 500)

# Rushing yards agreement for regular contributors
cat("=== RUSHING YARDS ===\n")
compare_stat(legacy_rushing, "rush_yds", volume_floor = 500)

# Passing yards agreement for regular contributors
cat("=== PASSING YARDS ===\n")
compare_stat(legacy_passing, "passing_yards", volume_floor = 2000)

# === NEW COLUMN SANITY === #
# The substituted columns should be populated and within sensible ranges
new_stats |>
  filter(passing_att >= 300) |>
  summarise(
    qbr_coverage = mean(QBR > 0),
    rate_range = paste(round(min(Rate)), "to", round(max(Rate))),
    epa_rate_range = paste(round(min(passing_epa_per_att), 2), "to", round(max(passing_epa_per_att), 2))
  ) |>
  print()

new_stats |>
  filter(Tgt >= 100) |>
  summarise(
    air_yards_populated = mean(receiving_air_yards > 0),
    epa_per_target_range = paste(round(min(receiving_epa_per_target), 2), "to", round(max(receiving_epa_per_target), 2))
  ) |>
  print()

# Draft capital should be present for most drafted players and capped for undrafted ones
new_stats |>
  count(draft_number == 300) |>
  print()
