##############################################################################
### Shared functions for the fantasy football pipeline

# === LEGACY PRO FOOTBALL REFERENCE FUNCTIONS === #
# PFR no longer permits scraping, these remain only for the archived analysis
# scripts in the tests directory and should not be used for new data pulls
scrapeData = function(urlprefix, urlend, startyr, endyr, stat) {
  master <- data.frame()

  for (i in startyr:endyr) {
    Sys.sleep(5)
    cat('Loading Year', i, '\n')
    URL <- paste(urlprefix, i, urlend, sep = "")

    # Retry logic - attempt up to 3 times
    attempt <- 0
    success <- FALSE

    while (attempt < 3 && !success) {
      attempt <- attempt + 1

      tryCatch({
        table <-
          read_html(
            URL,
            user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
          ) %>%
          html_nodes("table") %>%
          .[[1]] %>%
          html_table()

        table$Year <- i
        master <- rbind(table, master)
        success <- TRUE
        cat('  ✓ Year', i, 'loaded successfully\n')

      }, error = function(e) {
        cat('  Attempt', attempt, 'failed for Year', i, '\n')
        if (attempt < 3) {
          Sys.sleep(10)  # Wait longer before retry
        }
      })
    }

    if (!success) {
      cat('  ✗ Failed to load Year', i, 'after 3 attempts\n')
    }
  }

  assign(quo_name(enquo(stat)), master, envir=.GlobalEnv)
  return('Complete')
}

# Function to filter out split season stat rows, when players were traded or released to join a new team mid-season.
# Only needed for legacy PFR data, the nflverse season summaries arrive pre-aggregated
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

# === SHARED CLEANING FUNCTIONS === #

# Function for cleaning up player name columns for joining projection data together
clean_player_name <- function(name) {
  name %>%
    tolower() %>%
    str_remove_all("\\b(jr|sr|ii|iii|iv|v)\\b\\.?") %>%  # remove suffixes
    str_replace_all("['’\\.\\-]", "") %>%                # remove apostrophes, periods, hyphens
    str_replace_all("[^a-z ]", " ") %>%                  # remove any remaining non-letter chars
    str_squish() %>%
    str_to_title()
}

# Helper to fill missing values with zero, used in place of inline lambdas
fill_missing_with_zero <- function(x) {
  replace_na(x, 0)
}

# === NFLVERSE DATA INTAKE === #
# All player data is sourced from the nflverse ecosystem via nflreadr
# The GSIS player_id is the primary key for player joins going forward

# Function to pull player metadata from nflverse rosters
# Provides birth dates for age calculation, draft capital, and the ESPN id crosswalk
# Roster records often lack the ESPN id in a player's rookie season, so ids are
# backfilled across seasons and supplemented with the nflverse id crosswalk
load_player_metadata <- function(start_year, end_year) {

  # Static id crosswalk used to fill ESPN ids the roster data is missing
  espn_crosswalk <-
    load_ff_playerids() %>%
    filter(!is.na(gsis_id), !is.na(espn_id)) %>%
    transmute(gsis_id, espn_id_crosswalk = as.character(espn_id)) %>%
    distinct(gsis_id, .keep_all = TRUE)

  load_rosters(seasons = start_year:end_year) %>%
    filter(!is.na(gsis_id)) %>%
    # Older roster files store some columns as character, coercing for type stability across seasons
    mutate(
      espn_id = as.character(espn_id),
      birth_date = as.Date(birth_date),
      draft_number = as.numeric(draft_number),
      entry_year = as.numeric(entry_year)
    ) %>%
    select(season, gsis_id, espn_id, birth_date, draft_number, entry_year) %>%
    distinct(season, gsis_id, .keep_all = TRUE) %>%
    # Carrying a known ESPN id to player seasons where the roster record is missing it
    group_by(gsis_id) %>%
    mutate(espn_id = coalesce(espn_id, espn_id[!is.na(espn_id)][1])) %>%
    ungroup() %>%
    # Falling back to the static crosswalk when no roster season carries the id
    left_join(espn_crosswalk, by = "gsis_id") %>%
    mutate(espn_id = coalesce(espn_id, espn_id_crosswalk)) %>%
    select(-espn_id_crosswalk)
}

# Function to compute regular season team win totals from nflverse schedules
# Serves as the replacement for the quarterback win column from the legacy PFR data
compute_team_wins <- function(start_year, end_year) {

  # Limiting the schedule to completed regular season games
  schedules <-
    load_schedules(seasons = start_year:end_year) %>%
    filter(game_type == "REG") %>%
    filter(!is.na(home_score), !is.na(away_score))

  # Tallying results from the home team perspective
  home_results <-
    schedules %>%
    transmute(season, team = home_team, won = home_score > away_score)

  # Tallying results from the away team perspective
  away_results <-
    schedules %>%
    transmute(season, team = away_team, won = away_score > home_score)

  # Stacking both perspectives and summing wins for each team season
  bind_rows(home_results, away_results) %>%
    group_by(season, team) %>%
    summarise(wins = sum(won), .groups = "drop")
}

# Function to pull season level ESPN QBR ratings, joined to players via the ESPN id
# NOTE: ESPN only publishes season QBR for qualified quarterbacks, so backups and
# part time starters will be missing here and receive zero downstream
load_qbr_ratings <- function(start_year, end_year) {
  load_espn_qbr(seasons = start_year:end_year, summary_type = "season") %>%
    filter(season_type == "Regular") %>%
    mutate(espn_id = as.character(player_id)) %>%
    select(season, espn_id, QBR = qbr_total)
}

# Function to compute the traditional passer rating from its component stats
compute_passer_rating <- function(completions, attempts, yards, tds, ints) {
  comp_component <- pmax(pmin(((completions / attempts) - 0.3) * 5, 2.375), 0)
  yards_component <- pmax(pmin(((yards / attempts) - 3) * 0.25, 2.375), 0)
  td_component <- pmax(pmin((tds / attempts) * 20, 2.375), 0)
  int_component <- pmax(pmin(2.375 - ((ints / attempts) * 25), 2.375), 0)

  ((comp_component + yards_component + td_component + int_component) / 6) * 100
}

# Function to assemble the full player season dataset from nflverse sources
#
# Legacy PFR columns with no direct nflreadr equivalent are handled as follows:
#   - Success rate columns are replaced by EPA per play rates from load_player_stats
#   - Longest play columns are replaced by air yards and yards after catch for receivers, otherwise dropped
#   - Games started and its derived features are dropped, games played remains as the involvement signal
#   - Quarterback record wins are approximated with team regular season wins from load_schedules
#   - Fourth quarter comebacks and game winning drives are dropped with no substitute
#   - Award voting columns are dropped, draft capital serves as the talent pedigree signal
#     TODO: explore pulling award voting data from an alternative source
build_player_season_stats <- function(start_year, end_year) {

  # Pulling season level offensive stats, the foundation of the modeling dataset
  season_stats <- load_player_stats(seasons = start_year:end_year, summary_level = "reg")

  # Pulling supporting tables for player metadata, quarterback ratings, and team records
  player_metadata <- load_player_metadata(start_year, end_year)
  qbr_ratings <- load_qbr_ratings(start_year, end_year)
  team_wins <- compute_team_wins(start_year, end_year)

  season_stats %>%
    # Limiting the pool to fantasy relevant offensive positions
    filter(position %in% c("QB", "RB", "FB", "WR", "TE")) %>%
    # Attaching player metadata, quarterback ratings, and team win totals
    left_join(player_metadata, by = c("player_id" = "gsis_id", "season" = "season")) %>%
    left_join(qbr_ratings, by = c("espn_id" = "espn_id", "season" = "season")) %>%
    left_join(team_wins, by = c("recent_team" = "team", "season" = "season")) %>%
    # Computing age at the end of the calendar year to match the legacy PFR convention
    mutate(Age = floor(as.numeric(make_date(season, 12, 31) - birth_date) / 365.25)) %>%
    # Building the rate and efficiency stats the legacy pipeline sourced directly from PFR
    mutate(
      dropbacks = attempts + sacks_suffered,
      catch_percent = if_else(targets > 0, (receptions / targets) * 100, 0),
      receiving_yds_rec = if_else(receptions > 0, receiving_yards / receptions, 0),
      receiving_rec_g = if_else(games > 0, receptions / games, 0),
      receiving_y_g = if_else(games > 0, receiving_yards / games, 0),
      receiving_yards_target = if_else(targets > 0, receiving_yards / targets, 0),
      receiving_epa_per_target = if_else(targets > 0, receiving_epa / targets, 0),
      rush_yds_att = if_else(carries > 0, rushing_yards / carries, 0),
      rush_yds_game = if_else(games > 0, rushing_yards / games, 0),
      rush_attempts_per_game = if_else(games > 0, carries / games, 0),
      rush_epa_per_att = if_else(carries > 0, rushing_epa / carries, 0),
      # Total fumbles across play types, matching the all fumbles definition from legacy PFR data
      total_fumbles = rushing_fumbles + receiving_fumbles + sack_fumbles,
      passing_comp_pct = if_else(attempts > 0, (completions / attempts) * 100, 0),
      passing_td_pct = if_else(attempts > 0, (passing_tds / attempts) * 100, 0),
      passing_int_pct = if_else(attempts > 0, (passing_interceptions / attempts) * 100, 0),
      passing_yards_att = if_else(attempts > 0, passing_yards / attempts, 0),
      passing_avg_yards_att = if_else(attempts > 0, (passing_yards + (20 * passing_tds) - (45 * passing_interceptions)) / attempts, 0),
      passing_yards_comp = if_else(completions > 0, passing_yards / completions, 0),
      passing_yards_game = if_else(games > 0, passing_yards / games, 0),
      passing_epa_per_att = if_else(dropbacks > 0, passing_epa / dropbacks, 0),
      sack_percent = if_else(dropbacks > 0, (sacks_suffered / dropbacks) * 100, 0),
      passing_net_yards_att = if_else(dropbacks > 0, (passing_yards - sack_yards_lost) / dropbacks, 0),
      passing_adj_net_yards_att = if_else(dropbacks > 0, (passing_yards - sack_yards_lost + (20 * passing_tds) - (45 * passing_interceptions)) / dropbacks, 0),
      # Average depth of target: how far downfield the QB throws, isolating the air-yards
      # component the total-yards metrics above conflate with yards after the catch
      passing_adot = if_else(attempts > 0, passing_air_yards / attempts, 0),
      Rate = if_else(attempts > 0, compute_passer_rating(completions, attempts, passing_yards, passing_tds, passing_interceptions), 0)
    ) %>%
    # Renaming to the column names the downstream feature engineering and models expect
    transmute(
      player_id,
      Player = player_display_name,
      Team = recent_team,
      Pos = position,
      Year = season,
      Age,
      G = games,
      draft_number,
      entry_year,
      Tgt = targets,
      Rec = receptions,
      catch_percent,
      receiving_yds = receiving_yards,
      receiving_yds_rec,
      receiving_td = receiving_tds,
      receiving_1D = receiving_first_downs,
      receiving_epa,
      receiving_epa_per_target,
      receiving_rec_g,
      receiving_y_g,
      receiving_yards_target,
      receiving_air_yards,
      receiving_yards_after_catch,
      # Opportunity-share metrics: how much of the team's passing-game pie a player commands.
      # Stickier year over year than raw volume and inherently team-context adjusted.
      target_share,
      air_yards_share,
      wopr,
      racr,
      rush_att = carries,
      rush_yds = rushing_yards,
      rush_td = rushing_tds,
      rush_1D = rushing_first_downs,
      rushing_epa,
      rush_epa_per_att,
      rush_yds_att,
      rush_yds_game,
      rush_attempts_per_game,
      rush_fbl = total_fumbles,
      passing_comp = completions,
      passing_att = attempts,
      passing_comp_pct,
      passing_yards,
      passing_td = passing_tds,
      passing_td_pct,
      passing_int = passing_interceptions,
      passing_int_pct,
      passing_1D = passing_first_downs,
      passing_epa,
      passing_epa_per_att,
      passing_yards_att,
      passing_avg_yards_att,
      passing_yards_comp,
      passing_yards_game,
      Rate,
      QBR,
      passing_sack = sacks_suffered,
      sack_yds = sack_yards_lost,
      sack_percent,
      passing_net_yards_att,
      passing_adj_net_yards_att,
      passing_adot,
      # QB skill/efficiency signals: completion % over expected and passer air conversion ratio
      passing_cpoe,
      pacr,
      wins
    ) %>%
    # Undrafted players receive a draft position value beyond the final pick of the draft
    mutate(draft_number = replace_na(draft_number, 300)) %>%
    # Filling missing stat values with zero to match the legacy dataset convention
    # Entry year stays missing where unknown so downstream career logic can fall back gracefully
    mutate(across(where(is.numeric) & !any_of("entry_year"), fill_missing_with_zero))
}
