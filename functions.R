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

##############################################################################
### === FANTASYPROS PROJECTIONS INTAKE === #
### Consumed by 02_combine_projections.R. load_fp_projections() is the entry point; everything
### above it is machinery. The manual-CSV reader is kept alongside the scraper deliberately: it is
### both the automatic per-position fallback and the way back to hand-exported projections if the
### scrape ever stops working (see SKIP_FP_SCRAPE in 02).

# Minimum rows a scraped position must return to be trusted. Anonymous (or expired-login)
# requests come back fenced to 10 players, so anything at or below this means "not logged in".
FP_MIN_ROWS <- 20

##############################################################################
### FantasyPros projections intake
###
### FantasyPros fences the full projections table behind a logged-in account, serving anonymous
### requests a table truncated to 10 players. Three things are worth knowing before changing any
### of this, each established by testing (see tests/fp_scrape_auth_test.R):
###
###   - There is no CSV endpoint. The site's own "download" link is generated client-side by
###     window.exportTableToCSV, which serializes the rendered DOM table - so the manual exports
###     in data/fp_raw/ and this scrape read the very same table, which is why they agree.
###   - The fence is server-side row truncation, not user-agent filtering. Sending a browser
###     user-agent (as an older scraper here did) changes nothing; only being logged in does.
###   - Logging in from R is impossible: the form mints a reCAPTCHA v3 token via
###     grecaptcha.execute() before posting, and a POST without one is rejected as bad
###     credentials even when the credentials are correct.
###
### Hence headless Chrome against a dedicated, persistent profile: log in by hand once (a human
### clears reCAPTCHA v3 without friction), and Chrome keeps the cookie jar in the profile
### directory so later runs are already authenticated. No password or cookie is stored by this
### repo. When the scrape fails or comes back fenced, each position falls back to its manual CSV.
###
### FIRST-TIME (and whenever the login lapses) SETUP - run in a terminal, log in, close the window:
###   open -na "Google Chrome" --args --user-data-dir=~/.fp_chrome_profile https://secure.fantasypros.com/accounts/login/
FP_CHROME_PROFILE <- path.expand("~/.fp_chrome_profile")

fp_projection_url <- function(position, scoring = SCORING_TYPE) {
  sprintf("https://www.fantasypros.com/nfl/projections/%s.php?week=draft&scoring=%s",
          tolower(position), toupper(scoring))
}

# Chrome permits one process per profile directory, enforced with a SingletonLock symlink naming
# "<host>-<pid>". Launching against a locked profile surfaces as an opaque chromote "Cannot find an
# available port", so this reports the real cause - and clears the lock when it is merely stale,
# which a crashed or killed browser leaves behind and which would otherwise block every later run.
assert_chrome_profile_ready <- function(profile_dir = FP_CHROME_PROFILE) {
  # Checking existence first and refusing to continue without it: handed a --user-data-dir it
  # cannot use, Chrome quietly falls back to the DEFAULT profile - so a typo here would otherwise
  # drive the user's everyday browser session instead of this dedicated one
  if (!dir.exists(profile_dir)) {
    stop(sprintf(paste0("No Chrome profile at %s. Run the one-time login first:\n",
                        '  open -na "Google Chrome" --args --user-data-dir=%s %s\n',
                        "Log in, then close that window."),
                 profile_dir, shQuote(profile_dir),
                 "https://secure.fantasypros.com/accounts/login/"), call. = FALSE)
  }

  # Sys.readlink gives NA for a missing path and "" for a non-symlink; neither is a live lock
  lock_target <- Sys.readlink(file.path(profile_dir, "SingletonLock"))
  if (is.na(lock_target) || !nzchar(lock_target)) return(invisible(TRUE))

  locking_pid <- str_extract(lock_target, "[0-9]+$")
  if (is.na(locking_pid)) return(invisible(TRUE))

  # ps exits 0 only when the pid exists
  if (system2("ps", c("-p", locking_pid), stdout = FALSE, stderr = FALSE) == 0) {
    stop(sprintf(paste0("Chrome (pid %s) is already using %s.\n",
                        "Close that window - most likely the manual login window - then retry."),
                 locking_pid, profile_dir), call. = FALSE)
  }

  message("Clearing a stale Chrome profile lock left by pid ", locking_pid, ".")
  unlink(file.path(profile_dir, c("SingletonLock", "SingletonCookie", "SingletonSocket")))
  invisible(TRUE)
}

# Chrome reports a page loaded before the table is necessarily populated, so waiting on a row
# count rather than the load event is what makes this reliable
wait_for_projection_rows <- function(session, min_rows = FP_MIN_ROWS, timeout_sec = 20) {
  deadline <- Sys.time() + timeout_sec
  repeat {
    n_rows <- session$Runtime$evaluate(
      "document.querySelectorAll('#data tbody tr').length"
    )$result$value

    if (n_rows >= min_rows || Sys.time() > deadline) return(n_rows)
    Sys.sleep(0.5)
  }
}

# Parsing table#data into the three-column contract the blend below expects: Player (chr),
# Pos (upper), Projected_Points (chr - the caller strips formatting and coerces).
parse_fp_projections_html <- function(html_text, position) {
  # Selecting by id: the page also carries table#experts (the per-expert breakdown)
  projections_table <- html_element(read_html(html_text), "table#data")
  if (inherits(projections_table, "xml_missing")) {
    stop(sprintf("No table#data on the %s page - the page layout changed.", toupper(position)),
         call. = FALSE)
  }

  # The thead holds two rows: a stat-group banner of <td> cells (PASSING / RUSHING / MISC) and the
  # real header row of <th> cells. Selecting th skips the banner - which is the DOM origin of the
  # spacer row that shows up in the manual CSV exports.
  col_names <-
    projections_table %>%
    html_elements("thead tr th") %>%
    html_text2() %>%
    str_trim() %>%
    str_to_upper()

  if (!"FPTS" %in% col_names) {
    stop(sprintf("No FPTS column in the %s header row: %s", toupper(position),
                 paste(col_names, collapse = ", ")), call. = FALSE)
  }
  # Located by position, taking the last match: the header repeats YDS/TDS/ATT across the rushing
  # and receiving stat groups, so the names are ambiguous
  fpts_col <- last(which(col_names == "FPTS"))

  rows <- html_elements(projections_table, "tbody tr")

  # The player cell is anchor text followed by a bare team abbrev ("<a>Josh Allen</a> BUF"), so the
  # name is read from the anchor's fp-player-name attribute rather than regexing the abbrev off.
  # Falling back to the cell text in case a row ever ships without the anchor.
  player_links <- html_element(rows, "a.player-name")
  tibble(
    Player = coalesce(html_attr(player_links, "fp-player-name"), html_text2(player_links)),
    Projected_Points = map_chr(rows, nth_cell_text, index = fpts_col),
    Pos = toupper(position)
  ) %>%
    mutate(Player = str_squish(Player)) %>%
    select(Player, Pos, Projected_Points)
}

# Pulling one cell out of a row by column position
nth_cell_text <- function(row, index) {
  html_text2(html_elements(row, "td")[[index]])
}

# Scraping every position in ONE browser session - launching Chrome is the expensive part, so it
# is paid once rather than per position. Returns a named list keyed by position so the caller can
# fall back per position rather than all-or-nothing.
scrape_fp_projections <- function(positions, scoring = SCORING_TYPE, min_rows = FP_MIN_ROWS) {
  assert_chrome_profile_ready()

  chrome <- chromote::Chrome$new(
    args = c(chromote::get_chrome_args(), paste0("--user-data-dir=", FP_CHROME_PROFILE))
  )
  browser_session <- chromote::Chromote$new(browser = chrome)$new_session()
  on.exit(browser_session$parent$close(), add = TRUE)

  scrape_one <- function(position) {
    browser_session$Page$navigate(fp_projection_url(position, scoring))
    browser_session$Page$loadEventFired()
    n_rows <- wait_for_projection_rows(browser_session, min_rows = min_rows)

    # Below the threshold means the login has lapsed - returning NULL so the caller falls back
    if (n_rows < min_rows) {
      warning(sprintf("FantasyPros %s returned only %d rows - the Chrome login has likely lapsed.",
                      toupper(position), n_rows), call. = FALSE)
      return(NULL)
    }

    message(sprintf("Scraped %s: %d rows", toupper(position), n_rows))
    parse_fp_projections_html(
      browser_session$Runtime$evaluate("document.documentElement.outerHTML")$result$value,
      position
    )
  }

  set_names(lapply(positions, scrape_one), positions)
}

# Reading the manually exported FantasyPros projections for each position - the fallback path,
# used when SKIP_FP_SCRAPE is TRUE or a position's scrape came back fenced/failed.
# To refresh:
#   1. Log in to fantasypros.com, open the projections page for each position, and set the
#      scoring dropdown to match SCORING_TYPE
#   2. Click the CSV/Excel download link on each page
#   3. Drop the downloads - keeping their FantasyPros filenames, e.g.
#      FantasyPros_Fantasy_Football_Projections_RB.csv - into data/fp_raw/<SCORING_TYPE>/, i.e.
#      data/fp_raw/PPR/, data/fp_raw/HALF/ or data/fp_raw/STANDARD/
# QB projections are scoring-agnostic (no receptions - FPTS is identical under HALF/PPR/none),
# so the same QB export can simply be copied into each scoring folder.
read_fp_projections_csv <- function(position, scoring = "HALF") {
  # Exports live in one subfolder per SCORING_TYPE
  dir <- file.path("data/fp_raw", toupper(scoring))

  # Match FantasyPros' download name for this position. Globbing rather than requiring an exact
  # filename so a re-download that lands as "..._RB (1).csv" is still picked up - newest wins.
  pattern <- paste0("^FantasyPros_Fantasy_Football_Projections_", toupper(position), "\\b.*\\.csv$")
  matches <- list.files(dir, pattern = pattern, full.names = TRUE)
  if (length(matches) == 0) {
    stop(sprintf("No FantasyPros %s export found in %s/ (expected FantasyPros_Fantasy_Football_Projections_%s.csv) - see the export steps above read_fp_projections_csv().",
                 toupper(position), dir, toupper(position)), call. = FALSE)
  }
  path <- matches[which.max(file.mtime(matches))]

  # Read as text with no header. The export ships duplicate column names (YDS/TDS/ATT appear once
  # for rushing and again for receiving), which read_csv would otherwise mangle, and row 2 is a
  # spacer row - a non-breaking space plus empty fields - left over from the stat-group header
  # row of the HTML table. Column count varies by position, so FPTS is located by name.
  # The spacer row and the file's trailing blank line are both short, which read_csv flags; every
  # column is read as text, so a field-count mismatch is the only problem it can raise and both
  # offenders are dropped below - hence suppressing rather than surfacing the warning.
  raw <- suppressWarnings(
    read_csv(path, col_names = FALSE, col_types = cols(.default = "c"), progress = FALSE)
  )

  col_names <- str_to_upper(str_trim(unlist(raw[1, ])))
  if (!"FPTS" %in% col_names) {
    stop(sprintf("No FPTS column in the header row of %s - is this a FantasyPros projections export?", path),
         call. = FALSE)
  }

  # Drop the header row, then the spacer row falls out on its empty points cell. FantasyPros
  # ships Team in its own column, so the player cell needs no team-abbrev stripping.
  raw[-1, ] %>%
    select(Player = 1, Projected_Points = all_of(last(which(col_names == "FPTS")))) %>%
    filter(!is.na(Projected_Points)) %>%
    mutate(
      Player = str_squish(Player),
      Pos = toupper(position)
    ) %>%
    select(Player, Pos, Projected_Points)
}

# Loading every position's projections, scraping by default and falling back to the manual CSV
# export per position - so one lapsed position (or one page that failed to render) does not cost
# the whole run. Returns the stacked three-column frame the blend consumes.
#
# `use_scrape` is passed in rather than read from a global: the caller (02) owns the SKIP_FP_SCRAPE
# toggle, and this file should not reach into a script that sources it.
load_fp_projections <- function(positions, scoring = SCORING_TYPE, use_scrape = TRUE) {
  scraped <- if (!use_scrape) {
    set_names(vector("list", length(positions)), positions)
  } else {
    # A hard failure here (no Chrome, no profile) should not be fatal - every position simply
    # falls back to its export
    tryCatch(
      scrape_fp_projections(positions, scoring),
      error = function(e) {
        warning("FantasyPros scrape failed (", conditionMessage(e),
                ") - falling back to the manual CSV exports.", call. = FALSE)
        set_names(vector("list", length(positions)), positions)
      }
    )
  }

  load_one <- function(position) {
    if (!is.null(scraped[[position]])) return(scraped[[position]])
    message("Reading the manual ", toupper(position), " export from data/fp_raw/.")
    read_fp_projections_csv(position, scoring)
  }

  bind_rows(lapply(positions, load_one))
}
