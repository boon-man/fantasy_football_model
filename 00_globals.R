# Single source of truth for package dependencies across the pipeline.
# Every numbered script sources this file, so add new package requirements here
# rather than calling library() inside individual scripts or functions.
required_packages <- c(
  "rvest", "dplyr", "stringr", "forecast", "tidyr", "tidyverse",
  "zoo", "ggplot2", "lubridate", "data.table", "rBayesianOptimization",
  "caret", "xgboost", "ggrepel", "nflreadr", "ranger"
)

install_if_missing <- function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)  # Install if not found
    library(pkg, character.only = TRUE)         # Load after installing
  } else {
    library(pkg, character.only = TRUE)         # Load if already installed
  }
}

invisible(lapply(required_packages, install_if_missing))

EVAL_YEAR <- 2024                   # Final year in training dataframe
PRED_YEAR <- EVAL_YEAR + 1          # Year to predict
START_YEAR <- 2020                 # Earliest season to pull data. NOTE: Chosen to align with the start of ESPN QBR coverage
# Scoring type for fantasy points calculation, one of the keys in SCORING_MULTS below
SCORING_TYPE <- "HALF"

# Per-reception multiplier keyed by scoring type, the single source of truth for PPR weighting
SCORING_MULTS <- c(STANDARD = 0, HALF = 0.5, PPR = 1.0)

# PPR multiplier for receptions, derived from SCORING_TYPE so the two never drift apart
PPR_MULT <- SCORING_MULTS[[SCORING_TYPE]]
