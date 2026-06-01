required_packages <- c(
  "rvest", "dplyr", "stringr", "forecast", "tidyr", "tidyverse",
  "zoo", "ggplot2", "lubridate", "data.table", "rBayesianOptimization",
  "caret", "xgboost", "ggrepel"
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

EVAL_YEAR <- 2025                   # Final year in training dataframe
PRED_YEAR <- EVAL_YEAR + 1          # Year to predict
START_YEAR <- 2006                 # Earliest year to pull data. NOTE: Passing statistics were updated beginning in 2006, prior seasons are missing some efficiency stats
SCORING_TYPE <- "HALF"  # Scoring type for fantasy points calculation, either "PPR", "HALF", or "STANDARD"
PPR_MULT <- 0.5                  # PPR multiplier for receptions
