##############################################################################
### Feature Correlation Diagnostic (read-only)
#
# Flags highly-correlated feature clusters within each position's feature set so you can
# curate redundant columns. XGBoost is robust to multicollinearity for *accuracy*, so this
# is NOT about prediction error - it's about (1) cleaner feature-importance plots (correlated
# features split the credit), (2) modest overfitting reduction on a small dataset, and
# (3) steadier bootstrap prediction intervals across refits.
#
# This script DROPS NOTHING. It reports candidates; you review them and validate any actual
# removals against tests/model_proj_blend_analysis.R / the interval coverage backtest before
# committing to a trimmed feature set.
#
# HOW TO RUN: step through 01_build_nfl_model.R far enough that model_df and the three feature
# vectors (qb_features / rb_features / wr_features) exist in the global environment, then source
# or step through this script.
##############################################################################

source("00_globals.R")

# Pairwise |correlation| at or above this is treated as "highly correlated"
CORR_CUTOFF <- 0.95

# Guard: this script consumes objects produced by 01, fail early with a clear message if absent
required_objs <- c("model_df", "qb_features", "rb_features", "wr_features")
missing_objs <- required_objs[!vapply(required_objs, exists, logical(1))]
if (length(missing_objs) > 0) {
  stop(
    "Missing required objects: ", paste(missing_objs, collapse = ", "),
    ".\nStep through 01_build_nfl_model.R until model_df and the feature vectors exist first."
  )
}

# Mirror train_position_model's position grouping (QB alone; RB+FB; WR+TE) and row filter so
# the correlations reflect exactly the data each model is trained on
filter_position <- function(df, position) {
  df %>%
    filter(
      (position == "WR" & Pos %in% c("WR", "TE")) |
        (position == "QB" & Pos == "QB") |
        (position == "RB" & Pos %in% c("RB", "FB"))
    ) %>%
    filter(!is.na(points_next_year), G > 0)
}

# Build a clean numeric feature matrix for a position. Correlation is only meaningful for
# numeric columns, so Team/Pos (character) and Year (Date) are set aside, as are any
# zero-variance columns (correlation is undefined for them).
build_feature_matrix <- function(df, position, feature_cols) {
  feat <- filter_position(df, position) %>% select(any_of(feature_cols))

  numeric_feat <- feat %>% select(where(is.numeric))
  dropped_nonnumeric <- setdiff(names(feat), names(numeric_feat))

  variances <- vapply(numeric_feat, var, numeric(1), na.rm = TRUE)
  zero_var <- names(variances)[is.na(variances) | variances == 0]
  numeric_feat <- numeric_feat %>% select(-any_of(zero_var))

  list(
    matrix = numeric_feat,
    dropped_nonnumeric = dropped_nonnumeric,
    dropped_zero_var = zero_var
  )
}

# Extract the unique highly-correlated pairs (upper triangle only), strongest first
high_corr_pairs <- function(cor_mat, cutoff) {
  cor_mat[lower.tri(cor_mat, diag = TRUE)] <- NA
  pairs <- as.data.frame(as.table(cor_mat))
  names(pairs) <- c("Feature_1", "Feature_2", "Correlation")
  pairs %>%
    filter(!is.na(Correlation), abs(Correlation) >= cutoff) %>%
    arrange(desc(abs(Correlation))) %>%
    mutate(Correlation = round(Correlation, 3))
}

# Clustered correlation heatmap - features ordered by hierarchical clustering so correlated
# blocks sit together and are easy to spot
plot_corr_heatmap <- function(cor_mat, pos_label) {
  ord <- hclust(as.dist(1 - abs(cor_mat)))$order
  cor_ord <- cor_mat[ord, ord]

  long <- as.data.frame(as.table(cor_ord))
  names(long) <- c("Feature_1", "Feature_2", "Correlation")
  # Preserve the clustered ordering on both axes
  long$Feature_1 <- factor(long$Feature_1, levels = rownames(cor_ord))
  long$Feature_2 <- factor(long$Feature_2, levels = rownames(cor_ord))

  ggplot(long, aes(Feature_1, Feature_2, fill = Correlation)) +
    geom_tile() +
    scale_fill_gradient2(
      low = "#2C7BB6", mid = "white", high = "#D7191C",
      midpoint = 0, limits = c(-1, 1)
    ) +
    labs(
      title = paste(pos_label, "Feature Correlation (clustered)"),
      x = NULL, y = NULL
    ) +
    coord_fixed() +
    theme_minimal(base_size = 8) +
    theme(
      plot.title = element_text(face = "bold", size = 13),
      axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 5),
      axis.text.y = element_text(size = 5),
      panel.grid = element_blank()
    )
}

# Run the full diagnostic for one position: prints the report, returns results + the heatmap
diagnose_position <- function(df, position, feature_cols, pos_label, cutoff = CORR_CUTOFF) {
  built <- build_feature_matrix(df, position, feature_cols)
  cor_mat <- cor(built$matrix, use = "pairwise.complete.obs")

  pairs <- high_corr_pairs(cor_mat, cutoff)
  # findCorrelation returns the columns it would remove to push all pairwise |cor| below cutoff
  drop_candidates <- caret::findCorrelation(cor_mat, cutoff = cutoff, names = TRUE)

  cat("\n==================== ", pos_label, " ====================\n", sep = "")
  cat("Numeric features analyzed: ", ncol(built$matrix), "\n", sep = "")
  if (length(built$dropped_nonnumeric) > 0) {
    cat("Set aside (non-numeric): ", paste(built$dropped_nonnumeric, collapse = ", "), "\n", sep = "")
  }
  if (length(built$dropped_zero_var) > 0) {
    cat("Set aside (zero variance): ", paste(built$dropped_zero_var, collapse = ", "), "\n", sep = "")
  }

  cat("\nHighly-correlated pairs (|r| >= ", cutoff, "): ", nrow(pairs), "\n", sep = "")
  if (nrow(pairs) > 0) print(pairs, row.names = FALSE)

  cat("\nfindCorrelation drop candidates (", length(drop_candidates), "):\n", sep = "")
  if (length(drop_candidates) > 0) {
    cat(paste(" -", drop_candidates), sep = "\n")
    cat("\n")
  } else {
    cat(" (none above cutoff)\n")
  }

  list(
    cor_mat = cor_mat,
    pairs = pairs,
    drop_candidates = drop_candidates,
    heatmap = plot_corr_heatmap(cor_mat, pos_label)
  )
}

# --- Run per position ---
qb_corr <- diagnose_position(model_df, "QB", qb_features, "Quarterbacks")
rb_corr <- diagnose_position(model_df, "RB", rb_features, "Running Backs")
wr_corr <- diagnose_position(model_df, "WR", wr_features, "Receivers (WR + TE)")

# Render the heatmaps (step through these one at a time in an interactive session)
print(qb_corr$heatmap)
print(rb_corr$heatmap)
print(wr_corr$heatmap)

cat(
  "\nHow to use this:\n",
  "- 'Highly-correlated pairs' shows what is redundant with what - eyeball each pair.\n",
  "- 'findCorrelation drop candidates' is a *suggested* minimal set to remove to break the\n",
  "  redundancy; treat it as a starting point, not gospel (both features can carry signal).\n",
  "- Before removing anything from qb/rb/wr_features in 01, confirm the drop does NOT worsen\n",
  "  held-out error via the blend/interval backtests. If error holds, keep the trim for the\n",
  "  cleaner importance plots and steadier intervals.\n"
)
