##############################################################################
### Model evaluation visualizations
### Diagnostic plots for assessing the predictive performance of the position
### models, ported from the plotnine workflow in the baseball modeling project
### Sourced by 01_build_nfl_model.R, expects pred_df with columns:
###   Player, Year, Actual, Predicted

# === THEME === #

# Color palette for NFL visualizations
NFL_COLOR_PALETTE <- c("#637ae4", "#37d784", "#d7326f", "#51e9eb", "#103737")

# Custom ggplot theme for NFL visualizations
# Clean classic feel with bolder titles and faint gridlines for readability
theme_nfl <- function() {
  theme_classic(base_family = "serif") +
    theme(
      plot.title = element_text(size = 18, face = "bold"),
      plot.subtitle = element_text(size = 12, color = "#555555"),
      axis.title = element_text(size = 14),
      axis.text = element_text(size = 12),
      legend.title = element_text(size = 11),
      legend.text = element_text(size = 10),
      panel.background = element_rect(fill = "white", color = NA),
      panel.grid.major = element_line(color = "#e6e6e6", linewidth = 0.35),
      panel.grid.minor = element_line(color = "#f2f2f2", linewidth = 0.25)
    )
}

# === HELPERS === #

# Helper to add a signed residual column, defined as predicted minus actual
add_prediction_diff <- function(pred_df) {
  pred_df |>
    mutate(prediction_diff = Predicted - Actual)
}

# Helper to pull the largest absolute residuals and build their callout labels
# The year shown is the season being predicted rather than the feature season
build_outlier_labels <- function(pred_df, top_n) {
  pred_df |>
    slice_max(abs(prediction_diff), n = top_n, with_ties = FALSE) |>
    mutate(label = paste0(Player, " (", lubridate::year(Year) + 1, ")"))
}

# === DIAGNOSTIC PLOTS === #

#' Scatter plot of actual vs predicted fantasy points
#'
#' Includes a linear fit with confidence band, a 45 degree perfect prediction
#' reference line, labeled outliers by absolute residual, and region
#' annotations marking over and underperformers relative to the model.
#'
#' Parameters
#' ----------
#' pred_df : data.frame with Player, Year, Actual, Predicted
#' pos_label : character, position group name used in the plot title
#' top_n : integer, number of largest absolute residuals to label (default 15)
#' overperf_x, overperf_y : numeric, raw point coordinates for the "Overperformers"
#'   label. Exposed because the ideal spot drifts a lot between position groups. When
#'   left NULL they fall back to a data-relative default (0.80 * max predicted,
#'   0.95 * max actual).
#' underperf_x, underperf_y : numeric, raw point coordinates for the "Underperformers"
#'   label; NULL falls back to (0.95 * max predicted, 0.15 * max actual).
#'
#' Returns a ggplot object. Points above the dashed line are players the model
#' underpredicted (overperformers); points below were overpredicted.
plot_actual_vs_pred <- function(pred_df, pos_label = "", top_n = 15,
                                overperf_x = NULL, overperf_y = NULL,
                                underperf_x = NULL, underperf_y = NULL) {
  df <- add_prediction_diff(pred_df)
  outliers <- build_outlier_labels(df, top_n)

  # Dynamic placement for the region annotations
  max_x <- max(df$Predicted)
  max_y <- max(df$Actual)

  # Use raw point coordinates when supplied, otherwise fall back to data-relative
  # defaults so each position still lands a reasonable label spot
  overperf_x  <- overperf_x  %||% (0.80 * max_x)
  overperf_y  <- overperf_y  %||% (0.95 * max_y)
  underperf_x <- underperf_x %||% (0.95 * max_x)
  underperf_y <- underperf_y %||% (0.15 * max_y)

  ggplot(df, aes(x = Predicted, y = Actual)) +
    geom_point(alpha = 0.65, size = 1.4, color = NFL_COLOR_PALETTE[1]) +
    geom_smooth(
      method = "lm", se = TRUE, level = 0.99,
      color = "#4A79B8", fill = "#AFC7E8", alpha = 0.18, linewidth = 0.4
    ) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey", alpha = 0.6) +
    geom_text_repel(
      data = outliers, aes(label = label),
      size = 2.6, fontface = "italic", color = "#333333",
      segment.color = "darkgrey", segment.size = 0.3,
      min.segment.length = 0, max.overlaps = Inf
    ) +
    annotate(
      "text", x = overperf_x, y = overperf_y,
      label = "Overperformers", fontface = "bold", size = 4.5, alpha = 0.7
    ) +
    annotate(
      "text", x = underperf_x, y = underperf_y,
      label = "Underperformers", fontface = "bold", size = 4.5, alpha = 0.7
    ) +
    labs(
      title = paste(pos_label, "Actual vs Predicted Fantasy Points"),
      x = "Predicted Fantasy Points",
      y = "Actual Fantasy Points (Next Season)"
    ) +
    theme_nfl()
}

#' Residuals vs predicted values with labeled outliers
#'
#' Useful for checking heteroskedasticity, spotting systematic bias as
#' residuals drift above or below zero, and locating extreme misses in
#' prediction space. A shaded band marks the typical error tolerance.
#'
#' Parameters
#' ----------
#' pred_df : data.frame with Player, Year, Actual, Predicted
#' pos_label : character, position group name used in the plot title
#' top_n : integer, number of largest absolute residuals to label (default 20)
#' band : numeric, symmetric residual band shaded in the background (default 50 points)
plot_resid_vs_pred <- function(pred_df, pos_label = "", top_n = 20, band = 50) {
  df <- add_prediction_diff(pred_df)
  outliers <- build_outlier_labels(df, top_n)

  ggplot(df, aes(x = Predicted, y = prediction_diff)) +
    annotate("rect", xmin = -Inf, xmax = Inf, ymin = -band, ymax = band, alpha = 0.18, fill = "lightgrey") +
    geom_point(alpha = 0.65, size = 1.4, color = NFL_COLOR_PALETTE[2]) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey", alpha = 0.7) +
    geom_text_repel(
      data = outliers, aes(label = label),
      size = 2.6, fontface = "italic", color = "#333333",
      segment.color = "darkgrey", segment.size = 0.3,
      min.segment.length = 0, max.overlaps = Inf
    ) +
    labs(
      title = paste(pos_label, "Residuals vs Predicted Fantasy Points"),
      x = "Predicted Fantasy Points",
      y = "Prediction Diff (Predicted - Actual)"
    ) +
    theme_nfl()
}

#' Histogram of residuals with an annotated error tolerance band
#'
#' Shows the shape and spread of model errors, the share of predictions
#' landing within a practical tolerance band, and any skew indicating
#' consistent over or underprediction.
#'
#' Parameters
#' ----------
#' pred_df : data.frame with Actual and Predicted columns
#' pos_label : character, position group name used in the plot title
#' band : numeric, symmetric residual band shaded in the background (default 50 points)
#' binwidth : numeric, histogram bin width (default 10 points)
plot_resid_hist <- function(pred_df, pos_label = "", band = 50, binwidth = 10) {
  df <- add_prediction_diff(pred_df)

  # Share of predictions landing inside the tolerance band
  within_pct <- round(mean(abs(df$prediction_diff) <= band) * 100)

  ggplot(df, aes(x = prediction_diff)) +
    annotate("rect", xmin = -band, xmax = band, ymin = -Inf, ymax = Inf, alpha = 0.25, fill = "lightgrey") +
    geom_histogram(binwidth = binwidth, fill = NFL_COLOR_PALETTE[4], color = "white", alpha = 0.9) +
    labs(
      title = paste(pos_label, "Distribution of Prediction Errors"),
      subtitle = paste0(within_pct, "% of predictions within +/- ", band),
      x = "Prediction Diff (Predicted - Actual)",
      y = NULL
    ) +
    theme_nfl()
}

#' Decile based calibration curve
#'
#' Bins predictions into deciles and plots mean predicted against mean actual
#' fantasy points per decile, with percentage labels showing how far actuals
#' ran above or below the model in each bin. A well calibrated model tracks
#' the dashed identity line.
#'
#' Parameters
#' ----------
#' pred_df : data.frame with Actual and Predicted columns
#' pos_label : character, position group name used in the plot title
#' n_deciles : integer, number of quantile bins (default 10)
#' nudge_y : numeric, vertical offset for the percentage labels (default 5 points)
#' decile_cutoff: integer, minimum cutoff for decile plottting
plot_decile_calib <- function(pred_df, pos_label = "", n_deciles = 10, nudge_y = 5, decile_cutoff = 4) {

  # Binning into prediction deciles and averaging within each bin
  # Positive pct labels mean actuals ran above the model (underprediction)
  decile_calib <-
    pred_df |>
    mutate(pred_decile = ntile(Predicted, n_deciles)) |>
    group_by(pred_decile) |>
    summarise(
      mean_pred = mean(Predicted),
      mean_actual = mean(Actual),
      .groups = "drop"
    ) |>
    mutate(pct_diff = (mean_actual - mean_pred) / mean_actual * 100) |>
    mutate(diff_label = paste0(round(pct_diff, 1), "%")) %>%
    filter(pred_decile >= decile_cutoff)

  ggplot(decile_calib, aes(x = mean_pred, y = mean_actual)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "lightgrey", alpha = 0.8) +
    geom_line(color = NFL_COLOR_PALETTE[3], alpha = 0.9, linewidth = 0.75) +
    geom_point(color = NFL_COLOR_PALETTE[3], alpha = 0.9, size = 2) +
    geom_text(
      aes(label = diff_label),
      nudge_y = nudge_y, color = "#555555", fontface = "bold", size = 3
    ) +
    labs(
      title = paste(pos_label, "Calibration by Predicted Decile"),
      x = "Mean Predicted Fantasy Points",
      y = "Mean Actual Fantasy Points"
    ) +
    theme_nfl()
}

