##############################################################################
### Overall Player Tiering
# First, an overall relative value tier will be created for each player regardless of position
# Next, we will create relative value tiers for each player respective to their position

# Reading in global variables if not already set
# 00_globals.R loads all package dependencies for the pipeline
source("00_globals.R")

## SELECT EITHER UNDERDOG FANTASY OR ESPN ROSTER CUTOFF SUGGESTIONS
## Player pool size should be roughly 115% of total league roster spots available
# Underdog Fantasy Player Cutoffs
QB_CUTOFF <- 30
RB_CUTOFF <- 82
WR_CUTOFF <- 104
TE_CUTOFF <- 28

# Draftkings Fantasy Player Cutoffs
# QB_CUTOFF <- 32
# RB_CUTOFF <- 90
# WR_CUTOFF <- 118
# TE_CUTOFF <- 36

# # ESPN Draft Cutoffs
# QB_CUTOFF <- 24
# RB_CUTOFF <- 51
# WR_CUTOFF <- 64
# TE_CUTOFF <- 25

# # Yahoo Fantasy Player Cutoffs
# QB_CUTOFF <- 26
# RB_CUTOFF <- 45
# WR_CUTOFF <- 55
# TE_CUTOFF <- 25

# Setting the VORP cutoff to identify how a "replacement" player performs at the position
VORP_CUTOFF <- 0.66

# Defining the Dampening variable to adjust final relative value scores
# z-scores get thrown off due to large expert projections
# Generally, WRs are going to be the most valuable position by ADP
# rankings should roughly reflect expected positional scarcity when drafting
QB_DAMP <- 0.6
RB_DAMP <- 1.1
TE_DAMP <- 0.9
WR_DAMP <- 1.2

# Function for estimating optimal K value
plot_wss_elbow <- function(player_df, pos = NULL, max_k = 25) {
  # Optionally filter by position
  df_attrs <- player_df %>%
    { if (!is.null(pos)) filter(., Pos == pos) else . } %>%
    select(Relative_Value)
  
  # Calculate WSS
  wss <- numeric(max_k)
  for (i in 1:max_k) {
    km.out <- kmeans(df_attrs, centers = i, nstart = 50)
    wss[i] <- km.out$tot.withinss
  }
  
  # Plot elbow chart
  elbow_df <- data.frame(k = 1:max_k, wss = wss)
  ggplot(elbow_df, aes(x = k, y = wss)) +
    geom_line(color = "#3A6C99", linewidth = .75) +
    geom_point(color = "#4682B4", size = 1.5) +
    labs(
      title = paste("Elbow Plot for", pos, "Cluster WSS"),
      x = "Number of Clusters (k)",
      y = "Within-Cluster Sum of Squares"
    ) +
    theme_minimal()
}

# Function for creating player positional tiers
assign_positional_tiers <- function(player_df, pos, k = 5) {
  
  # Filtering total df to the position group
  pos_df <-
    player_df %>%
    filter(Pos == pos)
  
  # Filter to the positional group and extract attribute for clustering
  pos_attrs <- 
    pos_df %>%
    select(Relative_Value)

  # Run k-means clustering
  set.seed(123)  # for reproducibility
  kmeans_result <- kmeans(pos_attrs, centers = k, nstart = 50)
  
  # Assign tier back to dataframe, adjust tier labels to correspond to player value
  pos_df <- 
    pos_df %>%
    mutate(Pos_Tier = kmeans_result$cluster) %>%
    group_by(Pos_Tier) %>%
    mutate(tier_avg = mean(Relative_Value)) %>%
    ungroup() %>%
    mutate(Pos_Tier = dense_rank(desc(tier_avg))) %>%
    arrange(desc(Relative_Value)) %>%
    select(-tier_avg)
  
  return(pos_df)
}

# Visualizing the tier structure for a single position: every player is plotted at their
# positional ranking (x) against their blended Final_Projection (y), colored by Pos_Tier so the
# tier breaks read as bands down the projection curve. The top player (highest projection) in each
# tier is labeled by name to anchor where each tier begins. Colors come from the MetBrewer
# "Hiroshige" palette, interpolated across however many tiers the position has.
#
# Set interactive = TRUE to return a ggiraph htmlwidget instead of a static ggplot: every point
# gains a hover tooltip (player name, positional rank, projection), which surfaces the names of the
# players that aren't the per-tier leader label. The static repel labels render underneath unchanged.
# In the interactive view the hover ring and the tooltip text both take the hovered player's tier
# color (points are drawn as filled rings so the outline can be recolored), on a white tooltip card.
plot_positional_tiers <- function(player_df, pos = "RB", interactive = FALSE) {
  # Filter to the position, order by rank, and treat the tier as an ordered factor for coloring
  pos_df <-
    player_df %>%
    filter(Pos == pos) %>%
    arrange(Pos_Ranking) %>%
    mutate(Pos_Tier = factor(Pos_Tier, levels = sort(unique(Pos_Tier))))

  # The highest-projected player in each tier anchors that tier's label
  tier_leaders <-
    pos_df %>%
    group_by(Pos_Tier) %>%
    slice_max(Final_Projection, n = 1, with_ties = FALSE) %>%
    ungroup()

  # Pull one Hiroshige color per tier (continuous interpolation handles any tier count)
  tier_colors <- met.brewer("Hiroshige", n = nlevels(pos_df$Pos_Tier), type = "continuous")

  # Precompute each row's tier hex + an HTML tooltip whose text is colored to that tier, so the
  # tooltip can sit on a white card and still read in the tier color (inline color beats the card CSS)
  pos_df <-
    pos_df %>%
    mutate(
      tier_hex = tier_colors[as.integer(Pos_Tier)],
      tooltip_html = paste0(
        "<span style='color:", tier_hex, ";'>",
        "<b>", Player, " - ", Pos, Pos_Ranking, "</b></span>"
      )
    )

  # Swap in ggiraph's interactive point layer when requested. Points are drawn as shape 21 (a filled
  # ring): the ring color is mapped to Pos_Tier, so the hover effect only has to thicken it - keeping
  # the highlight in the tier's own color. A plain geom_point is used for the static version.
  point_layer <-
    if (interactive) {
      geom_point_interactive(
        aes(fill = Pos_Tier, tooltip = tooltip_html, data_id = Player),
        shape = 21, size = 2, stroke = 0.4, alpha = 0.9
      )
    } else {
      geom_point(size = 2, alpha = 0.9)
    }

  tier_plot <-
    ggplot(pos_df, aes(x = Pos_Ranking, y = Final_Projection, color = Pos_Tier)) +
    point_layer +
    # Name the top player of each tier, nudged clear of the points with connector lines
    geom_text_repel(
      data = tier_leaders,
      aes(label = Player),
      size = 3.2, fontface = "bold", show.legend = FALSE,
      min.segment.length = 0, box.padding = 0.6, max.overlaps = Inf, seed = 42
    ) +
    scale_color_manual(values = tier_colors, name = "Tier") +
    labs(
      title = paste0(pos, " Tiers by Positional Ranking"),
      subtitle = "Blended final projection vs positional rank, colored by tier (top player per tier labeled)",
      x = "Positional Ranking",
      y = "Final Projection"
    ) +
    theme_minimal(base_size = 13) +
    theme(
      plot.title = element_text(colour = "#262626", size = 16, face = "bold"),
      plot.subtitle = element_text(colour = "#595959", size = 11),
      axis.text = element_text(colour = "#262626"),
      panel.grid.minor = element_blank()
    )

  # Static ggplot for the plot pane
  if (!interactive) return(tier_plot)

  # Match the ring fill to the same palette (merges into the single "Tier" legend), then render.
  # Hover thickens the tier-colored ring; the tooltip is a white card with tier-colored inline text.
  tier_plot <- tier_plot + scale_fill_manual(values = tier_colors, name = "Tier")

  girafe(
    ggobj = tier_plot,
    width_svg = 8, height_svg = 6,
    options = list(
      opts_hover(css = "stroke-width:2.5px;"),
      opts_tooltip(
        use_fill = FALSE, use_stroke = FALSE,
        css = paste0(
          "background-color:#ffffff;border:1px solid #d9d9d9;border-radius:4px;",
          "padding:6px 8px;font-size:12px;box-shadow:1px 1px 4px rgba(0,0,0,0.25);"
        )
      )
    )
  )
}

# Scatter of the model prediction (x) against the FantasyPros expert projection (y), colored by
# tier. A dashed line marks agreement: points above it are players the experts are higher on than the
# model (expert reaches), points below are the model's relative values / sleepers. The n_label
# players with the largest *percentage* divergence between the two estimates (how much the higher
# estimate exceeds the lower, in either direction) are named so the sharpest outliers stand out.
# Pass a position to focus on one group (colors then use the positional tier); leave pos = NULL for
# the whole board (colors use the overall tier). interactive = TRUE mirrors plot_positional_tiers:
# hover surfaces each player's name/values with a tier-colored ring and tooltip text on a white card.
# Supply replacement_points (the Pos / Replacement_Value frame built earlier in this script) to gate
# the outlier LABELS to players at or above their position's replacement level - a percentage metric
# otherwise gets dominated by deep, low-projection players where a small point gap is a huge percent.
plot_model_vs_expert <- function(player_df, pos = NULL, interactive = FALSE, n_label = 10,
                                 replacement_points = NULL, width_svg = 12, height_svg = 8) {
  # Optionally focus on one position, and keep only players that have both estimates to compare
  plot_df <-
    player_df %>%
    { if (!is.null(pos)) filter(., Pos == pos) else . } %>%
    filter(!is.na(Model_Prediction), !is.na(FantasyPros_Prediction))

  # Color by the positional tier when focused on a position, otherwise the overall tier
  tier_source <- if (is.null(pos)) plot_df$Tier else plot_df$Pos_Tier
  plot_df <-
    plot_df %>%
    mutate(Tier_grp = factor(tier_source, levels = sort(unique(tier_source))))

  # One Hiroshige color per tier (continuous interpolation handles any tier count)
  tier_colors <- met.brewer("Hiroshige", n = nlevels(plot_df$Tier_grp), type = "continuous")

  # Per-row tier hex, plus two divergence measures: the signed point gap (for direction) and the
  # percent divergence (how much the higher estimate exceeds the lower - the outlier ranking metric).
  # The denominator is floored at 1 so a near-zero projection can't manufacture a spurious outlier.
  plot_df <-
    plot_df %>%
    mutate(
      tier_hex = tier_colors[as.integer(Tier_grp)],
      proj_gap = FantasyPros_Prediction - Model_Prediction,  # + = expert higher, - = model higher
      pct_gap = (pmax(Model_Prediction, FantasyPros_Prediction) /
                   pmax(pmin(Model_Prediction, FantasyPros_Prediction), 1) - 1) * 100,
      tooltip_html = paste0(
        "<span style='color:", tier_hex, ";'>",
        "<b>", Player, "</b><br/>",
        if_else(proj_gap >= 0, "Expert +", "Model +"), as.character(round(abs(proj_gap), 1)),
        " (", round(pct_gap), "%)",
        "</span>"
      )
    )

  # Restrict the label candidates to players at/above their position's replacement level (when a
  # replacement_points frame is supplied), so the percentage outliers reflect draftable players
  # rather than deep guys whose small point gap is a large percent of a tiny projection
  label_pool <- plot_df
  if (!is.null(replacement_points)) {
    label_pool <-
      plot_df %>%
      left_join(replacement_points, by = "Pos") %>%
      filter(is.na(Replacement_Value) | Final_Projection >= Replacement_Value)
  }

  # The players the model and expert diverge on most in percentage terms (the top outliers)
  divergers <-
    label_pool %>%
    slice_max(pct_gap, n = n_label, with_ties = FALSE)

  # Interactive filled-ring points (recolorable outline) or plain points for the static version
  point_layer <-
    if (interactive) {
      geom_point_interactive(
        aes(fill = Tier_grp, tooltip = tooltip_html, data_id = Player),
        shape = 21, size = 2.5, stroke = 0.4, alpha = 0.9
      )
    } else {
      geom_point(size = 2.5, alpha = 0.9)
    }

  mve_plot <-
    ggplot(plot_df, aes(x = Model_Prediction, y = FantasyPros_Prediction, color = Tier_grp)) +
    # Agreement line: model == expert
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#999999", linewidth = 0.4, alpha=0.9) +
    point_layer +
    # Name the biggest percentage model-vs-expert outliers
    geom_text_repel(
      data = divergers,
      aes(label = Player),
      size = 3.6, fontface = "bold", show.legend = FALSE,
      min.segment.length = 0, box.padding = 0.5, max.overlaps = Inf, seed = 42
    ) +
    # Corner annotations naming each side of the agreement line (replaces the subtitle):
    # top-left = experts higher than the model, bottom-right = model higher than experts
    annotate("text", x = -Inf, y = Inf, label = "Expert Favored",
             hjust = -0.3, vjust = 10.0, fontface = "bold", size = 7, alpha = 0.4, color = "#595959") +
    annotate("text", x = Inf, y = -Inf, label = "Model Favored",
             hjust = 1.5, vjust = -10, fontface = "bold", size = 7, alpha = 0.4, color = "#595959") +
    scale_color_manual(values = tier_colors, name = "Tier") +
    labs(
      title = paste0("Model vs Expert Projection", if (!is.null(pos)) paste0(" - ", pos) else ""),
      x = "Model Prediction",
      y = "FantasyPros Projection"
    ) +
    theme_minimal(base_size = 15) +
    theme(
      plot.title = element_text(colour = "#262626", size = 19, face = "bold"),
      axis.title = element_text(colour = "#262626", size = 15),
      axis.text = element_text(colour = "#262626", size = 13),
      legend.title = element_text(size = 14),
      legend.text = element_text(size = 12),
      panel.grid.minor = element_blank()
    )

  # Static ggplot for the plot pane
  if (!interactive) return(mve_plot)

  # Match ring fill to the palette (single "Tier" legend), then render interactively
  mve_plot <- mve_plot + scale_fill_manual(values = tier_colors, name = "Tier")

  girafe(
    ggobj = mve_plot,
    width_svg = width_svg, height_svg = height_svg,
    options = list(
      opts_hover(css = "stroke-width:2.5px;"),
      opts_tooltip(
        use_fill = FALSE, use_stroke = FALSE,
        css = paste0(
          "background-color:#ffffff;border:1px solid #d9d9d9;border-radius:4px;",
          "padding:6px 8px;font-size:12px;box-shadow:1px 1px 4px rgba(0,0,0,0.25);"
        )
      )
    )
  )
}

# Reading in the player projection's data
player_df <- read_csv(paste0("data/blended_proj_", as.character(PRED_YEAR), "_", SCORING_TYPE, ".csv"))

# Trimming down the player pool to the positional cutoff points
player_df <-
  player_df %>%
  group_by(Pos) %>%
  mutate(FP_Pos_Ranking = dense_rank(desc(FantasyPros_Prediction)),
         Pos_Ranking = dense_rank(desc(Final_Projection))) %>%
  ungroup() %>%
  filter(Pos == "QB" & Pos_Ranking <= QB_CUTOFF |
           Pos == "RB" & Pos_Ranking <= RB_CUTOFF |
           Pos == "WR" & Pos_Ranking <= WR_CUTOFF |
           Pos == "TE" & Pos_Ranking <= TE_CUTOFF)

# Creating positional value over replacement (VORP) tiers
# What does the "replacement" point value for each positional group?
# These replacement values will be used to determine the relative value of each player
## Replacement level by position based on defined cutoffs
replacement_points <- 
  player_df %>%
  filter(
    (Pos == "QB" & Pos_Ranking == round(QB_CUTOFF * VORP_CUTOFF)) |
      (Pos == "RB" & Pos_Ranking == round(RB_CUTOFF * VORP_CUTOFF)) |
      (Pos == "WR" & Pos_Ranking == round(WR_CUTOFF * VORP_CUTOFF)) |
      (Pos == "TE" & Pos_Ranking == round(TE_CUTOFF * VORP_CUTOFF))
  ) %>%
  select(Pos, Replacement_Value = Final_Projection)


########################## Overall Tiers #######################################
# Creating a "Relative Value" metric based on how valuable a player relative to the rest of the player pool above the positional cutoffs
# Ranking each player by their Final Projection value
# Filtering data so that only the top N players by position group (defined by above cutoffs) are included in the final rankings
total_df <-
  player_df %>%
  left_join(replacement_points, by = "Pos") %>%
  group_by(Pos) %>%
  arrange(Pos_Ranking, .by_group = TRUE) %>%
  mutate(
    # Z-score scaled projection
    z_score_value = ((Final_Projection - mean(Final_Projection)) / sd(Final_Projection)) * Final_Projection,
    
    # VORP = projection - replacement-level projection
    vorp = (Final_Projection - Replacement_Value),
    
    # Combine Z and VORP with weights, multiplying VORP by 1.3 to increase distribution for value estimation accuracy (1.3 was selected after regression analysis, see estimate_vorp_zscore..)
    Relative_Value = 0.5 * z_score_value + 0.5 * (vorp * 1.3)
  ) %>%
  # Adjusting the Relative Value based on position, QBs are generally less valuable in best ball fantasy football, so we reduce their value slightly
  mutate(
    Pos_Adjustment = case_when(
      Pos == "QB" ~ QB_DAMP,   # Reduce QB value
      Pos == "TE" ~ TE_DAMP,   # Reduce TE value
      Pos == "RB" ~ RB_DAMP,   # Reduce RB value
      Pos == "WR" ~ WR_DAMP,   # Increase WR value
      TRUE ~ 1
    ),
    Relative_Value = Relative_Value * Pos_Adjustment
  ) %>%
  mutate(Relative_Value = sort(Relative_Value, decreasing = TRUE)) %>% # Forcing player rankings to be in direct value order, z-scoring can get funky sometimes
  ungroup() %>%
  mutate(Overall_Ranking = dense_rank(desc(Relative_Value))) %>%
  arrange(Pos, desc(Relative_Value))

plot_wss_elbow(total_df, max_k = 25)

total_attrs <-
  total_df %>%
  select(Relative_Value)

#### Obtaining clusters with optimal K value
kmeans_attrs <- kmeans(total_attrs, centers = 9, nstart = 50)

total_df$Tier <- kmeans_attrs$cluster

# Build the tier dataset straight from total_df, which already carries every needed column.
# This replaces a name-based self-join that collided for same-named players (e.g. the two
# Adrian Petersons). Re-rank by cluster mean so Tier 1 is the most valuable group.
tier_df <-
  total_df %>%
  select(player_id, Player, Pos, Model_Prediction, FantasyPros_Prediction, Final_Projection,
         Floor, Ceiling, upside_index, ceiling_index, floor_index,   # model's own range/upside, carried to the final sheet
         FP_Pos_Ranking, Pos_Ranking, Overall_Ranking, Relative_Value, Tier) %>%
  group_by(Tier) %>%
  mutate(tier_avg = mean(Relative_Value)) %>%
  ungroup() %>%
  mutate(Tier = dense_rank(desc(tier_avg))) %>%
  arrange(Pos, desc(Relative_Value)) %>%
  select(-tier_avg)

########################## Positional Tiers ####################################
# Assign positional tiers for each position

## Quarterbacks
plot_wss_elbow(tier_df, pos = "QB", max_k = 15)

qb_df <- assign_positional_tiers(tier_df, pos = "QB", k = 8)

## Runningbacks
plot_wss_elbow(tier_df, pos = "RB", max_k = 15)

rb_df <- assign_positional_tiers(tier_df, pos = "RB", k = 7)

## Wide Receivers
plot_wss_elbow(tier_df, pos = "WR", max_k = 15)

wr_df <- assign_positional_tiers(tier_df, pos = "WR", k = 6)

## Tight Ends
plot_wss_elbow(tier_df, pos = "TE", max_k = 15)

te_df <- assign_positional_tiers(tier_df, pos = "TE", k = 6)

# Combine all positional tiers into a single dataframe
final_df <- bind_rows(list(qb_df, rb_df, wr_df, te_df)) %>%
  arrange(desc(Relative_Value), Tier, Pos_Tier) %>%
  select(player_id, Player, Pos, Model_Prediction, FantasyPros_Prediction,
         Final_Projection, Floor, Ceiling, ceiling_index, floor_index, upside_index,
         FP_Pos_Ranking, Pos_Ranking, Overall_Ranking,
         Relative_Value, Tier, Pos_Tier)

# Inspect each position's tier structure (projection vs positional rank, colored by tier)
plot_positional_tiers(final_df, pos = "QB", interactive = TRUE)
plot_positional_tiers(final_df, pos = "RB", interactive = TRUE)
plot_positional_tiers(final_df, pos = "WR", interactive = TRUE)
plot_positional_tiers(final_df, pos = "TE", interactive = TRUE)

# Model vs expert agreement: whole board (overall tiers) and per position (positional tiers).
# Pass replacement_points so the labeled outliers are gated to at/above-replacement players.
plot_model_vs_expert(final_df, pos = "QB", interactive = TRUE, replacement_points = replacement_points)
plot_model_vs_expert(final_df, pos = "RB", interactive = TRUE, replacement_points = replacement_points)
plot_model_vs_expert(final_df, pos = "WR", interactive = TRUE, replacement_points = replacement_points)
plot_model_vs_expert(final_df, pos = "TE", interactive = TRUE, replacement_points = replacement_points)

########################## MISSION COMPLETE ####################################
# Save the final dataframe with tiers
fwrite(final_df, paste0("data/final_projections_", as.character(PRED_YEAR), "_", SCORING_TYPE, ".csv"))

