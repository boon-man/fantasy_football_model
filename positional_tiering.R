##############################################################################
### Positional Tiering
library(tidyverse)

player_df <- read_csv('2024_proj_ppr_tiered.csv')

########################## QB Tiers ############################################
qb_df <-
  player_df %>%
  filter(Pos == 'QB') %>%
  arrange(desc(points))

qb_attrs <-
  qb_df %>% 
  select(points)

#### Initialize total within sum of squares error: wss
wss <- 0

for (i in 1:15) {
 km.out <- kmeans(qb_attrs, centers = i, nstart = 20)
 wss[i] <- km.out$tot.withinss
}

### Plot total within sum of squares vs. number of clusters
plot(1:15, wss, type = "b",
    xlab = "Number of Clusters",
    ylab = "Within groups sum of squares")
#### The optimal number of clusters is 5

#### Obtaining clusters with k = 5
kmeans_attrs <- kmeans(qb_attrs, 5, nstart = 20)

qb_df$tier <- kmeans_attrs$cluster

########################## RB Tiers ############################################

rb_df <-
  player_df %>%
  filter(Pos == 'RB') %>%
  arrange(desc(points))

rb_attrs <-
  rb_df %>% 
  select(points)

#### Initialize total within sum of squares error: wss
wss <- 0

for (i in 1:15) {
  km.out <- kmeans(rb_attrs, centers = i, nstart = 20)
  wss[i] <- km.out$tot.withinss
}

### Plot total within sum of squares vs. number of clusters
plot(1:15, wss, type = "b",
     xlab = "Number of Clusters",
     ylab = "Within groups sum of squares")
#### The optimal number of clusters is 7

#### Obtaining clusters with k = 7.
kmeans_attrs <- kmeans(rb_attrs, 7, nstart = 20)

rb_df$tier <- kmeans_attrs$cluster

########################## WR Tiers ############################################

wr_df <-
  player_df %>%
  filter(Pos == 'WR') %>%
  arrange(desc(points))

wr_attrs <-
  wr_df %>% 
  select(points)

#### Initialize total within sum of squares error: wss
wss <- 0

for (i in 1:15) {
  km.out <- kmeans(wr_attrs, centers = i, nstart = 20)
  wss[i] <- km.out$tot.withinss
}

### Plot total within sum of squares vs. number of clusters
plot(1:15, wss, type = "b",
     xlab = "Number of Clusters",
     ylab = "Within groups sum of squares")
#### The optimal number of clusters is 8

#### Obtaining clusters with k = 8.
kmeans_attrs <- kmeans(wr_attrs, 8, nstart = 20)

wr_df$tier <- kmeans_attrs$cluster

########################## TE Tiers ############################################

te_df <-
  player_df %>%
  filter(Pos == 'TE') %>%
  arrange(desc(points))

te_attrs <-
  te_df %>% 
  select(points)

#### Initialize total within sum of squares error: wss
wss <- 0

for (i in 1:15) {
  km.out <- kmeans(te_attrs, centers = i, nstart = 20)
  wss[i] <- km.out$tot.withinss
}

### Plot total within sum of squares vs. number of clusters
plot(1:15, wss, type = "b",
     xlab = "Number of Clusters",
     ylab = "Within groups sum of squares")
#### The optimal number of clusters is 6

#### Obtaining clusters with k = 6.
kmeans_attrs <- kmeans(te_attrs, 6, nstart = 20)

te_df$tier <- kmeans_attrs$cluster
