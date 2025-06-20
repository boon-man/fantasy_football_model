required_packages <- c(
  "rvest", "dplyr", "stringr", "forecast", "tidyr",
  "zoo", "ggplot2", "lubridate", "data.table", "prophet"
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

EVAL_YEAR <- 2024

# Function to scrape data from Pro Football Reference
scrapeData = function(urlprefix, urlend, startyr, endyr, stat) {
  master <- data.frame()
  for (i in startyr:endyr) {
    Sys.sleep(5)
    cat('Loading Year', i, '\n')
    URL <- paste(urlprefix, i, urlend, sep = "")
    table <-
      read_html(URL) %>%
      html_node('table') %>%
      html_table()
    
    table$Year <- i
    master <- rbind(table, master)
  }
  assign(quo_name(enquo(stat)), master, envir=.GlobalEnv)
  return('Complete')
}

scrapeData("https://www.pro-football-reference.com/years/",
           "/receiving.htm",
           2006,
           EVAL_YEAR,
           'receiving')

scrapeData("https://www.pro-football-reference.com/years/",
           "/rushing.htm",
           2006,
           EVAL_YEAR,
           'rushing')

scrapeData("https://www.pro-football-reference.com/years/",
           "/passing.htm",
           2006,
           EVAL_YEAR,
           'passing')

# Receiving data cleaning
colnames(receiving)[which(names(receiving) == "Ctch%")] <- "Catch_Percent"
receiving$Player <- gsub("[^[:alnum:][:space:]]","",receiving$Player)
receiving$Player <- str_squish(receiving$Player)
receiving$Catch_Percent <- gsub("%","",receiving$Catch_Percent)
receiving <- receiving[!grepl("Rk", receiving$Rk),]
receiving_drop <- c('Player', 'Tm', 'Pos')
receiving <-
  receiving %>%
  mutate_at(vars(-one_of(receiving_drop)), list(as.numeric)) %>%
  replace(is.na(.), 0) %>%
  select(-c(Rk)) %>%
  rename(receiving_yds = Yds,
         receiving_td = TD,
         receiving_long = Lng,
         receiving_y_g = `Y/G`,
         receiving_fmb = Fmb)

# Rushing data cleaning, 1st row of rushing data must be removed due to duplicate headers
rushing <- rushing[-1, ]
rushingcols <- c('Rank', 'Player', 'Tm', 'Age', 'Pos', 'G', 'GS',
                 'rush_att', 'rush_yds', 'rush_td', 'rush_1D', 'rush_success_pct', 
                 'rush_long', 'rush_yds_att', 'rush_yds_game', 'rush_fbl', 'Year')
colnames(rushing) <- rushingcols
rushing$Player <- gsub("[^[:alnum:][:space:]]","",rushing$Player)
rushing$Player <- str_squish(rushing$Player)
rushing <- rushing[!grepl("Rk", rushing$Rank),]
rushing_drop <- c('Player', 'Tm', 'Pos')
rushing <-
  rushing %>%
  mutate_at(vars(-one_of(rushing_drop)), list(as.numeric)) %>%
  replace(is.na(.), 0) %>%
  select(-c(Rank))


# Passing data cleaning
passing$Player <- gsub("[^[:alnum:][:space:]]","",passing$Player)
passing$Player <- str_squish(passing$Player)
passing <- passing[!grepl("Rk", passing$Rk),]
names(passing)[27]<-"Sack_Yds"
passing_drop <- c('Player', 'Tm', 'Pos')
passing <-
  passing %>%
  select(-QBrec) %>%
  mutate_at(vars(-one_of(passing_drop)), list(as.numeric)) %>%
  replace(is.na(.), 0) %>%
  rename(passing_comp = Cmp,
         passing_att = Att,
         passing_yards = Yds,
         passing_td = TD,
         passing_td_percent = `TD%`,
         passing_long = Lng,
         passing_yards_att = `Y/A`,
         passing_avg_yards_att = `AY/A`,
         passing_yards_comp = `Y/C`,
         passing_yards_game = `Y/G`,
         sack_percent = `Sk%`) %>%
  select(-c(Rk, `NY/A`, `ANY/A`))

#### Full dataset for forecasting
combined <-
  receiving %>%
  full_join(rushing, by = c('Player', 'Tm', 'Year', 'Age', 'G', 'GS', 'Pos')) %>%
  full_join(passing, by = c('Player', 'Tm', 'Year', 'Age', 'G', 'GS', 'Pos')) %>%
  select(Player, Year, Pos, everything()) %>%
  replace(is.na(.), 0) %>%
  mutate(points = (receiving_td * 6) + (receiving_yds * .1) + (Rec * .5) +
           (rush_td * 6) + (rush_yds * .1) +
           (passing_yards * .04) + (passing_td * 4) -
           (receiving_fmb * 2) - (rush_fbl * 2) - (Int * 2)) %>%
  arrange(Player, Year) %>%
  group_by(Player) %>%
  filter(any(Year == EVAL_YEAR)) %>%
  mutate(Pos = last(Pos)) %>% # Each player's most recent position will be used for their historical performance eval
  ungroup() %>%
  select(Player, Year, Pos, points) %>%
  mutate(Year = as.Date(as.yearmon(Year)))


#### Predicting total points for players with 4 or more years of experience
prophet_df <- 
  combined %>%
  group_by(Player) %>%
  select(Player, Year, Pos, points) %>%
  rename(ds = Year,
         y = points) %>%
  filter(n() >= 4)

## Selecting the most recent 4 years of performance for player trend analysis
positional_trends <-
  prophet_df %>%
  group_by(Player, Pos) %>%
  slice_tail(n = 4) %>%
  summarise(sd_trend = sd(y),
            avg_trend = mean(y)) %>%
  ungroup() %>%
  drop_na() %>%
  group_by(Pos) %>%
  summarise(pos_sd_trend = sd(sd_trend),
            pos_avg_trend = mean(avg_trend))

trend_df <-
  prophet_df %>%
  group_by(Player, Pos) %>%
  slice_tail(n = 4) %>%
  summarise(last_year = last(y),
            sd_trend = sd(y),
            avg_trend = mean(y)) %>%
  ungroup() %>%
  inner_join(positional_trends, by = 'Pos') %>%
  mutate(relative_sd = sd_trend / pos_sd_trend)
  

prophet_pred <-
  prophet_df %>%
  do(predict(prophet(.), make_future_dataframe(prophet(.), periods = 1, freq = 'year')))

prophet_final <-
  prophet_pred %>%
  filter(ds > '2023-01-01') %>%
  transmute(ds = as.Date(ds),
            y = yhat)

# prophet_test <-
#   prophet_df %>%
#   rbind(prophet_final) %>%
#   arrange(Player, ds) %>% 
#   group_by(Player) %>%
#   filter(sum(y) > 40) %>%
#   ungroup() %>%
#   subset(Player %in% with(prophet_df, sample(unique(Player), 8)))

# ggplot(prophet_test, aes(x = ds, y = y, color = Player)) +
#   geom_line()

prophet_2024 <-
  prophet_final %>%
  transmute(Year = as.Date(ds),
            points = y) %>%
  ungroup() %>%
  left_join(trend_df, by = 'Player') %>%
  mutate(pred_vs_last = (points - last_year),
         pred_vs_avg = (points - avg_trend)) %>%
  mutate(corrected_points = case_when(
    (relative_sd > 2.75) & (points > 0) ~ avg_trend,
    .default = points))

prophet_viz <-
  prophet_2024 %>%
  select(Player, Year, Pos, corrected_points) %>%
  rename(ds = Year,
            y = corrected_points)

test_pos <- 'WR'

prophet_test <-
  prophet_df %>%
  rbind(prophet_viz) %>%
  filter(Pos == test_pos) %>%
  arrange(Player, ds) %>%
  group_by(Player) %>%
  filter(sum(y) > 50) %>%
  ungroup() %>%
  subset(Player %in% with(prophet_viz, sample(unique(Player), 20)))

ggplot(prophet_test, aes(x = ds, y = y, color = Player)) +
  geom_line()


#### Average increase in points from year 1 to 2
year1to2 <-
  combined %>%
  group_by(Player) %>%
  mutate(experience = row_number()) %>%
  filter(experience <= 2,
         points >= 50) %>%
  mutate(diff = ((points - lag(points)) / lag(points)))
summary(year1to2)
sd(year1to2$diff, na.rm = TRUE)
#### mean: .305, sd = 0.593


#### Predicting 2023 for 2022 rookies
rookies <-
  combined %>%
  group_by(Player) %>%
  filter(n() == 1,
         Year >= '2023-01-01') %>%
  ungroup()

num_sims <- 20
rooksims <- matrix(nrow = nrow(rookies), ncol = num_sims)
for(i in 1:num_sims){
  rooksims[,i] <- rnorm(nrow(rookies), .305, 0.593)
}

rook_changes <- rowMeans(rooksims)
hist(rook_changes)

rookies_2024 <-
  rookies %>%
  transmute(Player = Player,
            Year = Year + duration(1, units = 'years'),
            points = (points + (points * rook_changes)))

#### Average increase in points from year 2 to 3
year2to3 <-
  combined %>%
  group_by(Player) %>%
  mutate(experience = row_number()) %>%
  filter(experience >= 2,
         experience <= 3,
         points >= 75) %>%
  mutate(diff = ((points - lag(points)) / lag(points)))
summary(year2to3)
sd(year2to3$diff, na.rm = TRUE)
#### mean = .051, sd = .403

#### Predicting 2020 for 2018 sophomores
sophomores <-
  combined %>%
  group_by(Player) %>%
  filter(n() == 2,
         Year >= '2022-01-01') %>%
  slice(2) %>%
  ungroup()

num_sims <- 20
sophsims <- matrix(nrow = nrow(sophomores), ncol = num_sims)
for(i in 1:num_sims){
  sophsims[,i] <- rnorm(nrow(sophomores), .051, .403)
}

soph_changes <- rowMeans(sophsims)
hist(soph_changes)

sophmores_2024 <-
  sophomores %>%
  transmute(Player = Player,
            Year = Year + duration(1, units = 'years'),
            points = (points + (points * soph_changes)))

#### Average increase in points from year 3 to 4
year3to4 <-
  combined %>%
  group_by(Player) %>%
  mutate(experience = row_number()) %>%
  filter(experience >= 3,
         experience <= 4,
         points >= 75) %>%
  mutate(diff = ((points - lag(points)) / lag(points)))
summary(year3to4)
sd(year3to4$diff, na.rm = TRUE)
#### mean: .082, sd = .481

#### Predicting 2021 for 2018 sophmores
juniors <-
  combined %>%
  group_by(Player) %>%
  filter(n() == 3,
         Year >= '2021-01-01') %>%
  slice(3) %>%
  ungroup()

num_sims <- 20
junsims <- matrix(nrow = nrow(juniors), ncol = num_sims)
for(i in 1:num_sims){
  junsims[,i] <- rnorm(nrow(juniors), .082, .481)
}

jun_changes <- rowMeans(junsims)
hist(jun_changes)

juniors_2024 <-
  juniors %>%
  transmute(Player = Player,
            Year = Year + duration(1, units = 'years'),
            points = (points + (points * jun_changes)))

#############################################################
##### Every player prediction for 2024
positions <-
  receiving %>%
  full_join(rushing, by = c('Player', 'Tm', 'Year', 'Age', 'G', 'GS', 'Pos')) %>%
  full_join(passing, by = c('Player', 'Tm', 'Year', 'Age', 'G', 'GS', 'Pos')) %>%
  filter(Pos != '') %>%
  arrange(Player, Year) %>%
  group_by(Player) %>%
  summarise(Pos = toupper(last(Pos))) %>%
  ungroup()

final_2024 <-
  prophet_2024 %>%
  select(Player, Year, points) %>%
  rbind(rookies_2024) %>%
  rbind(sophmores_2024) %>%
  rbind(juniors_2024) %>%
  left_join(positions, by = 'Player') %>%
  select(Player,
         Pos,
         points)

# all_data <-
#   prophet_2023 %>%
#   rbind(rookies_2023) %>%
#   rbind(sophmores_2023) %>%
#   rbind(juniors_2023) %>%
#   rbind(combined) %>%
#   arrange(Player, Year) %>%
#   filter(Year >= '2015-01-01')

fwrite(final_2024, '2024_proj_ppr.csv')
