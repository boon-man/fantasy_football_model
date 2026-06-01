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