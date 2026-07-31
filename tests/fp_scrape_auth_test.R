##############################################################################
### FantasyPros projections intake - diagnostic and investigation record
###
### Not a unit test. Two purposes:
###   1. Re-diagnosing the intake when it breaks (the usual symptom being a warning that a
###      position came back with ~10 rows, which means the Chrome login has lapsed).
###   2. Recording WHY the intake is built the way it is, including the two approaches that were
###      tried and abandoned - so neither gets re-attempted from scratch.
###
### The production functions live in functions.R and are sourced below rather than copied, so this
### script exercises the real code. What stays here is only the diagnostic scaffolding.
###
### WHAT WAS ESTABLISHED (all by testing against the live site, 2026-07-31)
###
### There is no CSV endpoint. The site's own download link (?export=xls) returns the ordinary HTML
### page; the CSV is built in the browser by window.exportTableToCSV, which serializes the rendered
### DOM table. So the manual exports in data/fp_raw/ and the scrape read the very same table -
### confirmed by the parity block below at a median difference of 0.0 across all four positions.
### That is also the origin of the exports' odd shape: the NBSP spacer row is the table's
### stat-group banner (PASSING / RUSHING / MISC), and the duplicate YDS/TDS/ATT headers are the
### rushing and receiving groups colliding.
###
### The fence is server-side row truncation to 10 players (window.registrationFence), not
### user-agent filtering. A retired scraper here sent a browser user-agent and retried with a
### back-off, which could never have worked: retrying does not defeat an auth gate. Confirmed
### further by driving a real headless browser while logged out - also 10 rows, with JS adding
### none, which is what proves the full table is server-rendered once authenticated.
###
### DEAD END 1 - pasting cookies. Django's `sessionid` from secure.fantasypros.com does not lift
### the fence on www.fantasypros.com (Apache/PHP - a different application), and hand-assembling a
### complete Cookie header from DevTools proved error-prone. The httr scaffolding for this is kept
### below since it is still the cheapest way to inspect what the server thinks of a request.
###
### DEAD END 2 - scripted login. The login form intercepts submit, calls grecaptcha.execute() to
### mint a reCAPTCHA v3 token into a hidden field, and only then posts. A POST with an empty token
### is rejected as *bad credentials* - the generic Django message - even when the credentials are
### valid and work in a browser. A v3 token cannot be minted outside a browser, so this is closed.
### fp_login() below is retained only as the evidence.
###
### THE APPROACH THAT WORKS - headless Chrome against a dedicated, persistent profile. A human
### clears reCAPTCHA v3 once; Chrome persists the cookie jar in the profile directory, so later
### runs are already authenticated. No password or cookie is stored by this repo.
###
### ONE-TIME SETUP (and whenever the login lapses) - run in a terminal, log in, CLOSE the window:
###   open -na "Google Chrome" --args --user-data-dir=~/.fp_chrome_profile https://secure.fantasypros.com/accounts/login/
### Chrome permits one process per profile directory, so that window must be closed before scraping.

source("00_globals.R")
source("functions.R")  # the production intake: scrape_fp_projections, read_fp_projections_csv, ...

FP_POSITIONS <- c("qb", "rb", "wr", "te")
FP_USER_AGENT <- "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
FP_LOGIN_URL <- "https://secure.fantasypros.com/accounts/login/"


### Plain-HTTP scaffolding (for inspecting what the server makes of a request) -----------

split_cookie_pairs <- function(cookie_header) {
  cookie_header %>%
    str_split(";\\s*") %>%
    unlist() %>%
    keep(str_detect, pattern = "=")
}

# Optional: FP_COOKIE (a whole browser Cookie header) or FP_SESSIONID (the single Django cookie),
# both from .Renviron. Only used by the diagnostic blocks - the production path needs neither.
fp_cookie_header <- function() {
  full_cookie <- Sys.getenv("FP_COOKIE")
  if (nzchar(full_cookie)) return(str_remove_all(str_trim(full_cookie), "^['\"]|['\"]$"))

  session_id <- Sys.getenv("FP_SESSIONID")
  if (!nzchar(session_id)) {
    stop("Neither FP_COOKIE nor FP_SESSIONID is set (only the diagnostic blocks need them).",
         call. = FALSE)
  }
  # Tolerating the shapes a copy-paste arrives in: a whole pair, or a quoted value
  paste0("sessionid=", session_id %>%
           str_trim() %>%
           str_remove("^sessionid=") %>%
           str_remove_all("^['\"]|['\"]$"))
}

# Reporting cookie NAMES only, never values - catches the common paste mistakes
report_cookie_config <- function() {
  cat("FP_COOKIE chars:   ", nchar(Sys.getenv("FP_COOKIE")), "\n", sep = "")
  cat("FP_SESSIONID chars:", nchar(Sys.getenv("FP_SESSIONID")), "\n", sep = "")
  if (!nzchar(Sys.getenv("FP_COOKIE")) && !nzchar(Sys.getenv("FP_SESSIONID"))) {
    cat("\nNeither is set. Note .Renviron is read only at R startup - or call readRenviron().\n")
    return(invisible(NULL))
  }

  pairs <- split_cookie_pairs(fp_cookie_header())
  cookie_names <- sort(str_extract(pairs, "^[^=]+"))
  cat("\nCookie pairs parsed:", length(pairs), "\nNames:", paste(cookie_names, collapse = ", "), "\n")

  # sessionid is Django's; fptoken/fpuserinfo are the cross-subdomain pair the PHP site reads.
  # fp_level is set even anonymously, so it proves nothing about being logged in.
  expected <- c("sessionid", "fptoken", "fpuserinfo")
  cat("\nAuth cookies present:\n")
  print(tibble(cookie = expected, present = expected %in% cookie_names))

  if (length(pairs) <= 1) {
    cat("\nOnly one pair parsed - that is a single cookie value, not a whole Cookie header.\n")
  }
  invisible(NULL)
}

fetch_fp_page <- function(position, scoring = SCORING_TYPE, authenticated = FALSE, cookie = NULL) {
  headers <- c("Accept-Language" = "en-US,en;q=0.9")
  if (authenticated) {
    # Raw header rather than set_cookies() so a browser cookie string passes through byte for byte
    headers <- c(headers, Cookie = if (is.null(cookie)) fp_cookie_header() else cookie)
  }
  GET(
    fp_projection_url(position, scoring),
    user_agent(FP_USER_AGENT),
    do.call(add_headers, as.list(headers))
  )
}

# Reporting whether the SERVER considers us logged in, independently of the row count - this is
# what tells apart a rejected cookie from an accepted one feeding a client-side-hydrated table.
report_auth_state <- function(resp, label) {
  page_text <- content(resp, as = "text", encoding = "UTF-8")

  # The registration fence ships its own JSON payload server-side; is_visible flips to false once
  # the account is recognized, which makes it the cleanest logged-in signal on the page
  fence_visible <- str_match(page_text, '"id":"registration-fence".*?"is_visible":\\s*(true|false)')[, 2]

  tibble(
    label = label,
    status = status_code(resp),
    rows = length(html_elements(read_html(page_text), "table#data tbody tr")),
    fence_visible = coalesce(fence_visible, "<not found>"),
    has_sign_in = str_detect(page_text, ">\\s*Sign In\\s*<"),
    x_cache = coalesce(headers(resp)[["x-cache"]], "<none>")
  )
}


### Anonymous baseline, and any cookie experiment ---------------------------------------
### The control case: expect ~10 rows and fence_visible = true. Establishing this means a healthy
### row count elsewhere can only be explained by authentication.

anon_resp <- fetch_fp_page("qb", authenticated = FALSE)
print(as.data.frame(report_auth_state(anon_resp, "anonymous")))

# With a cookie, if you want to revisit DEAD END 1. Reading the result:
#   fence_visible = true,  has_sign_in = TRUE  -> cookie rejected (still anonymous)
#   fence_visible = false, rows still ~10      -> accepted, but the table is client-side
#   fence_visible = false, rows >> 10          -> cookies would work after all
# report_cookie_config()
# print(as.data.frame(report_auth_state(fetch_fp_page("qb", authenticated = TRUE), "cookie")))


### The production scrape ---------------------------------------------------------------
### Calls the real function from functions.R. Expect roughly QB 82 / RB 131 / WR 189 / TE 117.
### A position reporting ~10 rows means the profile login has lapsed - re-run the setup command
### at the top of this file.

scraped <-
  bind_rows(scrape_fp_projections(FP_POSITIONS)) %>%
  mutate(Projected_Points = as.numeric(str_remove_all(Projected_Points, "[A-Za-z,]"))) %>%
  drop_na(Projected_Points)

print(count(scraped, Pos))


### Parity against the manual exports ---------------------------------------------------
### The correctness check: scraped numbers must match the CSVs that previously drove the pipeline.
### Points agree to rounding (median difference 0.0); the remaining gaps are deep-bench players and
### genuine consensus drift, since FantasyPros re-runs projections continuously while an export is
### a point-in-time snapshot.

exported <-
  bind_rows(lapply(FP_POSITIONS, read_fp_projections_csv, scoring = SCORING_TYPE)) %>%
  mutate(Projected_Points = as.numeric(str_remove_all(Projected_Points, "[A-Za-z,]"))) %>%
  drop_na(Projected_Points)

parity <-
  scraped %>%
  mutate(Player_clean = clean_player_name(Player)) %>%
  full_join(
    exported %>% mutate(Player_clean = clean_player_name(Player)),
    by = c("Player_clean", "Pos"),
    suffix = c("_scraped", "_exported")
  ) %>%
  mutate(points_diff = abs(Projected_Points_scraped - Projected_Points_exported))

print(as.data.frame(
  parity %>%
    group_by(Pos) %>%
    summarise(
      n_scraped = sum(!is.na(Projected_Points_scraped)),
      n_exported = sum(!is.na(Projected_Points_exported)),
      n_matched = sum(!is.na(Projected_Points_scraped) & !is.na(Projected_Points_exported)),
      median_diff = median(points_diff, na.rm = TRUE),
      max_diff = max(points_diff, na.rm = TRUE),
      .groups = "drop"
    )
))

cat("\n-- Scraped but not in the export --\n")
parity %>%
  filter(is.na(Projected_Points_exported)) %>%
  arrange(desc(Projected_Points_scraped)) %>%
  select(Player_scraped, Pos, Projected_Points_scraped) %>%
  print(n = 30)

cat("\n-- In the export but not scraped --\n")
parity %>%
  filter(is.na(Projected_Points_scraped)) %>%
  arrange(desc(Projected_Points_exported)) %>%
  select(Player_exported, Pos, Projected_Points_exported) %>%
  print(n = 30)

cat("\n-- Largest matched disagreements --\n")
parity %>%
  filter(!is.na(points_diff)) %>%
  arrange(desc(points_diff)) %>%
  select(Player_scraped, Pos, Projected_Points_scraped, Projected_Points_exported, points_diff) %>%
  print(n = 15)


### Column-shape check ------------------------------------------------------------------
### Confirming FPTS is still the last column for every position, since both readers locate it with
### last(which(col_names == "FPTS")) to survive the duplicated YDS/TDS/ATT headers. Run this if a
### position ever starts returning odd point values - a layout change would show up here first.

report_table_shape <- function(position) {
  page <- read_html(content(fetch_fp_page(position), as = "text", encoding = "UTF-8"))
  col_names <-
    page %>%
    html_element("table#data") %>%
    html_elements("thead tr th") %>%
    html_text2() %>%
    str_trim() %>%
    str_to_upper()

  Sys.sleep(2)
  tibble(
    Pos = toupper(position),
    n_cols = length(col_names),
    fpts_col = last(which(col_names == "FPTS")),
    fpts_is_last = last(which(col_names == "FPTS")) == length(col_names),
    header = paste(col_names, collapse = "|")
  )
}

print(as.data.frame(bind_rows(lapply(FP_POSITIONS, report_table_shape))))


### DEAD END 2, kept as evidence: scripted login ----------------------------------------
### Rejected with "Please enter a correct username and password." even on valid credentials,
### because `token` (the reCAPTCHA v3 slot filled by grecaptcha.execute() in the browser) is empty.
### Credentials, if you re-test: FP_USERNAME / FP_PASSWORD in .Renviron. Note .Renviron drops
### backslashes and expands ${...} even inside quotes, so check nchar() matches the real password.

cookie_header_from_response <- function(resp) {
  jar <- cookies(resp)
  if (nrow(jar) == 0) return("")
  jar %>%
    filter(str_detect(domain, "fantasypros\\.com")) %>%
    mutate(pair = paste0(name, "=", value)) %>%
    pull(pair) %>%
    paste(collapse = "; ")
}

fp_login <- function(username = Sys.getenv("FP_USERNAME"), password = Sys.getenv("FP_PASSWORD")) {
  if (!nzchar(username) || !nzchar(password)) {
    stop("FP_USERNAME and FP_PASSWORD must both be set in .Renviron (then restart R).", call. = FALSE)
  }
  cat("Username length:", nchar(username), "| password length:", nchar(password), "\n")

  # One handle for the whole exchange so the csrftoken cookie from the GET is returned with the
  # POST - Django matches the cookie against the form field and rejects the form otherwise
  login_handle <- handle(FP_LOGIN_URL)
  form_page <- GET(FP_LOGIN_URL, user_agent(FP_USER_AGENT), handle = login_handle)
  csrf_token <-
    content(form_page, as = "text", encoding = "UTF-8") %>%
    read_html() %>%
    html_element("input[name='csrfmiddlewaretoken']") %>%
    html_attr("value")

  if (is.na(csrf_token)) stop("No csrfmiddlewaretoken on the login page - the form changed.", call. = FALSE)

  # Referer is required by Django's CSRF check over HTTPS. `token` is the reCAPTCHA v3 slot, and
  # sending it empty is exactly what gets this rejected.
  login_resp <- POST(
    FP_LOGIN_URL,
    user_agent(FP_USER_AGENT),
    add_headers(Referer = FP_LOGIN_URL),
    body = list(csrfmiddlewaretoken = csrf_token, username = username, password = password,
                token = "", captchaversion = "3", import_id = ""),
    encode = "form",
    handle = login_handle
  )

  body_text <- content(login_resp, as = "text", encoding = "UTF-8")
  auth_cookies <- intersect(c("sessionid", "fptoken", "fpuserinfo"), cookies(login_resp)$name)

  cat("Status:", status_code(login_resp), "| final URL:", login_resp$url, "\n")
  cat("Auth cookies:", if (length(auth_cookies)) paste(auth_cookies, collapse = ", ") else "<none>", "\n")
  cat("Still on the login form:", str_detect(body_text, "name=['\"]csrfmiddlewaretoken['\"]"), "\n")

  page_messages <-
    read_html(body_text) %>%
    html_elements(".alert, .errorlist, .help-block, .invalid-feedback") %>%
    html_text2() %>%
    str_squish() %>%
    keep(nzchar)
  if (length(page_messages)) cat("Messages:", paste(unique(page_messages), collapse = " | "), "\n")

  invisible(cookie_header_from_response(login_resp))
}
