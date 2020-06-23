library(dplyr)
library(here)

city_data <- read.csv(here("data", "wiki_main.csv"))
county_data <- read.csv(here("data", "census_cleaned.csv"))

internet_usage <-
  read.csv(here("data", "census_internet.csv")) %>%
  select(GEO.id2, HC02_EST_VC18) %>%
  slice(2:nrow(.)) %>%
  rename(county_fips = GEO.id2,
         internet_usage = HC02_EST_VC18) %>%
  mutate(county_fips = as.numeric(county_fips),
         internet_usage = as.numeric(internet_usage))

res <-
  city_data %>%
  group_by(county_fips, city, hyperlink) %>%
  summarize(across(c(longitude, latitude, author, created_at, modified_at), unique),
            across(c(population,
                     density,
                     watchers,
                     pageviews,
                     pageviews_offset,
                     revisions,
                     editors,
                     author_editcount,
                     secs_since_last_edit,
                     characters,
                     words,
                     references,
                     unique_references,
                     sections,
                     external_links,
                     links_from_this_page,
                     links_to_this_page,
                     redirects_count),
                   sum, na.rm = TRUE),
            .groups = "drop_last") %>%
  ungroup()

merged_data <-
  res %>%
  inner_join(county_data, by = c("county_fips" = "GEOID")) %>%
  inner_join(internet_usage, by = "county_fips") %>%
  select(-X_merge)

merged_final <-
  merged_data %>%
  group_by(county_fips) %>% 
  summarize_all(mean, na.rm = TRUE) %>%
  select_if(function(x) !all(is.na(x), na.rm = TRUE))

write.csv(merged_final,
          here("data", "wikipedia_final.csv"),
          row.names = FALSE)
