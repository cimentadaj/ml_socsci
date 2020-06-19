library(dplyr)
library(here)

dt_all <- read.csv(here("data", "university_ranking.csv"))

final_df <-
  dt_all %>%
  select(institution, everything(), -score)

write.csv(final_df,
          here("data", "university_ranking_final.csv"),
          row.names = FALSE)
