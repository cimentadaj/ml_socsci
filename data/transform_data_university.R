library(dplyr)

dt_all <- read.csv("./university_ranking.csv")

final_df <-
  dt_all %>%
  select(institution, everything(), -score)

write.csv(final_df, "university_ranking_final.csv")
