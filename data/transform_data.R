library(haven)
library(recipes)
library(here)
library(readr)
library(dplyr)

pisa <- read_spss(here("data", "pisa_2018.sav"))

pisa_us <-
  pisa %>%
  filter(CNT == "USA") %>% 
  mutate_if(is.labelled, zap_labels) %>% 
  mutate_at(vars(MISCED, FISCED, REPEAT, IMMIG, DURECEC), as.character)


calculate_pv <- function(dt, type) {
  pv_ind <- grepl(paste0("PV.+", type), names(dt))
  dt[[paste0(tolower(type), "_score")]] <- rowMeans(dt[pv_ind])
  dt
}

# Calculate the average test score in mathematics as the average of all
# 10 plausible columns
pisa_us <-
  pisa_us %>%
  calculate_pv("MATH") %>%
  calculate_pv("READ") %>%
  calculate_pv("SCIE")

noncogn_questions <- c("ST182Q03HA",
                       "ST182Q04HA",
                       "ST182Q05HA",
                       "ST182Q06HA")

pisa_us$noncogn <- rowMeans(pisa_us[noncogn_questions], na.rm = TRUE)

pisa_us_ready <-
  pisa_us %>%
  # WORKMAST is a index that has the noncogn questions
  select(-starts_with("PV"),
         -starts_with("W_"),
         -all_of(noncogn_questions),
         -VER_DAT,
         -WORKMAST,
         -STRATUM,
         -COMPETE,
         -RESILIENCE,
         -MASTGOAL) %>%
  recipe(math_score ~ ., data = .) %>%
  step_zv(all_predictors()) %>%
  step_nzv(all_predictors()) %>% 
  step_meanimpute(all_predictors(), -all_nominal()) %>% 
  step_modeimpute(all_predictors(), -all_numeric()) %>%
  prep() %>%
  juice()

write_csv(pisa_us_ready, here("data", "pisa_us_2018.csv"))

