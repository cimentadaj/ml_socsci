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

# Calculate the average test score in mathematics as the average of all
# 10 plausible columns
pv_ind <- grepl("PV.+MATH", names(pisa_us))
pisa_us$math_score <- rowMeans(pisa_us[pv_ind])

pisa_us_ready <-
  pisa_us %>%
  select(-matches("PV.+MATH")) %>% 
  recipe(math_score ~ ., data = .) %>%
  step_zv(all_predictors()) %>% 
  step_meanimpute(all_predictors(), -all_nominal()) %>% 
  step_modeimpute(all_predictors(), -all_numeric()) %>%
  prep() %>%
  juice()

write_csv(pisa_us_ready, here("data", "pisa_us_2018.csv"))
