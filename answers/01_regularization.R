library(caret)
library(vip)
library(rsample)
library(dplyr)

data_link <- "https://raw.githubusercontent.com/cimentadaj/ml_socsci/master/data/pisa_us_2018.csv"
pisa <- read.csv(data_link)

### Split the data into test/training data

# Remember to set the seed to `2341` so that everyone can compare their results.

set.seed(2341)
split_pisa <- initial_split(data = pisa, prop = .7)
pisa_test <- testing(split_pisa)
pisa_train <- training(split_pisa)

### Run a ridge regression with non-cognitive as the dependent variable

# Use as many variables as you want (you can reuse the previous variables from the examples or pick all of them). A formula of the like `noncogn ~ .` will regress `noncogn` on all variables.

# Define ridge grid of values for lambda
ridge_grid <- data.frame(
  lambda = seq(0, 3, length.out = 100),
  alpha = 0
)

# Use the train function to train the model on the *training set*
ridge_mod <- train(
  form = noncogn ~ .,
  data = pisa_train,
  method = "glmnet",
  preProc = c("center", "scale"),
  tuneGrid = ridge_grid,
  trControl = trainControl(method = "cv", number = 5)
)

# Extract the best lambda and calculate the RMSE
# on the test set
best_lambda_ridge <- ridge_mod$bestTune$lambda
holdout_ridge <-
  RMSE(
    predict(ridge_mod, pisa_test),
    pisa_test$noncogn
  )

# Extract the RMSE of the training set
train_rmse_ridge <-
  ridge_mod$results %>%
  filter(alpha == ridge_mod$bestTune$alpha, lambda == best_lambda_ridge) %>%
  pull(RMSE)

# Compare both holdout and training RMSE
c(holdout_rmse = holdout_ridge, train_rmse = train_rmse_ridge)

### Which are the most important variables?

# Comment on their coefficients and whether they make sense to be included in the model.

vip(ridge_mod)

### Run a lasso regression with the same specification as above

# Define ridge grid of values for lambda
# Reproduce previous steps

# Define ridge grid of values for lambda
lasso_grid <- data.frame(
  lambda = seq(0, 3, length.out = 100),
  alpha = 0
)

# Use the train function to train the model on the *training set*
lasso_mod <- train(
  form = noncogn ~ .,
  data = pisa_train,
  method = "glmnet",
  preProc = c("center", "scale"),
  tuneGrid = lasso_grid,
  trControl = trainControl(method = "cv", number = 5)
)

# Extract the best lambda and calculate the RMSE
# on the test set
best_lambda_lasso <- lasso_mod$bestTune$lambda
holdout_lasso <-
  RMSE(
    predict(lasso_mod, pisa_test),
    pisa_test$noncogn
  )

# Extract the RMSE of the training set
train_rmse_lasso <-
  lasso_mod$results %>%
  filter(alpha == lasso_mod$bestTune$alpha, lambda == best_lambda_lasso) %>%
  pull(RMSE)

# Compare both holdout and training RMSE
c(holdout_rmse = holdout_lasso, train_rmse = train_rmse_lasso)

# Which model is performing better? Ridge or Lasso? Are the same variables the strongest predictors across models? Which variables are the strongest predictors?

### Run an elastic net regression on non cognitive skills

# Since `train` already takes care of trying all possible values, there's no need to pass a grid of lambda values. It is only needed to set the `tuneLength` to a number of alpha values.

# Use the train function to train the model on the *training set*
elnet_mod <- train(
  form = noncogn ~ .,
  data = pisa_train,
  method = "glmnet",
  preProc = c("center", "scale"),
  trControl = trainControl(method = "cv", number = 5),
  tuneLength = 5
)

# Extract the best lambda and calculate the RMSE
# on the test set
best_lambda_elnet <- elnet_mod$bestTune$lambda
holdout_elnet <-
  RMSE(
    predict(elnet_mod, pisa_test),
    pisa_test$noncogn
  )

# Extract the RMSE of the training set
train_rmse_elnet <-
  elnet_mod$results %>%
  filter(alpha == elnet_mod$bestTune$alpha, lambda == best_lambda_elnet) %>%
  pull(RMSE)

# Compare both holdout and training RMSE
c(holdout_rmse = holdout_elnet, train_rmse = train_rmse_elnet)

### Compare the three models graphically
# * Comment on which models is better in out-of-sample fit
# * Is it better to keep the most accurate model or a model that includes relevant confounders (even if they're relationship is somewhat weak)?
