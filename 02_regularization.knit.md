# Regularization



Regularization is a common topic in machine learning and bayesian statistics. In this document, we will describe the three most common regularized linear models in the machine learning literature and introduce them in the context of the PISA data set. At the end of the document you'll find exercises that will put your knowledge to the test. Most of this material is built upon Boehmke & Greenwell (2019) and Friedman et al. (2001).

## Ridge regularization

Do no let others fool you into thinking that ridge regression is a fancy artificial intelligence algorithm. Are you familiar with linear regression? If you are, then ridge regression is just a very **simple** adaptation of linear regression. 

The whole aim of linear regression, or Ordinary Least Squares (OLS), is to minimize the sum of the squared residuals. In other words, fit `N` number of regression lines to the data and keep only the one that has the lowest sum of squared residuals. In simple formula jargon, OLS tries to **minimize** this:

\begin{equation}
RSS = \sum_{k = 1}^n(actual_i - predicted_i)^2
\end{equation}

For each fitted regression line, you compare the predicted value ($predicted_i$) versus the actual value ($actual_i$), square it, and add it all up. Each fitted regression line then has an associated Residual Sum of Squares (RSS) and the linear model chooses the line with the lowest RSS.

> Note: Social scientists are familiar with the RSS and call it just by it's name. However, be aware that in machine learning jargon, the RSS belongs to a general family called  **loss functions**. Loss functions are metrics that evaluate the **fit** of your model and there are many around (such as AIC, BIC or R2).

Ridge regression takes the previous RSS loss function and adds one term:

\begin{equation}
RSS + \lambda \sum_{k = 1}^n \beta^2_j
\end{equation}

The new term is called a *shrinkage penalty* because it forces each coefficient $\beta_j$ closer to zero by squaring it. The shrinkage part is clearer once you think of this term as forcing each coefficient to be as small as possible but also considering having the smallest Residual Sum of Squares (RSS). In other words, we want the smallest coefficients that don't affect the fit of the line (RSS).

An intuitive example is to think of RSS and $\sum_{k = 1}^n \beta^2_j$ as two separate things. RSS estimates how the model fits the data and $\sum_{k = 1}^n \beta^2_j$ limits how much you overfit the data. Finally, the little $\lambda$ between these two terms can be interpreted as a "weight". The higher the lambda, the higher the weight that will be given to the shrinkage term of the equation. If $\lambda$ is 0, then multiplying 0 by $\sum_{k = 1}^n \beta^2_j$ will always return zero, forcing our previous equation to simply be reduced to the single term $RSS$.

Why is there a need to "limit" how well the model fits the data? Because we, social scientists and data scientists, very commonly **overfit** the data. The plot below shows a simulation from [Simon Jackson](https://drsimonj.svbtle.com/ridge-regression-with-glmnet) where we can see that when tested on a training set, OLS and Ridge tend to overfit the data. However, when tested on the test data, Ridge regression has lower out of sample error as the $R2$ is higher for models with different observations.

<img src="./figs/unnamed-chunk-1-1.png" width="80%" style="display: block; margin: auto;" />

The strength of the ridge regression comes from the fact that it compromises fitting the training data really well for improved generalization. In other words, we increase **bias** (because we force the coefficients to be smaller) for lower **variance** (but we make it more general). In other words, the whole gist behind ridge regression is penalizing very large coefficients for better generalization. 

Having that intuition in mind, the predictors of the ridge regression need to be standardized. Why is this the case? Because due to the scale of a predictor, its coefficient can be more penalized than other predictors. Suppose that you have the income of a particular person (measured in thousands per months) and time spent with their families (measured in seconds) and you're trying to predict happiness. A one unit increase in salary could be penalized much more than a one unit increase in time spent with their families **just** because a one unit increase in salary can be much bigger due to it's metric.

In R, you can fit a ridge regression (and nearly all other machine learning models) through the `caret` package. Let's load the packages that we will work with and read the data:









































