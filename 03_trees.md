# Tree-based methods



In this chapter we will touch upon the most popular tree-based methods used in machine learning. Haven't heard of the term "tree-based methods"? Do not panic. The idea behind tree-based methods is very simple and we'll explain how they work step by step through the basics. Most of the material on this chapter was built upon @boehmke2019 and @james2013. 

Before we begin, let's load `tidyflow` and `tidymodels` and read the data that we'll be using.


```r
library(tidymodels)
library(tidyflow)
library(rpart.plot)

data_link <- "https://raw.githubusercontent.com/cimentadaj/ml_socsci/master/data/pisa_us_2018.csv"
pisa <- read.csv(data_link)
```

## Decision trees

Decision trees are simple models. In fact, they are even simpler than linear models. They require little statistical background and are in fact among the simplest models to communicate to a general audience. In particular, the visualizations used for decision trees are very powerful in conveying information and can even serve as an exploratory avenue for social research.

## TODO
Add here that there are no coefficients. WHAT? Yes, you heard it right!

Throughout this chapter, we'll be using the PISA data set from the regularization chapter. On this example we'll be focusing on predicting the `math_score` of students in the United States, based on the socio economic status of the parents (named `HISEI` in the data; the higher the `HISEI` variable, the higher the socio economic status), the father's education (named `FISCED` in the data; coded as several categories from 0 to 6 where 6 is high education) and whether the child repeated a grade (named `REPEAT` in the data). On the other hand, `REPEAT` is a dummy variable where `1` means the child repeated a grade and `0` no repetition.

Decision trees, as their name conveys, are tree-like diagrams. The work by defining yes-or-no rules based on the data to try to predict the most common value in each final branch. Let's begin learning about decision trees by looking at one:

<img src="03_trees_files/figure-html/unnamed-chunk-3-1.png" width="672" />

In this example the top-most box which says `HISEI < 56` is the **root node**. This is the most important variable that predicts `math_score`. Inside the blue box you can see two numbers: $100\%$ which means that the entire sample is present in this **node** and the number `474`, the average test score for mathematics for the entire sample:

<img src="03_trees_files/figure-html/unnamed-chunk-4-1.png" width="672" />

On both sides of the root node (`HISEI < 56`) there is a `yes` and a `no`. Decision trees work by **partitioning** variables into `yes-or-no` branches. The `yes` branch satisfies the name of **root** (`HISEI < 56`) and always branches out to the left:

<img src="03_trees_files/figure-html/unnamed-chunk-5-1.png" width="672" />

In contrast, the `no` branch always branches out to the right:

<img src="03_trees_files/figure-html/unnamed-chunk-6-1.png" width="672" />

The criteria for separating into `yes-or-no` branches is that respondents must be very similar within branches and very different between branches (later in this chapter I will explain in detail which criteria is used and how). The decision tree figures out that respondents that have an `HISEI` below $56$ and above $56$ are the most different with respect to the mathematics score. The left branch (where there is a `yes` in the **root node**) are those which have a `HISEI` below 56 and the right branch (where there is a `no`) are those which have a `HISEI` above $56$. Let's call these two groups the low and high SES respectively. If we look at the two boxes that come down from these branches, the low SES branch has an average math score of $446$ while the high SES branch has an average test score of $501$:

<img src="03_trees_files/figure-html/unnamed-chunk-7-1.png" width="672" />

For the sake of simplicity, let's focus now on the branch of the low SES group (the left branch). The second node coming out of the low SES branch contains 50\% of the sample and an average math score of $446$. This is the node with the rule `REPEAT >= 0.5`:

<img src="03_trees_files/figure-html/unnamed-chunk-8-1.png" width="672" />

This 'intermediate' node is called **internal node**. For calculating this **internal node**, the decision tree algorithm limits the entire data set to only those which have low SES (literally, the decision tree does something like `pisa[pisa$HISEI < 56, ]`) and asks the same question that it did in the **root node**: of all the variables in the model which one separates two branches such that respondents are very similar within the branch but very different between the branches with respect to `math_score`? 

For those with low SES background, this variable is whether the child repeated a grade or not. In particular, those coming from low SES background which repeated a grade, had an average math score of $387$ whereas those who didn't have an average math score of $456$:

<img src="03_trees_files/figure-html/unnamed-chunk-9-1.png" width="672" />

These two nodes at the bottom are called **leaf nodes** because they are like the 'leafs of the tree'. **Leaf nodes**  are of particular importance because they are the ones that dictate what the final value of `math_score` will be. Any new data that is predicted with this model will always give an average `math_score` of $456$ for those of low SES background who didn't repeat a grade:

<img src="03_trees_files/figure-html/unnamed-chunk-10-1.png" width="672" />

Similarly, any respondent from high SES background, with a highly educated father who didn't repeat a grade, will get assigned a `math_score` of $527$:

<img src="03_trees_files/figure-html/unnamed-chunk-11-1.png" width="672" />

That is it. That is a decision tree in it's simplest form. It contains a **root node** and several **internal** and **leaf nodes** and it can be interpreted just as we just did. The right branch of the tree can be summarized with the same interpretation. For example, for high SES respondents, father's education (`FISCED`) is more important than `REPEAT` to separate between math scores:

<img src="03_trees_files/figure-html/unnamed-chunk-12-1.png" width="672" />

This is the case because it comes first in the tree. Substantially, this might be due to the fact that there is higher variation in education credentials for parents of high SES background than for those of low SES background. We can see that those with the highest father's education (`FISCED` above $5.5$), the average math score is $524$ whereas those with father's education below $5.5$ have a math score of $478$:

<img src="03_trees_files/figure-html/unnamed-chunk-13-1.png" width="672" />

I hope that these examples show that decision trees are a great tool for exploratory analysis and I strongly believe they have an inmense potential for exploring interactions in social science research. In case you didn't notice it, we literally just interpreted an interaction term that social scientists would routinely use in linear models. Without having to worry about statistical significance or plotting marginal effects, social scientists can use decision trees as an exploratory medium to understand interactions in an intuitive way. 

You might be asking yourself, how do we fit these models and visualize them? `tidyflow` and `tidymodels` have got you covered. For example, for fitting the model from above, we can begin our `tidyflow`, add a split, a formula and define the decision tree:


```r
# Define the decision tree and tell it the the dependent
# variable is continuous ('mode' = 'regression')
mod1 <- set_engine(decision_tree(mode = "regression"), "rpart")

tflow <-
  # Plug the data
  pisa %>%
  # Begin the tidyflow
  tidyflow(seed = 23151) %>%
  # Separate the data into training/testing
  plug_split(initial_split) %>%
  # Plug the formula
  plug_formula(math_score ~ FISCED + HISEI + REPEAT) %>%
  # Plug the model
  plug_model(mod1)

vanilla_fit <- fit(tflow)
tree <- pull_tflow_fit(vanilla_fit)$fit
rpart.plot(tree)
```

If you read the chapter on reguralization, the only thing new here should be `rpart.plot`. All `plug_*` functions serve to build your machine learning workflow and the model `decision_tree` is the equivalent of `linear_reg` that we saw in the previous chapter. We are just recycling the same code for this model. `rpart.plot` on the other hand, is a function used specifically for plotting the decision tree (that is why we loaded the package `rpart.plot` at the beginning). No need to delve much into this function. It just works if you pass it a decision tree model: that is why `pull` the model fit before calling it.

Now I've told all the good things about decision trees but they are not a smoking gun. They have serious limitations. In particular, there are two that we'll discuss in this chapter. The first one is that decision trees tend to overfit a lot. Substantially, this example doesn't make much sense but look at the percentages in the **leaf nodes**:


```r
tflow <-
  tflow %>%
  replace_formula(ST163Q03HA ~ .)

fit_complex <- fit(tflow)
tree <- pull_tflow_fit(fit_complex)$fit
rpart.plot(tree)
```

<img src="03_trees_files/figure-html/unnamed-chunk-15-1.png" width="672" />

Decision trees can capture a lot of noise. $5$ out of the $8$ **leaf nodes** have less than $2\%$ of the sample. These are **leaf nodes** with very weak statistical power:

<img src="03_trees_files/figure-html/unnamed-chunk-16-1.png" width="672" />

What would happen if a tiny %1\% of those **leaf nodes** respondend **slightly** different? It is possible we get a complete different tree. Decision trees are not well known for being robust. In fact, it is one of its main weaknesses. However, decision trees have an argument called `min_n` that force the tree to discard any **node** that has a number of observations below your minimum. Let's run the model above and set the minimum number of observation per **node** to be $200$:


```r
dectree <- update(mod1, min_n = 250)
tflow <-
  tflow %>%
  replace_model(dectree)

fit_complex <- fit(tflow)
tree <- pull_tflow_fit(fit_complex)$fit
rpart.plot(tree)
```

<img src="03_trees_files/figure-html/unnamed-chunk-17-1.png" width="672" />

The tree was reduced considerably now. There are fewer **leaf nodes** and all nodes have a greater sample size than before. 

You might be wondering: what should the minimum sample size be? It depends. The rule of thumb should be relative to your data. In particular, the identification of small nodes should be analyzed with care. Perhaps there **is** a group of outliers that consitute a node and it's not a problem of statistical noise. By increasing the minimum sample size for each node you would be destroying that statistical finding. For example, suppose we are studying welfare social expenditure as the dependent variable and then we had other independent variables, among which are country names. Scandinavian countries might group pretty well into a solitary node because they are super powers in welfare spending (these are Denmark, Norway, Sweden and Finland). If we increased the minimum sample size to $10$, we might group them with Germany and France, which are completely different in substantive terms. The best rule of thumb I can recommend is no other than to study your problem at hand with great care and make decisions accordingly. It might make sense to increase the sample or it might not depending on the research question, the total sample size, whether you're exploring the data or whether you're interested in predicting on new data.

The second problem with decision trees is the tree depth. As can be seen from the previous plot, decision trees can create **leaf nodes** which are very small. In other more complicated scenarios, your tree might get huge. Yes, huge:

<img src="./img/large_tree.png" width="100%" />

More often that not, these huge trees are just overfitting the data. They are creating very small nodes that capture noise from the data and when you're predicting on new data, they perform terribly bad.

## LEFT OFF HERE


```r
dectree2 <- set_engine(update(mod1, min_n = 200, tree_depth = 5), "rpart", model = TRUE)

tflow <-
  tflow %>%
  replace_formula(math_score ~ . - scie_score - read_score) %>% 
  replace_model(dectree2)

fit_complex <- fit(tflow)
tree <- pull_tflow_fit(fit_complex)$fit
rpart.plot(tree)
```

<img src="03_trees_files/figure-html/unnamed-chunk-19-1.png" width="672" />

Due to lack of space, we won't cover tree pruning but it is a very important technique used to generate very complex trees and 'prune' them to avoid overfitting.

### How do they really work?

Before we go into

<img src="03_trees_files/figure-html/unnamed-chunk-20-1.png" width="672" />

Then we apply a random split


```r
p2
```

<img src="03_trees_files/figure-html/unnamed-chunk-21-1.png" width="672" />


```r
p3
```

<img src="03_trees_files/figure-html/unnamed-chunk-22-1.png" width="672" />


```r
p4
```

<img src="03_trees_files/figure-html/unnamed-chunk-23-1.png" width="672" />


```r
p_many
```

<img src="03_trees_files/figure-html/unnamed-chunk-24-1.png" width="672" />





```
## # A tibble: 8 x 3
##   variable random_split total_rss            
##   <chr>    <chr>        <chr>                
## 1 "HISEI"  41.22        "Total RSS: 34423362"
## 2 "HISEI"  53.77        "Total RSS: 34400218"
## 3 "HISEI"  56.57        "Total RSS: 32523560"
## 4 ""       ...          ""                   
## 5 "FISCED" 2            "Total RSS: 35901660"
## 6 "FISCED" 1            "Total RSS: 36085201"
## 7 "FISCED" 5            "Total RSS: 34083264"
## 8 ""       ...          ""
```


