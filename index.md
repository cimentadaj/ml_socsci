--- 
title: "Machine Learning for Social Scientists"
author: "Jorge Cimentada"
date: "2020-07-07"
site: bookdown::bookdown_site
documentclass: book
bibliography: [book.bib]
biblio-style: apalike
link-citations: yes
github-repo: cimentadaj/ml_socsci
url: 'http\://cimentadaj.github.io/ml_socsci/'
description: "Notes, content and exercises for the RECSM 2020 course Machine Learning for Social Scientists."
---

# Preface {-}

Notes, content and exercises for the RECSM 2020 course Machine Learning for Social Scientists. These are intended to introduce social scientists to concepts in machine learning using traditional social science examples and datasets. Currently, it is not intended to be a book but rather supporting material for the course. Perhaps it evolves enough to be a book some day.

To be able to run everything in the material and the slides, make sure you can install the packages in the code chunk below. If you're using Windows, be sure to have the latest version of R and Rtools installed before installing these packages. You can download Rtools from [here](https://cran.r-project.org/bin/windows/Rtools/). Read the instructions in detail to make sure it's installed correctly. 


```r
all_pkgs <- c('devtools', 'tidymodels', 'ggplot2', 'baguette', 'rpart.plot', 'vip', 'plotly', 'dplyr', 'ggfortify', 'tidyflow', 'tidyr')
install.packages(all_pkgs, dependencies = TRUE)
devtools::install_github("cimentadaj/tidyflow")
```

These lines of code can take a while to install (more than 30 minutes), so don't worry. 

Once that's finished, make sure you all of these are installed with this:


```r
setdiff(c(all_pkgs, "tidyflow"), row.names(installed.packages()))
```

The expected result should be `character()`

Slides for course:

Day 1 - Morning: [Slides](./slides/day1_morning/01_introduction.html)

Day 1 - Afternoon: [Slides](./slides/day1_afternoon/02_regularization_afternoon.html)

Day 2 - Morning: [Slides](./slides/day2_morning/03_loss_trees.html)

Day 2 - Afternoon: [Slides](./slides/day2_afternoon/tree_methods.html)

Day 3 - Morning: [Slides](./slides/day3_morning/boosting_pca.html)

Day 3 - Afternoon: [Slides](./slides/day3_afternoon/kmeans_competition.html)
