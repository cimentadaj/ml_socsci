# No free lunch



Throughout this course we've explained several different methods that are used in machine learning for predictive problems. Although we presented the benefits and pitfalls of each one when possible, there's no clear cut rule on which one to use. The 'No free lunch' theorem is a simple axiom that states that since every predictive algorithm has different assumptions, no single model is known to perform better than all others *a priori*. In other words, machine learning practitioners need to try different models to check which one predicts better for their task. 

However, for different scenarios, this might be different. Let's discuss some hypothetial scenarios.

## Causal Inference

There is growing interesting from the social science literature on achieving causal inference using tree-based methods [@athey2016]. By definition, this type of analysis is not interested in predictive accuracy alone. This means that we would not try several different models and check which one is better. Instead, we need to carefully understand how tree-based methods work and how they can help us estimate a causal effect. 

## Explanaining complex models

In business settings, there are scenarios where interpretability is often needed more than accuracy. For example, for explaining a complex model to key stakeholders it is sometimes better to have a simple model that performs worse but to be able to walk through the stakeholder into how the final prediction was made. I've experienced situations like this one where we used simple decision trees that performed worse than other methods simply because it was much more important that the stakeholder understand how we achieved at a final prediction and which variables were the most important ones.


## Inference

For social scientists, we can use machine learning methods for exploring hypothesis in the data. In particular, tree-based methods and regularized regressions can help us understand variables which are very good for prediction but that we weren't aware of. Moreover, it can help us understand the role of interactions from a more intuitive point of view through exploration.

## Prediction

If you're aim is the best predictive accuracy out there, then there's also evidence that some models seem to perform better than others. Tree based methods such as random forests and gradient boosting seem to continually perform the best in predictive competitions, together with more advanced models such as neural networks and support vector machines. For raw accuracy, there's no rule on which model to use. You might have a hunch depending on the distribution and exploration of your data but since these methods are quite complex, there's no single rule that states that one will perform better. We simply need to try many of them.

Having said this, we need to explore our data and understand it. This can help a lot in figuring out why some models work more than others.


## Prediction challenge

As part of the end of the course, we will have a prediction competition. This means you'll get to use all the methods we've discussed so far and compare your predictions to your fellow class mates. 

In the 2019 Summer Institute In Computational Social Science (SICSS), Mark Verhagen, Christopher Barrie, Arun Frey, Pablo Beytía, Arran Davis and me collected data on the number of people that visit the Wikipedia website of all counties in the United States. This data can be used to understand whether countries with different racial composition and poverty levels get more edits from the Wikipedia community. This can help assess whether there is a fundamental bias in Wikipedia contribution to richer counties.

We will use this data to predict the total number of edits through the history of each country in Wikipedia (variable `revisions`). We've matched this data with census-level indicators for each county for a total of 150 columns. Below is the codebook:

* `county_fips`: the county code
* `longitude/latitude`: the location of the county
* `population`: total population of county
* `density`: density of population
* `watchers`: number of wikipedia users who 'watch' the page
* `pageviews`: number of pageviews
* `pageviews_offset`: minimum number of pageviews as an offset
* `revisions`: total number of edits (from the creation of the website)
* `editors`: total number of editors (from the creation of the website)
* `secs_since_last_edit`: seconds since last edit
* `characters`: number of characters in the website
* `words`: number of words in the website
* `references`: number of references in the article
* `unique_references`: number of unique references in the article
* `sections`: number of sections in the wikipedia article
* `external_links`: number of external links
* `links_from_this_page`: number of hyperlinks used in this page
* `links_to_this_page`: number of hyperlinks that point to this page (from other wikipedia websites)
* `male_*_*`: these are the number of males within age groups
* `female_*_*`: these are the number of females within age groups
* `total_*`: These are the total population for different demographic groups
* `latino`: total count of latinos
* `latino_*`: total count of latinos from different races
* `no_schooling_completed`: total respondents with no schooling
* `nursery_school`: total respondents with only nursery school
* `kindergarten`: total respondents with kindergarten
* `grade_*`: These are the number of people the completed certain level of education
* `hs_diploma`: total respondents with high school diploma
* `ged`: total respondents with a GED diploma
* `less_than_1_year_college`: total respondents with less than one year of college
* `more_than_1_year_college`: total respondents with more than one year of college
* `associates_degree`: total respondents with associates degree
* `bachelors_degree`: total respondents with bachelors degree
* `masters_degree`: total respondents with masters degree
* `professional_degree`: total respondents with professional degree
* `doctorate_degree`: total respondents with doctorate degree
* `total_with_poverty_status`: total respondents with poverty status
* `income_below_poverty`: total respondents with income below povert levels
* `born_in_usa`: total respondents born in USA
* `foreign_born`: total respondents foreing born
* `speak_only_english`: total respondents who speak only english
* `speak_other_languages`: total respondents who speak other languages
* `count_*`: total number of respondents within age groups
* `percent_age_*`: percentage of people within age groups
* `percent_*`: percentage of people form different demogaphic groups. For example, whites, blacks, less than highschool, born in USA, etc..
* `internet_usage`: percentage of internet usage in county


For all of your analysis, use the `rmse` loss function, so that we can compare across participants.

Here are some ideas you can try in your analysis:

* Does it make sense to reduce the number of correlated variables into a few principal components?
* Do some counties cluster on very correlated variables? Is it fesiable to summarize some of these variables through predicting the cluster membership?
* Do we really need to use all variables?
* Does regularized regression or tree-based methods do better?

You can read the data with:


```r
wiki_dt <- read.csv("https://raw.githubusercontent.com/cimentadaj/ml_socsci/master/data/wikipedia_final.csv")
```

You have 45 minutes, start!
