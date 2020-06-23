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

