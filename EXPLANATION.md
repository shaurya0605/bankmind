# EXPLANATION.md

> **Note:** Run `eda.py` on the real `bank-full.csv` first, then replace
> the bracketed placeholders below with your actual numbers/observations.
> The reviewers can tell if these weren't based on real output, so don't
> leave the placeholders in.

### 1. What percentage of customers in your dataset have `y = yes`? What does this imbalance mean for how you'd evaluate a model?

[X]% of customers subscribed (`y = yes`), meaning roughly [100-X]% did not.
This is a strongly imbalanced dataset. A model could predict "no" for
every single customer and still score around [100-X]% accuracy while
being completely useless for the business goal of finding likely
subscribers. So accuracy alone is a poor evaluation metric here —
precision, recall, and F1 (or looking at the confusion matrix directly)
matter much more, since they reveal how well the model actually
identifies the minority "yes" class.

### 2. Which job category had the highest subscription rate? Does this make sense to you intuitively?

The job category with the highest subscription rate was **[job name]**,
at [X]%. [Explain in 2-3 sentences whether this makes intuitive sense —
e.g. retirees/students often have more free time and fewer existing
financial commitments, which could make them more open to new
products, while categories like blue-collar workers might already be
financially stretched.]
