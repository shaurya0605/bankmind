# EXPLANATION.md

> **Note:** Run `eda.py` and `train_model.py` on the real `bank-full.csv`
> first, then replace the bracketed placeholders below with your actual
> numbers/observations. The reviewers can tell if these weren't based
> on real output, so don't leave the placeholders in.

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

### 3. Which feature had the highest importance in your tree-based model? Why do you think that is?

The highest-importance feature in the Random Forest was **[feature name]**
(importance score: [X]). [Explain in 2-3 sentences why this makes sense
— e.g. if it's `duration` (call length), longer conversations likely
mean a more engaged customer; if it's `balance`, having more savings
may correlate with being a more attractive target for term deposits.]

### 4. Why is F1 a better metric than accuracy for this particular dataset?

Because the dataset is heavily imbalanced (only ~[X]% are "yes"), a
model can score high accuracy just by predicting "no" for almost
everyone, while completely failing at the actual business goal of
finding likely subscribers. F1 balances precision and recall, so it
penalizes a model that ignores the minority class — which is exactly
the class we care about here.

### 5. Pick one of your 5 sample predictions. Do you actually agree with the model's call, given that customer's features? Walk through your thinking.

[Pick one row from your printed sample predictions table. State the
customer's age, job, balance, housing/loan status, the model's
prediction, and probability. Then explain in 2-3 sentences whether you
agree with the call — e.g. "The model predicted 'yes' with 0.53
probability for a 71-year-old retired customer with a high balance
and no existing loans — I agree, since retirees with savings and no
debt obligations match the pattern the data shows for subscribers."
If you disagree, say why — e.g. the probability was close to 0.5,
meaning the model itself wasn't confident.]

### 6. Why did you scale features for Logistic Regression but not for Random Forest?

Logistic Regression is a linear model that computes a weighted sum of
the input features — if one feature (e.g. `balance`, which ranges into
the thousands) has a much larger numeric range than another (e.g.
`campaign`, which is a small integer), it can dominate the model's
weights purely due to its scale, not its actual predictive value.
Scaling (via `StandardScaler`) puts every feature on a comparable
range so the model learns from their actual relationships rather than
their units. Random Forest, on the other hand, makes decisions by
splitting on individual feature thresholds (e.g. "is balance > 5000?")
one feature at a time, so the relative scale between different
features never enters the calculation — it doesn't need scaling.

### 7. What real value would an LLM-powered `/explain` endpoint add on top of the raw prediction and probability?

A raw probability score like "0.53" tells a bank employee *that* the
model thinks this customer might subscribe, but not *why* — which
matters if a manager wants to sanity-check the model or a sales rep
wants context to actually use during the call. An LLM-powered
`/explain` endpoint could take the customer's features and the model's
top contributing factors (e.g. from `feature_importances_` or SHAP
values) and turn them into a plain-language reason like "this customer
has a high balance and no existing loans, which historically
correlates with subscribing." That's something a non-technical
stakeholder can act on directly, instead of having to interpret a bare
probability number.
