---
layout: post
title: "Logistic Regression on Titanic: a tidy notebook-to-blog walkthrough"
date: 2025-10-08 15:00:00 +1100
description: "Cleaning, feature engineering, and classification with pandas and scikit‑learn — reproduced from my Jupyter notebook"
tags: [python, pandas, scikit-learn, logistic-regression, data-science]
---

> This post is a cleaned-up version of my Jupyter notebook where I build a simple logistic regression model on the Titanic dataset. I’ve kept the narrative, code, and outputs-oriented structure so it reads like a reproducible walkthrough you can drop into your own project.

## Setup

```python
# libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
```

## Load & peek

```python
# Kaggle Titanic training data (assumes train.csv in the working directory)
data = pd.read_csv("train.csv")
data.head()
```
Typical columns include: `PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked`

## Make a safety copy

```python
data_original = data.copy()
```

## Impute missing Age (by Sex × Pclass)

Why not use `Survived` for imputation? Because it risks **data leakage** — leaking the target into features biases training and inflates metrics.

```python
# fill Age by group mean (Sex, Pclass)
data["Age"] = (
    data.groupby(["Sex","Pclass"])["Age"]
        .transform(lambda s: s.fillna(s.mean()))
)
```

(Optional) Compare distributions pre/post-imputation:

```python
fig, axes = plt.subplots(1, 2, figsize=(10,4), sharey=True)
sns.histplot(data_original["Age"], kde=True, ax=axes[0])
axes[0].set_title("Age (before)")
sns.histplot(data["Age"], kde=True, ax=axes[1])
axes[1].set_title("Age (after)")
plt.show()
```

### Q: What does `plt.subplots` do?  
It creates a figure and one or more axes (subplots) at once, returning `(fig, axes)`. Check `help(plt.subplots)` or the Matplotlib docs for details.

## One‑hot encode categoricals

Regression needs numeric inputs. Convert `Sex` and `Embarked` to indicator variables.

```python
data = pd.get_dummies(data, columns=["Sex","Embarked"], drop_first=True)
data.head()
```

**Q:** What did `pd.get_dummies` do?  
It creates 0/1 columns for each category so models can use them as numeric features.

## Train/test split

```python
# features for the first pass
X = data[["Pclass","Age","SibSp","Parch","Fare","Sex_male","Embarked_Q","Embarked_S"]]
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

## Fit logistic regression

`max_iter` raises the ceiling for the optimiser steps (helpful if the default doesn’t converge).

```python
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
```

Why logistic regression here? The target is **binary** (`Survived` = 0/1).

## Evaluate

```python
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
print(cm)
```

- **Confusion matrix** counts TP/TN/FP/FP and is standard for classifiers
- **Accuracy** = correct predictions / total

## Coefficients as a quick “importance” proxy

```python
importance = pd.DataFrame({
    "feature": X.columns,
    "coef": model.coef_[0]
}).sort_values("coef", ascending=False)

importance
```

Interpretation tip:
- Positive coef → higher values push towards class **1** (survived)
- Negative coef → higher values push towards class **0** (did not survive)

Optionally visualise:
```python
plt.figure(figsize=(6,4))
sns.barplot(x="coef", y="feature", data=importance)
plt.title("Logistic coefficients")
plt.show()
```

## Practical feature: FamilySize

Let’s engineer `FamilySize = SibSp + Parch` and re‑fit.

```python
data["FamilySize"] = data["SibSp"] + data["Parch"]

X = data[["Pclass","Age","FamilySize","Fare","Sex_male","Embarked_Q","Embarked_S"]]
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)

print(f"Accuracy with FamilySize: {acc:.4f}")
print(cm)
```

## Predict on Kaggle test set

**Golden rule:** apply **the same preprocessing** you used for training.

```python
test_data = pd.read_csv("test.csv")

# impute Age the same way
test_data["Age"] = (
    test_data.groupby(["Sex","Pclass"])["Age"]
             .transform(lambda s: s.fillna(s.mean()))
)

# one missing Fare — fill with group mean (Sex × Pclass)
if test_data["Fare"].isna().any():
    test_data["Fare"] = (
        test_data.groupby(["Sex","Pclass"])["Fare"]
                 .transform(lambda s: s.fillna(s.mean()))
    )

# one‑hot to match training
test_proc = pd.get_dummies(test_data, columns=["Sex","Embarked"], drop_first=True)

# ensure same columns/order
test_proc = test_proc.reindex(columns=X.columns, fill_value=0)

# predict
test_preds = model.predict(test_proc)

# attach for inspection / submission formatting
out = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": test_preds
})
out.head()
```

You can save for Kaggle submission with:
```python
out.to_csv("submission.csv", index=False)
```

## Notes & gotchas

- Don’t impute using the target (`Survived`) → **data leakage**
- Keep preprocessing **identical** between train and test
- If logistic regression warns about convergence, consider:
  - increasing `max_iter`
  - standardising features
  - trying a different `solver`

---

### Repro tips
- Put preprocessing in functions/pipelines to guarantee identical steps
- Lock `random_state` for reproducibility when teaching/demonstrating
- Track metrics beyond accuracy (precision/recall, ROC‑AUC) for imbalanced data

*Thanks for reading — if you reuse this, please link back or drop a comment with your tweaks!*

