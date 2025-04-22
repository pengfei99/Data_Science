# Model evaluation

In this chapiter, we will learn how to evaluate a model.

## 1. Basic terms

### 1.1  Confusion Matrix and basic stats

A **confusion matrix** is a table that shows how well a `classification model` is performing. It compares 
the `actual labels` with the `predicted labels`.

| Actual label value     | Predicted Negative label	 | Predicted Positive label |
|------------------------|---------------------------|--------------------------|
| Actual Negative label	 | TN (True Negative)	       | FP (False Positive)      |
| Actual Positive label  | 	FN (False Negative)	     | TP (True Positive)       |

- **TP**: Correctly predicted positive 
- **TN**: Correctly predicted negative
- **FP**: Incorrectly predicted positive
- **FN**: Incorrectly predicted negative

For example, suppose we have a model that predicts if an email is `spam (1)` or `not spam (0)`. The model tests 10 
emails, and the below two lists represent `actual label(truth)` and `predicted label(model prediction)`:

```text
Actual:    [1, 0, 1, 1, 0, 0, 1, 0, 0, 1]
Predicted: [1, 0, 1, 0, 0, 1, 1, 0, 0, 1]
```

With the above results, we will have the below Confusion Matrix:

| Actual label value   | Predicted: 0 (Not spam)	 | Predicted : 1 (Spam) |
|----------------------|--------------------------|----------------------|
| Actual: 0 (Not spam) | 3 (TN)                   | 1 (FP)               |
| Actual: 1 (Spam)     | (1) FN                   | (5) TP               |

Based on the Confusion Matrix, we can calculate some basic stats:
- **Accuracy**:	How often the model is correct overall.
- **Precision**: When the model predicts positive, how often is it right?
- **Recall**: Of all the actual positives, how many did the model find?
- **F1 Score**: A balance between precision and recall.

## Choose metrics to evaluate a model

The appropriate metrics are important to evaluate a model. There are many factors that we need to consider, such as
the goal of the model(e.g. classification, regression, etc.), the dataset properties(balanced, imbalanced).

### For classification model

In general, we use confusion matrix and related stats such as `Accuracy`, `Precision`, `Recall` and `F1 score` to 
evaluate a classification model.


For balanced dataset, we can also use `ROC Curve` and `AUC value` .
For imbalanced dataset, we can use `Precision-Recall Curve` and `AUC value`





