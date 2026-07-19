# Evaluation Results

Run on 2026-07-19, training `RandomForestClassifier` on `fraudTest.csv` via `train_fraud_model.py`.
Full raw log: [`training_log.txt`](training_log.txt). Confusion matrix plot: [`confusion_matrix.png`](confusion_matrix.png).

## Data

| | |
|---|---|
| Total records | 555,719 |
| Non-fraud (`is_fraud=0`) | 553,574 |
| Fraud (`is_fraud=1`) | 2,145 (0.39%) |
| Train / test split | 444,803 / 110,916 (80/20, seed=42) |
| Class weight applied to fraud class | 258.08 |

## Best model (selected via `TrainValidationSplit`, optimizing AUC-ROC)

| Hyperparameter | Value |
|---|---|
| numTrees | 50 |
| maxDepth | 8 |
| Training time | 385.74s (6.43 min) |

## Overall test-set metrics

| Metric | Value |
|---|---|
| AUC-ROC | 0.9656 |
| AUC-PR | 0.4073 |
| Accuracy | 0.9816 |
| F1-Score (weighted) | 0.9878 |
| Weighted Precision | 0.9955 |
| Weighted Recall | 0.9816 |

## Confusion matrix

| | Predicted Not Fraud | Predicted Fraud |
|---|---|---|
| **Actual Not Fraud** | 108,584 (TN) | 1,909 (FP) |
| **Actual Fraud** | 130 (FN) | 293 (TP) |

## Fraud class (class 1) metrics

| Metric | Value |
|---|---|
| Precision | 0.1331 |
| Recall | 0.6927 |
| F1-Score | 0.2232 |

## Interpretation

Overall accuracy/weighted metrics look strong, but they're dominated by the 99.6% non-fraud class and are not meaningful here. The metrics that matter for fraud detection are the **fraud-class** numbers:

- **Recall 0.69** — the model catches ~69% of actual fraud (293 of 423 fraud cases in the test set), missing 130.
- **Precision 0.13** — of every ~7.5 transactions flagged as fraud, only 1 is actually fraudulent; the other ~6.5 are false alarms (1,909 false positives) that would need manual review.
- **AUC-PR 0.41** vs. AUC-ROC 0.97 — AUC-ROC looks great but is known to be optimistic on highly imbalanced data; AUC-PR is the more honest signal here and shows there's real room to improve precision/recall trade-off (e.g. threshold tuning, better features, gradient-boosted trees, or under/oversampling instead of/alongside class weighting).
