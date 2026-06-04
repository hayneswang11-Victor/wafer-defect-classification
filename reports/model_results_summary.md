# Model Results Summary

This summary records the current synthetic prototype metrics. These results are
based on generated data only and must not be interpreted as real semiconductor
manufacturing performance.

## Test Metrics

| Map Type | Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
| wafer | logistic_regression | 0.967 | 0.969 | 0.967 | 0.967 | 0.967 |
| wafer | random_forest | 0.978 | 0.979 | 0.978 | 0.978 | 0.978 |
| dram | logistic_regression | 0.983 | 0.984 | 0.983 | 0.983 | 0.983 |
| dram | random_forest | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Artifact Locations

Metrics JSON files:

- `reports/metrics/wafer_logistic_regression_metrics.json`
- `reports/metrics/wafer_random_forest_metrics.json`
- `reports/metrics/dram_logistic_regression_metrics.json`
- `reports/metrics/dram_random_forest_metrics.json`

Model artifacts:

- `models/wafer_logistic_regression.joblib`
- `models/wafer_random_forest.joblib`
- `models/dram_logistic_regression.joblib`
- `models/dram_random_forest.joblib`

Figures:

- `reports/figures/wafer_logistic_regression_confusion_matrix.png`
- `reports/figures/wafer_logistic_regression_coefficients.png`
- `reports/figures/wafer_random_forest_confusion_matrix.png`
- `reports/figures/wafer_random_forest_feature_importance.png`
- `reports/figures/dram_logistic_regression_confusion_matrix.png`
- `reports/figures/dram_logistic_regression_coefficients.png`
- `reports/figures/dram_random_forest_confusion_matrix.png`
- `reports/figures/dram_random_forest_feature_importance.png`

## Interpretation Note

The DRAM random forest reaches `1.000` because the current synthetic defect
classes are cleanly separable. This is a useful prototype sanity check, but it
does not imply real production performance. Real deployment would require
authorized production or test data, engineering labels, drift validation, and
process/test metadata.
