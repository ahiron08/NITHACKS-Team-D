# EDITH Model Comparison Report

## Performance Summary

| Model            |   Val_Accuracy |   Val_Precision |   Val_Recall |   Val_F1 |   Val_ROC_AUC |   Val_PR_AUC |   Test_Accuracy |   Test_Precision |   Test_Recall |   Test_F1 |   Test_ROC_AUC |   Test_PR_AUC |   Test_FPR |   Test_FNR |
|:-----------------|---------------:|----------------:|-------------:|---------:|--------------:|-------------:|----------------:|-----------------:|--------------:|----------:|---------------:|--------------:|-----------:|-----------:|
| Isolation Forest |         0.434  |          0.2101 |       0.9973 |   0.347  |        0.9033 |       0.7701 |          0.5375 |           0.3875 |        0.994  |    0.5576 |         0.9179 |        0.8676 |     0.6519 |     0.006  |
| Xgboost          |         0.9988 |          0.9947 |       0.9973 |   0.996  |        0.9996 |       0.9988 |          0.9978 |           0.9991 |        0.9932 |    0.9962 |         0.9994 |        0.9993 |     0.0004 |     0.0068 |
| Random Forest    |         0.9992 |          0.9947 |       1      |   0.9974 |        1      |       1      |          0.997  |           0.994  |        0.9957 |    0.9949 |         0.9995 |        0.9995 |     0.0025 |     0.0043 |
| Autoencoder      |         0.9448 |          0.8187 |       0.8143 |   0.8165 |        0.9597 |       0.897  |          0.931  |           0.9149 |        0.8431 |    0.8776 |         0.9654 |        0.9451 |     0.0325 |     0.1569 |
| Ensemble         |         0.9996 |          1      |       0.9973 |   0.9987 |        1      |       1      |          0.9972 |           1      |        0.9906 |    0.9953 |         0.9997 |        0.9995 |     0      |     0.0094 |

## Analysis

- **Best model by test F1**: Xgboost (F1=0.9962)

### Target Achievement (Test Set)
Targets: 90% accuracy, 90% precision, 90% recall, 90% F1

- Isolation Forest: ACCURACY: FAIL (0.537), PRECISION: FAIL (0.388), RECALL: PASS (0.994), F1: FAIL (0.558)
- Xgboost: ACCURACY: PASS (0.998), PRECISION: PASS (0.999), RECALL: PASS (0.993), F1: PASS (0.996)
- Random Forest: ACCURACY: PASS (0.997), PRECISION: PASS (0.994), RECALL: PASS (0.996), F1: PASS (0.995)
- Autoencoder: ACCURACY: PASS (0.931), PRECISION: PASS (0.915), RECALL: FAIL (0.843), F1: FAIL (0.878)
- Ensemble: ACCURACY: PASS (0.997), PRECISION: PASS (1.000), RECALL: PASS (0.991), F1: PASS (0.995)
