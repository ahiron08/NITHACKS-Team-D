# EDITH Dataset Audit Report
Generated: 2026-11-09T14:30:30.271683
Random seed: 42

## 1. Dataset Dimensions
- Training set shape: 10000 rows x 30 columns
- Test set shape: 4000 rows x 30 columns
- Feature columns (excluding target): 29

## 2. Feature Data Types
### Training set
- app_id: str (nulls: 0)
- session_id: str (nulls: 0)
- hour_of_day: int64 (nulls: 0)
- day_of_week: int64 (nulls: 0)
- requests_per_min: float64 (nulls: 0)
- denied_requests: int64 (nulls: 0)
- sensitive_access_denied: int64 (nulls: 0)
- request_rate_deviation: float64 (nulls: 0)
- microphone_access_count: int64 (nulls: 0)
- location_access_count: int64 (nulls: 0)
- payment_access_count: int64 (nulls: 0)
- camera_access_count: int64 (nulls: 0)
- file_access_count: int64 (nulls: 0)
- file_modification_count: int64 (nulls: 0)
- background_time_ratio: float64 (nulls: 0)
- background_requests: int64 (nulls: 0)
- background_mic_access: int64 (nulls: 0)
- background_location_access: int64 (nulls: 0)
- background_camera_access: int64 (nulls: 0)
- permission_count: int64 (nulls: 0)
- sensitive_permission_count: int64 (nulls: 0)
- permission_changes: int64 (nulls: 0)
- network_connections: int64 (nulls: 0)
- data_transferred_mb: float64 (nulls: 0)
- transaction_count: int64 (nulls: 0)
- failed_transaction_count: int64 (nulls: 0)
- unconfirmed_transaction_count: int64 (nulls: 0)
- avg_transaction_amount: float64 (nulls: 0)
- total_transaction_amount: float64 (nulls: 0)
- is_anomaly: int64 (nulls: 0)

### Test set
- app_id: str (nulls: 0)
- session_id: str (nulls: 0)
- hour_of_day: int64 (nulls: 0)
- day_of_week: int64 (nulls: 0)
- requests_per_min: float64 (nulls: 0)
- denied_requests: int64 (nulls: 0)
- sensitive_access_denied: int64 (nulls: 0)
- request_rate_deviation: float64 (nulls: 0)
- microphone_access_count: int64 (nulls: 0)
- location_access_count: int64 (nulls: 0)
- payment_access_count: int64 (nulls: 0)
- camera_access_count: int64 (nulls: 0)
- file_access_count: int64 (nulls: 0)
- file_modification_count: int64 (nulls: 0)
- background_time_ratio: float64 (nulls: 0)
- background_requests: int64 (nulls: 0)
- background_mic_access: int64 (nulls: 0)
- background_location_access: int64 (nulls: 0)
- background_camera_access: int64 (nulls: 0)
- permission_count: int64 (nulls: 0)
- sensitive_permission_count: int64 (nulls: 0)
- permission_changes: int64 (nulls: 0)
- network_connections: int64 (nulls: 0)
- data_transferred_mb: float64 (nulls: 0)
- transaction_count: int64 (nulls: 0)
- failed_transaction_count: int64 (nulls: 0)
- unconfirmed_transaction_count: int64 (nulls: 0)
- avg_transaction_amount: float64 (nulls: 0)
- total_transaction_amount: float64 (nulls: 0)
- is_anomaly: int64 (nulls: 0)

## 3. Missing Values
|                               |   train_missing |   test_missing |   train_pct |   test_pct |
|:------------------------------|----------------:|---------------:|------------:|-----------:|
| app_id                        |               0 |              0 |           0 |          0 |
| session_id                    |               0 |              0 |           0 |          0 |
| hour_of_day                   |               0 |              0 |           0 |          0 |
| day_of_week                   |               0 |              0 |           0 |          0 |
| requests_per_min              |               0 |              0 |           0 |          0 |
| denied_requests               |               0 |              0 |           0 |          0 |
| sensitive_access_denied       |               0 |              0 |           0 |          0 |
| request_rate_deviation        |               0 |              0 |           0 |          0 |
| microphone_access_count       |               0 |              0 |           0 |          0 |
| location_access_count         |               0 |              0 |           0 |          0 |
| payment_access_count          |               0 |              0 |           0 |          0 |
| camera_access_count           |               0 |              0 |           0 |          0 |
| file_access_count             |               0 |              0 |           0 |          0 |
| file_modification_count       |               0 |              0 |           0 |          0 |
| background_time_ratio         |               0 |              0 |           0 |          0 |
| background_requests           |               0 |              0 |           0 |          0 |
| background_mic_access         |               0 |              0 |           0 |          0 |
| background_location_access    |               0 |              0 |           0 |          0 |
| background_camera_access      |               0 |              0 |           0 |          0 |
| permission_count              |               0 |              0 |           0 |          0 |
| sensitive_permission_count    |               0 |              0 |           0 |          0 |
| permission_changes            |               0 |              0 |           0 |          0 |
| network_connections           |               0 |              0 |           0 |          0 |
| data_transferred_mb           |               0 |              0 |           0 |          0 |
| transaction_count             |               0 |              0 |           0 |          0 |
| failed_transaction_count      |               0 |              0 |           0 |          0 |
| unconfirmed_transaction_count |               0 |              0 |           0 |          0 |
| avg_transaction_amount        |               0 |              0 |           0 |          0 |
| total_transaction_amount      |               0 |              0 |           0 |          0 |
| is_anomaly                    |               0 |              0 |           0 |          0 |

## 4. Duplicate Records
- Exact duplicate rows (train): 0
- Exact duplicate rows (test): 0
- Duplicate feature rows excluding target (train): 0
- Duplicate feature rows excluding target (test): 0
- Duplicate (app_id, session_id) pairs (train): 0
- Duplicate (app_id, session_id) pairs (test): 0

## 5. Infinite and Invalid Numerical Values
- hour_of_day: train inf=0, test inf=0, train negative=0, test negative=0
- day_of_week: train inf=0, test inf=0, train negative=0, test negative=0
- requests_per_min: train inf=0, test inf=0, train negative=0, test negative=0
- denied_requests: train inf=0, test inf=0, train negative=0, test negative=0
- sensitive_access_denied: train inf=0, test inf=0, train negative=0, test negative=0
- request_rate_deviation: train inf=0, test inf=0, train negative=0, test negative=0
- microphone_access_count: train inf=0, test inf=0, train negative=0, test negative=0
- location_access_count: train inf=0, test inf=0, train negative=0, test negative=0
- payment_access_count: train inf=0, test inf=0, train negative=0, test negative=0
- camera_access_count: train inf=0, test inf=0, train negative=0, test negative=0
- file_access_count: train inf=0, test inf=0, train negative=0, test negative=0
- file_modification_count: train inf=0, test inf=0, train negative=0, test negative=0
- background_time_ratio: train inf=0, test inf=0, train negative=0, test negative=0
- background_requests: train inf=0, test inf=0, train negative=0, test negative=0
- background_mic_access: train inf=0, test inf=0, train negative=0, test negative=0
- background_location_access: train inf=0, test inf=0, train negative=0, test negative=0
- background_camera_access: train inf=0, test inf=0, train negative=0, test negative=0
- permission_count: train inf=0, test inf=0, train negative=0, test negative=0
- sensitive_permission_count: train inf=0, test inf=0, train negative=0, test negative=0
- permission_changes: train inf=0, test inf=0, train negative=0, test negative=0
- network_connections: train inf=0, test inf=0, train negative=0, test negative=0
- data_transferred_mb: train inf=0, test inf=0, train negative=0, test negative=0
- transaction_count: train inf=0, test inf=0, train negative=0, test negative=0
- failed_transaction_count: train inf=0, test inf=0, train negative=0, test negative=0
- unconfirmed_transaction_count: train inf=0, test inf=0, train negative=0, test negative=0
- avg_transaction_amount: train inf=0, test inf=0, train negative=0, test negative=0
- total_transaction_amount: train inf=0, test inf=0, train negative=0, test negative=0

## 6. Class Distribution
### Training set
- is_anomaly=0: 8491 (84.91%)
- is_anomaly=1: 1509 (15.09%)
### Test set
- is_anomaly=0: 2827 (70.67%)
- is_anomaly=1: 1173 (29.33%)

## 7. Feature Distributions (Numerical Summary)
|       |   hour_of_day |   day_of_week |   requests_per_min |   denied_requests |   sensitive_access_denied |   request_rate_deviation |   microphone_access_count |   location_access_count |   payment_access_count |   camera_access_count |   file_access_count |   file_modification_count |   background_time_ratio |   background_requests |   background_mic_access |   background_location_access |   background_camera_access |   permission_count |   sensitive_permission_count |   permission_changes |   network_connections |   data_transferred_mb |   transaction_count |   failed_transaction_count |   unconfirmed_transaction_count |   avg_transaction_amount |   total_transaction_amount |
|:------|--------------:|--------------:|-------------------:|------------------:|--------------------------:|-------------------------:|--------------------------:|------------------------:|-----------------------:|----------------------:|--------------------:|--------------------------:|------------------------:|----------------------:|------------------------:|-----------------------------:|---------------------------:|-------------------:|-----------------------------:|---------------------:|----------------------:|----------------------:|--------------------:|---------------------------:|--------------------------------:|-------------------------:|---------------------------:|
| count |    10000      |   10000       |         10000      |       10000       |               10000       |             10000        |                10000      |             10000       |              10000     |           10000       |          10000      |                10000      |            10000        |           10000       |              10000      |                  10000       |                10000       |        10000       |                   10000      |          10000       |             10000     |             10000     |          10000      |                10000       |                     10000       |               10000      |                 10000      |
| mean  |       11.4993 |       2.976   |            55.9312 |           2.9741  |                   0.4987  |                 0.44424  |                    7.4163 |                 3.8138  |                  9.505 |               4.1519  |             58.9035 |                   11.7781 |                0.454715 |               1.9205  |                  2.1032 |                      0.7445  |                    1.0654  |           14.0911  |                       4.4191 |              2.5333  |               106.722 |              1268.64  |             17.0097 |                    0.5933  |                         1.0335  |                  94.5765 |                  4028.07   |
| std   |        6.9402 |       2.01326 |           100.526  |           9.67323 |                   1.89709 |                 0.420596 |                   14.7062 |                 4.43377 |                 52.883 |               6.60409 |             56.0678 |                   19.9125 |                0.251661 |               3.18829 |                 17.078  |                      4.76366 |                    7.06617 |            7.70741 |                       4.0729 |              8.74646 |               136.506 |              3382.99  |             58.0344 |                    3.09442 |                         3.36254 |                 138.818  |                 12530.7    |
| min   |        0      |       0       |             2.03   |           0       |                   0       |                 0.0002   |                    0      |                 0       |                  0     |               0       |              1      |                    0      |                0.05     |               0       |                  0      |                      0       |                    0       |            3       |                       0      |              0       |                 2     |                 5.1   |              0      |                    0       |                         0       |                   0      |                     0      |
| 25%   |        5      |       1       |            16.2125 |           0       |                   0       |                 0.1884   |                    0      |                 1       |                  0     |               0       |             21      |                    2      |                0.199275 |               0       |                  0      |                      0       |                    0       |            8       |                       2      |              0       |                24     |               172.627 |              1      |                    0       |                         0       |                  17.5175 |                     8.8075 |
| 50%   |       12      |       3       |            33.645  |           1       |                   0       |                 0.3818   |                    1      |                 2       |                  0     |               0       |             42      |                    6      |                0.4688   |               1       |                  0      |                      0       |                    0       |           13       |                       4      |              1       |                52     |               476.46  |              2      |                    0       |                         0       |                  43.01   |                    78.945  |
| 75%   |       17      |       5       |            69.155  |           3       |                   1       |                 0.571125 |                    7      |                 6       |                  4     |               7       |             80.25   |                   14      |                0.648825 |               3       |                  1      |                      0       |                    1       |           18       |                       6      |              3       |               137     |              1364.32  |              8      |                    1       |                         1       |                  90.8775 |                   499.585  |
| max   |       23      |       6       |          2893.84   |         279       |                  53       |                 3.7482   |                  138      |                36       |               2037     |              53       |            534      |                  428      |                1        |              33       |                528      |                    124       |                  166       |           65       |                      65      |            258       |              1300     |            110031     |           1618      |                  102       |                       118       |                1393.08   |                214811      |

## 8. Outlier Analysis (IQR Method)
| Feature | Train Outliers | Train Outlier % | Test Outliers | Test Outlier % |
|---------|---------------|----------------|---------------|----------------|
| hour_of_day | 0 | 0.00% | 0 | 0.00% |
| day_of_week | 0 | 0.00% | 0 | 0.00% |
| requests_per_min | 381 | 3.81% | 303 | 7.58% |
| denied_requests | 739 | 7.39% | 452 | 11.30% |
| sensitive_access_denied | 251 | 2.51% | 221 | 5.53% |
| request_rate_deviation | 489 | 4.89% | 417 | 10.42% |
| microphone_access_count | 1468 | 14.68% | 629 | 15.72% |
| location_access_count | 365 | 3.65% | 241 | 6.02% |
| payment_access_count | 1861 | 18.61% | 812 | 20.30% |
| camera_access_count | 454 | 4.54% | 275 | 6.88% |
| file_access_count | 543 | 5.43% | 325 | 8.12% |
| file_modification_count | 923 | 9.23% | 484 | 12.10% |
| background_time_ratio | 0 | 0.00% | 0 | 0.00% |
| background_requests | 566 | 5.66% | 332 | 8.30% |
| background_mic_access | 1472 | 14.72% | 698 | 17.45% |
| background_location_access | 1791 | 17.91% | 918 | 22.95% |
| background_camera_access | 553 | 5.53% | 392 | 9.80% |
| permission_count | 189 | 1.89% | 191 | 4.78% |
| sensitive_permission_count | 256 | 2.56% | 240 | 6.00% |
| permission_changes | 335 | 3.35% | 302 | 7.55% |
| network_connections | 934 | 9.34% | 480 | 12.00% |
| data_transferred_mb | 1020 | 10.20% | 515 | 12.88% |
| transaction_count | 1954 | 19.54% | 847 | 21.18% |
| failed_transaction_count | 255 | 2.55% | 229 | 5.73% |
| unconfirmed_transaction_count | 1111 | 11.11% | 605 | 15.12% |
| avg_transaction_amount | 1429 | 14.29% | 614 | 15.35% |
| total_transaction_amount | 2096 | 20.96% | 900 | 22.50% |

## 9. Constant and Near-Constant Features

## 10. Feature Correlation Matrix
|                               |   hour_of_day |   day_of_week |   requests_per_min |   denied_requests |   sensitive_access_denied |   request_rate_deviation |   microphone_access_count |   location_access_count |   payment_access_count |   camera_access_count |   file_access_count |   file_modification_count |   background_time_ratio |   background_requests |   background_mic_access |   background_location_access |   background_camera_access |   permission_count |   sensitive_permission_count |   permission_changes |   network_connections |   data_transferred_mb |   transaction_count |   failed_transaction_count |   unconfirmed_transaction_count |   avg_transaction_amount |   total_transaction_amount |
|:------------------------------|--------------:|--------------:|-------------------:|------------------:|--------------------------:|-------------------------:|--------------------------:|------------------------:|-----------------------:|----------------------:|--------------------:|--------------------------:|------------------------:|----------------------:|------------------------:|-----------------------------:|---------------------------:|-------------------:|-----------------------------:|---------------------:|----------------------:|----------------------:|--------------------:|---------------------------:|--------------------------------:|-------------------------:|---------------------------:|
| hour_of_day                   |         1     |         0.025 |             -0.003 |             0.007 |                     0.006 |                   -0.008 |                    -0.004 |                   0.008 |                  0.021 |                 0.005 |               0.003 |                    -0.001 |                   0.006 |                -0.005 |                  -0.007 |                       -0.017 |                      0.002 |              0.01  |                        0.005 |                0.006 |                 0.012 |                 0.022 |               0.022 |                      0.018 |                           0.019 |                    0.005 |                      0.009 |
| day_of_week                   |         0.025 |         1     |              0.011 |            -0.005 |                     0.01  |                    0.007 |                    -0.006 |                   0.013 |                 -0.004 |                -0.006 |               0     |                     0.019 |                  -0.009 |                 0.007 |                   0.011 |                        0.005 |                      0.007 |             -0.002 |                        0.009 |                0.016 |                 0.001 |                 0.009 |               0.002 |                      0.011 |                           0.009 |                    0.008 |                     -0.004 |
| requests_per_min              |        -0.003 |         0.011 |              1     |             0.159 |                     0.164 |                    0.317 |                     0.21  |                   0.335 |                  0.009 |                 0.328 |               0.09  |                    -0.009 |                   0.16  |                 0.614 |                   0.065 |                        0.099 |                      0.097 |              0.333 |                        0.297 |                0.135 |                 0.383 |                 0.194 |               0.012 |                      0.062 |                           0.057 |                    0.04  |                      0.021 |
| denied_requests               |         0.007 |        -0.005 |              0.159 |             1     |                     0.621 |                    0.149 |                     0.086 |                   0.25  |                 -0.018 |                 0.206 |               0.061 |                    -0.02  |                   0.154 |                 0.328 |                   0.022 |                        0.075 |                      0.061 |              0.22  |                        0.205 |                0.109 |                 0.296 |                 0.151 |              -0.028 |                      0.029 |                           0.042 |                   -0.024 |                     -0.035 |
| sensitive_access_denied       |         0.006 |         0.01  |              0.164 |             0.621 |                     1     |                    0.131 |                     0.121 |                   0.244 |                 -0.014 |                 0.234 |               0.046 |                    -0.028 |                   0.17  |                 0.307 |                   0.046 |                        0.084 |                      0.092 |              0.245 |                        0.231 |                0.114 |                 0.331 |                 0.183 |              -0.026 |                      0.044 |                           0.048 |                   -0.031 |                     -0.043 |
| request_rate_deviation        |        -0.008 |         0.007 |              0.317 |             0.149 |                     0.131 |                    1     |                     0.042 |                   0.103 |                  0.071 |                 0.043 |               0.111 |                     0.159 |                   0.114 |                 0.305 |                   0.093 |                        0.132 |                      0.092 |              0.125 |                        0.156 |                0.097 |                 0.065 |                 0.101 |               0.069 |                      0.093 |                           0.096 |                    0.069 |                      0.079 |
| microphone_access_count       |        -0.004 |        -0.006 |              0.21  |             0.086 |                     0.121 |                    0.042 |                     1     |                  -0.015 |                 -0.079 |                 0.597 |              -0.179 |                    -0.189 |                   0.196 |                 0.311 |                   0.398 |                        0.043 |                      0.206 |              0.13  |                        0.093 |                0.139 |                 0.186 |                 0.124 |              -0.12  |                     -0.034 |                          -0.099 |                   -0.199 |                     -0.153 |
| location_access_count         |         0.008 |         0.013 |              0.335 |             0.25  |                     0.244 |                    0.103 |                    -0.015 |                   1     |                  0.15  |                 0.255 |               0.299 |                     0.154 |                   0.068 |                 0.459 |                   0.025 |                        0.343 |                      0.128 |              0.537 |                        0.535 |                0.236 |                 0.53  |                 0.304 |               0.204 |                      0.192 |                           0.257 |                    0.264 |                      0.201 |
| payment_access_count          |         0.021 |        -0.004 |              0.009 |            -0.018 |                    -0.014 |                    0.071 |                    -0.079 |                   0.15  |                  1     |                -0.079 |               0.301 |                     0.226 |                  -0.176 |                -0.052 |                  -0.019 |                        0.01  |                     -0.019 |              0.171 |                        0.119 |               -0.011 |                -0.005 |                -0.007 |               0.626 |                      0.422 |                           0.437 |                    0.307 |                      0.274 |
| camera_access_count           |         0.005 |        -0.006 |              0.328 |             0.206 |                     0.234 |                    0.043 |                     0.597 |                   0.255 |                 -0.079 |                 1     |              -0.128 |                    -0.198 |                   0.315 |                 0.539 |                   0.199 |                        0.131 |                      0.367 |              0.333 |                        0.301 |                0.231 |                 0.479 |                 0.307 |              -0.128 |                      0.001 |                          -0.045 |                   -0.208 |                     -0.182 |
| file_access_count             |         0.003 |         0     |              0.09  |             0.061 |                     0.046 |                    0.111 |                    -0.179 |                   0.299 |                  0.301 |                -0.128 |               1     |                     0.522 |                  -0.435 |                -0.034 |                  -0.005 |                        0.116 |                      0.026 |              0.372 |                        0.315 |                0.065 |                 0.063 |                 0.074 |               0.4   |                      0.233 |                           0.313 |                    0.553 |                      0.446 |
| file_modification_count       |        -0.001 |         0.019 |             -0.009 |            -0.02  |                    -0.028 |                    0.159 |                    -0.189 |                   0.154 |                  0.226 |                -0.198 |               0.522 |                     1     |                  -0.358 |                -0.135 |                  -0.042 |                        0.015 |                     -0.033 |              0.229 |                        0.141 |               -0.025 |                -0.079 |                 0.135 |               0.318 |                      0.183 |                           0.247 |                    0.498 |                      0.453 |
| background_time_ratio         |         0.006 |        -0.009 |              0.16  |             0.154 |                     0.17  |                    0.114 |                     0.196 |                   0.068 |                 -0.176 |                 0.315 |              -0.435 |                    -0.358 |                   1     |                 0.472 |                   0.1   |                        0.083 |                      0.129 |             -0.088 |                        0.035 |                0.142 |                 0.262 |                 0.154 |              -0.278 |                     -0.081 |                          -0.144 |                   -0.516 |                     -0.373 |
| background_requests           |        -0.005 |         0.007 |              0.614 |             0.328 |                     0.307 |                    0.305 |                     0.311 |                   0.459 |                 -0.052 |                 0.539 |              -0.034 |                    -0.135 |                   0.472 |                 1     |                   0.153 |                        0.196 |                      0.219 |              0.421 |                        0.444 |                0.287 |                 0.609 |                 0.376 |              -0.089 |                      0.068 |                           0.042 |                   -0.156 |                     -0.155 |
| background_mic_access         |        -0.007 |         0.011 |              0.065 |             0.022 |                     0.046 |                    0.093 |                     0.398 |                   0.025 |                 -0.019 |                 0.199 |              -0.005 |                    -0.042 |                   0.1   |                 0.153 |                   1     |                        0.255 |                      0.54  |              0.077 |                        0.031 |                0.036 |                 0.07  |                 0.032 |              -0.029 |                     -0.005 |                          -0.022 |                   -0.042 |                     -0.036 |
| background_location_access    |        -0.017 |         0.005 |              0.099 |             0.075 |                     0.084 |                    0.132 |                     0.043 |                   0.343 |                  0.01  |                 0.131 |               0.116 |                     0.015 |                   0.083 |                 0.196 |                   0.255 |                        1     |                      0.466 |              0.194 |                        0.145 |                0.07  |                 0.192 |                 0.101 |               0.022 |                      0.038 |                           0.053 |                    0.052 |                      0.042 |
| background_camera_access      |         0.002 |         0.007 |              0.097 |             0.061 |                     0.092 |                    0.092 |                     0.206 |                   0.128 |                 -0.019 |                 0.367 |               0.026 |                    -0.033 |                   0.129 |                 0.219 |                   0.54  |                        0.466 |                      1     |              0.154 |                        0.105 |                0.07  |                 0.192 |                 0.082 |              -0.028 |                      0.004 |                          -0.007 |                   -0.033 |                     -0.041 |
| permission_count              |         0.01  |        -0.002 |              0.333 |             0.22  |                     0.245 |                    0.125 |                     0.13  |                   0.537 |                  0.171 |                 0.333 |               0.372 |                     0.229 |                  -0.088 |                 0.421 |                   0.077 |                        0.194 |                      0.154 |              1     |                        0.595 |                0.239 |                 0.491 |                 0.301 |               0.219 |                      0.18  |                           0.261 |                    0.319 |                      0.229 |
| sensitive_permission_count    |         0.005 |         0.009 |              0.297 |             0.205 |                     0.231 |                    0.156 |                     0.093 |                   0.535 |                  0.119 |                 0.301 |               0.315 |                     0.141 |                   0.035 |                 0.444 |                   0.031 |                        0.145 |                      0.105 |              0.595 |                        1     |                0.52  |                 0.513 |                 0.27  |               0.142 |                      0.154 |                           0.185 |                    0.24  |                      0.177 |
| permission_changes            |         0.006 |         0.016 |              0.135 |             0.109 |                     0.114 |                    0.097 |                     0.139 |                   0.236 |                 -0.011 |                 0.231 |               0.065 |                    -0.025 |                   0.142 |                 0.287 |                   0.036 |                        0.07  |                      0.07  |              0.239 |                        0.52  |                1     |                 0.293 |                 0.142 |              -0.023 |                      0.038 |                           0.039 |                   -0.013 |                     -0.033 |
| network_connections           |         0.012 |         0.001 |              0.383 |             0.296 |                     0.331 |                    0.065 |                     0.186 |                   0.53  |                 -0.005 |                 0.479 |               0.063 |                    -0.079 |                   0.262 |                 0.609 |                   0.07  |                        0.192 |                      0.192 |              0.491 |                        0.513 |                0.293 |                 1     |                 0.381 |              -0.024 |                      0.095 |                           0.092 |                   -0.049 |                     -0.072 |
| data_transferred_mb           |         0.022 |         0.009 |              0.194 |             0.151 |                     0.183 |                    0.101 |                     0.124 |                   0.304 |                 -0.007 |                 0.307 |               0.074 |                     0.135 |                   0.154 |                 0.376 |                   0.032 |                        0.101 |                      0.082 |              0.301 |                        0.27  |                0.142 |                 0.381 |                 1     |              -0.007 |                      0.049 |                           0.048 |                   -0.006 |                     -0.016 |
| transaction_count             |         0.022 |         0.002 |              0.012 |            -0.028 |                    -0.026 |                    0.069 |                    -0.12  |                   0.204 |                  0.626 |                -0.128 |               0.4   |                     0.318 |                  -0.278 |                -0.089 |                  -0.029 |                        0.022 |                     -0.028 |              0.219 |                        0.142 |               -0.023 |                -0.024 |                -0.007 |               1     |                      0.513 |                           0.547 |                    0.426 |                      0.51  |
| failed_transaction_count      |         0.018 |         0.011 |              0.062 |             0.029 |                     0.044 |                    0.093 |                    -0.034 |                   0.192 |                  0.422 |                 0.001 |               0.233 |                     0.183 |                  -0.081 |                 0.068 |                  -0.005 |                        0.038 |                      0.004 |              0.18  |                        0.154 |                0.038 |                 0.095 |                 0.049 |               0.513 |                      1     |                           0.5   |                    0.25  |                      0.201 |
| unconfirmed_transaction_count |         0.019 |         0.009 |              0.057 |             0.042 |                     0.048 |                    0.096 |                    -0.099 |                   0.257 |                  0.437 |                -0.045 |               0.313 |                     0.247 |                  -0.144 |                 0.042 |                  -0.022 |                        0.053 |                     -0.007 |              0.261 |                        0.185 |                0.039 |                 0.092 |                 0.048 |               0.547 |                      0.5   |                           1     |                    0.331 |                      0.284 |
| avg_transaction_amount        |         0.005 |         0.008 |              0.04  |            -0.024 |                    -0.031 |                    0.069 |                    -0.199 |                   0.264 |                  0.307 |                -0.208 |               0.553 |                     0.498 |                  -0.516 |                -0.156 |                  -0.042 |                        0.052 |                     -0.033 |              0.319 |                        0.24  |               -0.013 |                -0.049 |                -0.006 |               0.426 |                      0.25  |                           0.331 |                    1     |                      0.811 |
| total_transaction_amount      |         0.009 |        -0.004 |              0.021 |            -0.035 |                    -0.043 |                    0.079 |                    -0.153 |                   0.201 |                  0.274 |                -0.182 |               0.446 |                     0.453 |                  -0.373 |                -0.155 |                  -0.036 |                        0.042 |                     -0.041 |              0.229 |                        0.177 |               -0.033 |                -0.072 |                -0.016 |               0.51  |                      0.201 |                           0.284 |                    0.811 |                      1     |

### Highly Correlated Feature Pairs (|r| > 0.85)
- None found

## 11. Correlation with Target Label (is_anomaly)
| Feature | Train | Test | |
|---------|-------|------|----|
| request_rate_deviation | 0.5027 | 0.5148 |
| sensitive_permission_count | 0.3510 | 0.3978 |
| requests_per_min | 0.3075 | 0.3203 |
| permission_count | 0.2960 | 0.3846 |
| file_modification_count | 0.2896 | 0.3090 |
| sensitive_access_denied | 0.2871 | 0.2791 |
| denied_requests | 0.2854 | 0.2891 |
| permission_changes | 0.2800 | 0.2789 |
| file_access_count | 0.2760 | 0.3485 |
| background_location_access | 0.2752 | 0.2728 |
| unconfirmed_transaction_count | 0.2666 | 0.2617 |
| background_requests | 0.2623 | 0.3270 |
| location_access_count | 0.2574 | 0.3460 |
| data_transferred_mb | 0.2461 | 0.2486 |
| failed_transaction_count | 0.2309 | 0.2189 |
| background_camera_access | 0.2247 | 0.2095 |
| background_time_ratio | 0.1908 | 0.2629 |
| transaction_count | 0.1850 | 0.1782 |
| background_mic_access | 0.1821 | 0.1640 |
| network_connections | 0.1802 | 0.2202 |
| total_transaction_amount | 0.1769 | 0.2017 |
| camera_access_count | 0.1763 | 0.2348 |
| payment_access_count | 0.1702 | 0.1494 |
| avg_transaction_amount | 0.1646 | 0.2284 |
| microphone_access_count | 0.1441 | 0.1432 |
| day_of_week | 0.0109 | 0.0174 |
| hour_of_day | 0.0045 | -0.0347 |

### Features with |correlation| > 0.3 with target
- requests_per_min: train r=0.3075, test r=0.3203
- request_rate_deviation: train r=0.5027, test r=0.5148
- sensitive_permission_count: train r=0.3510, test r=0.3978

## 12. Flagged Feature Investigation
### denied_requests
- Train: mean=2.974, std=9.673, min=0.000, max=279.000
- Test: mean=4.598, std=15.237, min=0.000, max=269.000
- Train is_anomaly=0: mean=1.811, std=2.191, n=8491
- Train is_anomaly=1: mean=9.522, std=23.300, n=1509
### sensitive_access_denied
- Train: mean=0.499, std=1.897, min=0.000, max=53.000
- Test: mean=0.783, std=2.847, min=0.000, max=57.000
- Train is_anomaly=0: mean=0.269, std=0.557, n=8491
- Train is_anomaly=1: mean=1.791, std=4.489, n=1509
### request_rate_deviation
- Train: mean=0.444, std=0.421, min=0.000, max=3.748
- Test: mean=0.554, std=0.615, min=0.000, max=4.301
- Train is_anomaly=0: mean=0.355, std=0.208, n=8491
- Train is_anomaly=1: mean=0.946, std=0.795, n=1509
### requests_per_min
- Train: mean=55.931, std=100.526, min=2.030, max=2893.840
- Test: mean=73.043, std=147.385, min=2.010, max=2376.930
- Train is_anomaly=0: mean=42.902, std=35.300, n=8491
- Train is_anomaly=1: mean=129.246, std=231.636, n=1509
### microphone_access_count
- Train: mean=7.416, std=14.706, min=0.000, max=138.000
- Test: mean=8.129, std=16.348, min=0.000, max=150.000
- Train is_anomaly=0: mean=6.523, std=12.000, n=8491
- Train is_anomaly=1: mean=12.445, std=24.361, n=1509
### location_access_count
- Train: mean=3.814, std=4.434, min=0.000, max=36.000
- Test: mean=4.478, std=5.437, min=0.000, max=45.000
- Train is_anomaly=0: mean=3.333, std=3.576, n=8491
- Train is_anomaly=1: mean=6.521, std=7.051, n=1509
### payment_access_count
- Train: mean=9.505, std=52.883, min=0.000, max=2037.000
- Test: mean=14.710, std=90.865, min=0.000, max=2002.000
- Train is_anomaly=0: mean=5.710, std=11.913, n=8491
- Train is_anomaly=1: mean=30.860, std=131.175, n=1509
## 13. denied_requests vs sensitive_access_denied Relationship
- Records where sensitive_access_denied > denied_requests (train): 0
- Records where sensitive_access_denied > denied_requests (test): 0
- Correlation (train): 0.6207
- Correlation (test): 0.6801
- sensitive/denied ratio (train): mean=0.082, std=0.162
- sensitive/denied ratio (test): mean=0.100, std=0.180

## 14. Data Leakage Investigation
### Potential leakage sources checked:
- No features perfectly separate normal and anomalous classes.
- Records where denied_requests == sensitive_access_denied: 3392 (33.92%)
- These features are correlated but NOT redundant (sensitive denials are a subset of total denials).
- No feature directly encodes is_anomaly (verified by generation script).
- Anomalies are generated through rule-based behavioural shifts, not by modifying a single leaked column.

## 15. Application and Session Analysis
- Unique apps in train: 500
- Unique apps in test: 200
- Common apps between train and test: 200
- Apps only in test: 0
- Unique sessions in train: 9949
- Unique sessions in test: 3988

## 16. Summary and Recommendations
### Key Findings
1. No missing values detected in either dataset.
2. No infinite or invalid numerical values detected.
3. denied_requests and sensitive_access_denied are correlated but not redundant.
4. No feature perfectly separates classes - no obvious target leakage.
5. Train and test sets share application identities, enabling group-based evaluation.

### Recommendations
- Use stratified splitting to maintain class balance across splits.
- Implement group-based evaluation by app_id to test generalization to unseen apps.
- Monitor denied_requests / sensitive_access_denied for potential redundancy.
- Apply feature engineering to create behavioural ratios before model training.
