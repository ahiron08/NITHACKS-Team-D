import sys, joblib, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from feature_engineering import engineer_features, RAW_NUMERIC_FEATURES

# Test with random data
minimal_data = {f: float(np.random.uniform(0, 100)) for f in RAW_NUMERIC_FEATURES}
df = pd.DataFrame([minimal_data])

# Engineer features
df_eng, scaler, feature_cols = engineer_features(df, fit_scaler=None)
print(f'Feature engineering: {len(feature_cols)} features engineered')

# Test detector
from edith_detector import EdithDetector
detector = EdithDetector(model_type='xgboost')
detector.load()
telemetry = {f: float(minimal_data[f]) for f in RAW_NUMERIC_FEATURES}
result = detector.predict(telemetry)
print(f'Detector: is_anomaly={result["is_anomaly"]}, score={result["anomaly_score"]:.4f}, risk={result["risk_category"]}')
print('V2 integration WORKS')