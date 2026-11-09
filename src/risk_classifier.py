"""
EDITH Risk Classifier

Classifies anomaly scores into risk categories (LOW, MEDIUM, HIGH, CRITICAL)
using the V2 risk thresholds.
"""

RISK_THRESHOLDS = {
    "LOW": 0.0,
    "MEDIUM": 0.3,
    "HIGH": 0.6,
    "CRITICAL": 0.8,
}

RISK_DESCRIPTIONS = {
    "LOW": "Normal behaviour",
    "MEDIUM": "Unusual behaviour",
    "HIGH": "Suspicious behaviour",
    "CRITICAL": "Multiple serious behavioural indicators",
}


def classify_risk(score, thresholds=None):
    """
    Classify an anomaly score into a risk category.
    
    Args:
        score: Anomaly score (or ensemble probability)
        thresholds: Dict of {risk_level: threshold_value}. If None, uses RISK_THRESHOLDS.
    
    Returns:
        dict with keys: category, description, is_anomaly
    """
    if thresholds is None:
        thresholds = RISK_THRESHOLDS
    
    if score >= thresholds["CRITICAL"]:
        category = "CRITICAL"
    elif score >= thresholds["HIGH"]:
        category = "HIGH"
    elif score >= thresholds["MEDIUM"]:
        category = "MEDIUM"
    else:
        category = "LOW"
    
    return {
        "category": category,
        "description": RISK_DESCRIPTIONS.get(category, "No description available"),
        "is_anomaly": category in ("HIGH", "CRITICAL"),
    }