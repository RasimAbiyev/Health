"""
Model Monitoring Module for Churn Prediction

This module provides utilities for monitoring model performance and detecting data drift
in production environments. It includes:
- Population Stability Index (PSI) calculation for drift detection
- Feature distribution comparison
- Model performance tracking
- Alerting mechanisms

Note: This is a reference implementation for academic purposes.
Production deployment would require integration with enterprise monitoring tools.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI) to detect distribution drift.
    
    PSI measures the change in distribution between two samples:
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.25: Moderate change, investigate
    - PSI >= 0.25: Significant change, action required
    
    Args:
        expected: Reference distribution (training data)
        actual: New distribution (production data)
        bins: Number of bins for discretization
        
    Returns:
        PSI value (float)
        
    Example:
        >>> training_recency = df_train['Recency']
        >>> production_recency = df_prod['Recency']
        >>> psi = calculate_psi(training_recency, production_recency)
        >>> if psi > 0.25:
        >>>     print("Significant drift detected!")
    """
    # Handle edge cases
    if len(expected) == 0 or len(actual) == 0:
        logging.warning("Empty data provided to PSI calculation")
        return np.nan
    
    # Create bins based on expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # Remove duplicates
    
    if len(breakpoints) < 2:
        logging.warning("Insufficient unique values for PSI calculation")
        return np.nan
    
    # Calculate distributions
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    
    # Avoid division by zero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    # Calculate PSI
    psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    
    return psi_value


def detect_data_drift(
    reference_data: pd.DataFrame,
    new_data: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str] = None,
    psi_threshold: float = 0.25
) -> Dict[str, Dict]:
    """
    Detect data drift across multiple features.
    
    Args:
        reference_data: Training/reference dataset
        new_data: New production dataset
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names (optional)
        psi_threshold: Threshold for PSI alert (default: 0.25)
        
    Returns:
        Dictionary with drift metrics for each feature
        
    Example:
        >>> drift_report = detect_data_drift(
        >>>     df_train, df_prod,
        >>>     numerical_features=['Recency', 'Frequency', 'Monetary'],
        >>>     categorical_features=['customer_state']
        >>> )
        >>> for feature, metrics in drift_report.items():
        >>>     if metrics['drift_detected']:
        >>>         print(f"Drift detected in {feature}: PSI={metrics['psi']:.3f}")
    """
    drift_report = {}
    
    # Check numerical features
    for feature in numerical_features:
        if feature not in reference_data.columns or feature not in new_data.columns:
            logging.warning(f"Feature {feature} not found in data")
            continue
            
        psi = calculate_psi(reference_data[feature], new_data[feature])
        
        drift_report[feature] = {
            'type': 'numerical',
            'psi': psi,
            'drift_detected': psi >= psi_threshold if not np.isnan(psi) else False,
            'reference_mean': reference_data[feature].mean(),
            'new_mean': new_data[feature].mean(),
            'reference_std': reference_data[feature].std(),
            'new_std': new_data[feature].std()
        }
    
    # Check categorical features
    if categorical_features:
        for feature in categorical_features:
            if feature not in reference_data.columns or feature not in new_data.columns:
                logging.warning(f"Feature {feature} not found in data")
                continue
            
            # Calculate distribution shift for categorical
            ref_dist = reference_data[feature].value_counts(normalize=True)
            new_dist = new_data[feature].value_counts(normalize=True)
            
            # Align distributions
            all_categories = set(ref_dist.index) | set(new_dist.index)
            ref_aligned = pd.Series({cat: ref_dist.get(cat, 0.0001) for cat in all_categories})
            new_aligned = pd.Series({cat: new_dist.get(cat, 0.0001) for cat in all_categories})
            
            # Calculate PSI for categorical
            psi = np.sum((new_aligned - ref_aligned) * np.log(new_aligned / ref_aligned))
            
            drift_report[feature] = {
                'type': 'categorical',
                'psi': psi,
                'drift_detected': psi >= psi_threshold,
                'new_categories': list(set(new_dist.index) - set(ref_dist.index)),
                'missing_categories': list(set(ref_dist.index) - set(new_dist.index))
            }
    
    return drift_report


def monitor_model_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    baseline_metrics: Dict[str, float],
    alert_threshold: float = 0.05
) -> Dict[str, any]:
    """
    Monitor model performance and compare against baseline.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        baseline_metrics: Dictionary of baseline metrics (accuracy, precision, recall, f1, roc_auc)
        alert_threshold: Performance drop threshold for alerts (default: 5%)
        
    Returns:
        Dictionary with current metrics and alerts
        
    Example:
        >>> baseline = {'accuracy': 0.873, 'precision': 0.845, 'recall': 0.792, 'f1': 0.817}
        >>> performance = monitor_model_performance(y_true, y_pred, y_proba, baseline)
        >>> if performance['alerts']:
        >>>     print("Performance degradation detected!")
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    current_metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan
    }
    
    # Check for performance degradation
    alerts = []
    for metric_name, current_value in current_metrics.items():
        if metric_name in baseline_metrics and not np.isnan(current_value):
            baseline_value = baseline_metrics[metric_name]
            drop = baseline_value - current_value
            
            if drop > alert_threshold:
                alerts.append({
                    'metric': metric_name,
                    'baseline': baseline_value,
                    'current': current_value,
                    'drop': drop,
                    'severity': 'high' if drop > 0.1 else 'medium'
                })
    
    return {
        'timestamp': datetime.now().isoformat(),
        'current_metrics': current_metrics,
        'baseline_metrics': baseline_metrics,
        'alerts': alerts,
        'performance_ok': len(alerts) == 0
    }


def log_prediction_metrics(
    predictions: pd.DataFrame,
    timestamp: Optional[datetime] = None,
    log_file: str = 'logs/predictions.log'
) -> None:
    """
    Log prediction metrics for monitoring and analysis.
    
    Args:
        predictions: DataFrame with columns ['customer_id', 'churn_probability', 'prediction']
        timestamp: Timestamp for logging (default: current time)
        log_file: Path to log file
        
    Example:
        >>> predictions_df = pd.DataFrame({
        >>>     'customer_id': ['c1', 'c2', 'c3'],
        >>>     'churn_probability': [0.85, 0.23, 0.67],
        >>>     'prediction': [1, 0, 1]
        >>> })
        >>> log_prediction_metrics(predictions_df)
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    # Calculate summary statistics
    summary = {
        'timestamp': timestamp.isoformat(),
        'total_predictions': len(predictions),
        'predicted_churners': predictions['prediction'].sum(),
        'churn_rate': predictions['prediction'].mean(),
        'avg_churn_probability': predictions['churn_probability'].mean(),
        'high_risk_customers': (predictions['churn_probability'] > 0.8).sum(),
        'medium_risk_customers': ((predictions['churn_probability'] > 0.5) & 
                                  (predictions['churn_probability'] <= 0.8)).sum(),
        'low_risk_customers': (predictions['churn_probability'] <= 0.5).sum()
    }
    
    # Log to file
    import json
    import os
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    with open(log_file, 'a') as f:
        f.write(json.dumps(summary) + '\n')
    
    logging.info(f"Logged prediction metrics: {summary['total_predictions']} predictions, "
                f"{summary['churn_rate']:.2%} churn rate")


def generate_drift_report(drift_results: Dict[str, Dict]) -> str:
    """
    Generate a human-readable drift report.
    
    Args:
        drift_results: Output from detect_data_drift()
        
    Returns:
        Formatted string report
    """
    report_lines = ["=" * 60]
    report_lines.append("DATA DRIFT DETECTION REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 60)
    
    drift_detected = False
    
    for feature, metrics in drift_results.items():
        psi = metrics['psi']
        
        if metrics['drift_detected']:
            drift_detected = True
            status = "⚠️ DRIFT DETECTED"
        elif psi >= 0.1:
            status = "⚡ MODERATE CHANGE"
        else:
            status = "✅ STABLE"
        
        report_lines.append(f"\n{feature}:")
        report_lines.append(f"  Status: {status}")
        report_lines.append(f"  PSI: {psi:.4f}")
        
        if metrics['type'] == 'numerical':
            report_lines.append(f"  Mean: {metrics['reference_mean']:.2f} → {metrics['new_mean']:.2f}")
            report_lines.append(f"  Std: {metrics['reference_std']:.2f} → {metrics['new_std']:.2f}")
        else:
            if metrics['new_categories']:
                report_lines.append(f"  New categories: {metrics['new_categories']}")
            if metrics['missing_categories']:
                report_lines.append(f"  Missing categories: {metrics['missing_categories']}")
    
    report_lines.append("\n" + "=" * 60)
    if drift_detected:
        report_lines.append("⚠️ ACTION REQUIRED: Significant drift detected. Consider retraining.")
    else:
        report_lines.append("✅ No significant drift detected. Model is stable.")
    report_lines.append("=" * 60)
    
    return "\n".join(report_lines)


# Example usage
if __name__ == "__main__":
    print("Model Monitoring Module - Reference Implementation")
    print("=" * 60)
    print("\nThis module provides monitoring utilities for production ML models.")
    print("\nKey functions:")
    print("  - calculate_psi(): Detect distribution drift")
    print("  - detect_data_drift(): Monitor multiple features")
    print("  - monitor_model_performance(): Track model metrics")
    print("  - log_prediction_metrics(): Log predictions for analysis")
    print("\nFor usage examples, see function docstrings.")
    print("=" * 60)
