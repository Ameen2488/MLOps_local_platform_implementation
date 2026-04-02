"""
Model monitoring and data drift detection using Evidently.

This module provides tools to monitor production ML systems:
1. DriftMonitor class: Detect data drift between training and production data
2. generate_report(): Create detailed HTML visualization reports
3. get_alerts(): Get historical alerts when drift exceeded thresholds

WHY THIS MATTERS:
In production, ML models degrade over time because:
- User behavior changes (concept drift)
- Data pipelines change (data drift)
- External factors shift (seasonality, market changes)

Without monitoring, you'd only know about problems when users complain.
With drift detection, you can:
- Proactively detect issues before they affect users
- Understand WHY model performance changed
- Know when to retrain your model

This is a critical part of ML observability - "you can't improve what you don't measure."
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Evidently for drift detection and reporting
# Evidently compares reference (training) data against current (production) data
# and detects statistical drift using various methods (KS test, PSI, etc.)
from evidently import Report
from evidently.metrics import ValueDrift, DriftedColumnsCount
from evidently.presets import DataDriftPreset


class DriftMonitor:
    """
    Monitor for detecting data drift between reference (training) and current (production) data.
    
    This class implements the core drift detection logic:
    - Store reference (training) data as baseline
    - Compare new data against baseline using statistical tests
    - Track drift history for alerting and analysis
    
    Usage pattern:
        1. Initialize with training data (reference)
        2. Periodically check new data for drift
        3. Alert when drift exceeds threshold
        4. Generate reports for investigation
    
    Why detect drift?
    - Models trained on historical data may not generalize to new patterns
    - Drift often precedes model performance degradation
    - Early detection gives time to retrain before major issues
    """
    
    def __init__(self, reference_data: pd.DataFrame, feature_columns: Optional[List[str]] = None):
        """
        Initialize the drift monitor with reference (training) data.
        
        The reference data represents the "known good" state - typically
        your training data or a validated production snapshot.
        
        Args:
            reference_data: The training/production baseline data to compare against
            feature_columns: Columns to monitor (default: all numeric columns)
        
        Example:
            >>> train_df = pd.read_csv('data/train.csv')
            >>> monitor = DriftMonitor(train_df, feature_cols=['amount', 'hour'])
        """
        self.reference = reference_data
        # Default to all numeric columns if not specified
        self.feature_columns = feature_columns or reference_data.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        # History tracks all drift checks for alerting and analysis
        self.history: List[Dict[str, Any]] = []
        
        print(f"Drift monitor initialized with {len(self.reference):,} reference samples")
        print(f"Monitoring columns: {self.feature_columns}")
    
    def check_drift(self, current_data: pd.DataFrame, threshold: float = 0.1) -> Dict[str, Any]:
        """
        Check for drift between reference and current data.
        
        Uses the Kolmogorov-Smirnov (KS) test to detect distributional changes:
        - For each feature, compare the distribution of reference vs current data
        - If p-value < 0.05, the distributions are significantly different
        - Drift share = fraction of features with significant drift
        
        Why KS test?
        - Non-parametric (doesn't assume specific distribution)
        - Detects any kind of distribution change (mean, variance, shape)
        - Widely used for drift detection
        
        Args:
            current_data: Current/production data to check for drift
            threshold: Drift share threshold for alerting (default 10%)
                      If drift_share > threshold, alert is triggered
        
        Returns:
            Dictionary with drift results:
            - drift_detected: Boolean - was any drift found?
            - drift_share: Fraction of features that drifted (0.0 to 1.0)
            - drifted_columns: List of columns that drifted
            - alert: Boolean - should we alert based on threshold?
        
        Example:
            >>> result = monitor.check_drift(production_data, threshold=0.1)
            >>> if result['alert']:
            ...     send_alert("Drift detected!")
        """
        from scipy import stats
        
        # Extract the columns we're monitoring
        ref_subset = self.reference[self.feature_columns]
        cur_subset = current_data[self.feature_columns]
        
        # Run KS test for each feature
        # KS test compares whether two samples come from the same distribution
        drifted_columns = []
        for col in self.feature_columns:
            # Drop NaN values to avoid errors
            ref_values = ref_subset[col].dropna()
            cur_values = cur_subset[col].dropna()
            
            # KS test: if p-value < 0.05, distributions are significantly different
            statistic, p_value = stats.ks_2samp(ref_values, cur_values)
            if p_value < 0.05:
                drifted_columns.append(col)
        
        # Calculate drift metrics
        n_features = len(self.feature_columns)
        n_drifted = len(drifted_columns)
        drift_share = n_drifted / n_features if n_features > 0 else 0
        
        # Build result dictionary
        result = {
            'timestamp': datetime.now().isoformat(),
            'drift_detected': n_drifted > 0,  # Any drift at all?
            'drift_share': drift_share,  # What fraction of features drifted?
            'drifted_columns': drifted_columns,  # Which specific columns?
            'n_features': n_features,
            'n_drifted': n_drifted,
            'current_samples': len(current_data),
            'threshold': threshold,
            'alert': drift_share > threshold  # Should we alert?
        }
        
        # Save to history for later analysis
        self.history.append(result)
        
        return result
    
    def generate_report(self, current_data: pd.DataFrame, output_path: str = "drift_report.html"):
        """
        Generate a detailed HTML drift report.
        
        Creates an interactive HTML report with:
        - Drift summary metrics
        - Column-by-column drift analysis
        - Distribution comparison visualizations (text-based)
        
        This custom report works regardless of Evidently API changes.
        
        Args:
            current_data: Current data to compare against reference
            output_path: Where to save the HTML report
        
        Example:
            >>> monitor.generate_report(production_data, "drift_report.html")
            >>> # Open in browser to see report
        """
        from scipy import stats
        
        ref_subset = self.reference[self.feature_columns]
        cur_subset = current_data[self.feature_columns]
        
        # Build analysis for each column
        column_analysis = []
        for col in self.feature_columns:
            ref_vals = ref_subset[col].dropna()
            cur_vals = cur_subset[col].dropna()
            
            # KS test
            ks_stat, ks_pval = stats.ks_2samp(ref_vals, cur_vals)
            
            # Basic statistics
            ref_stats = {
                'mean': ref_vals.mean(),
                'std': ref_vals.std(),
                'min': ref_vals.min(),
                'max': ref_vals.max(),
                'median': ref_vals.median()
            }
            cur_stats = {
                'mean': cur_vals.mean(),
                'std': cur_vals.std(),
                'min': cur_vals.min(),
                'max': cur_vals.max(),
                'median': cur_vals.median()
            }
            
            drift_detected = ks_pval < 0.05
            
            column_analysis.append({
                'column': col,
                'drift_detected': drift_detected,
                'ks_pvalue': ks_pval,
                'ks_statistic': ks_stat,
                'ref_stats': ref_stats,
                'cur_stats': cur_stats
            })
        
        # Count drifted columns
        n_drifted = sum(1 for c in column_analysis if c['drift_detected'])
        drift_share = n_drifted / len(self.feature_columns) if self.feature_columns else 0
        
        # Build HTML report
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Drift Detection Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
        .summary {{ background: white; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .alert {{ background: #ffebee; border: 1px solid #ef5350; padding: 15px; border-radius: 4px; margin: 10px 0; }}
        .no-alert {{ background: #e8f5e9; border: 1px solid #66bb6a; padding: 15px; border-radius: 4px; margin: 10px 0; }}
        table {{ border-collapse: collapse; width: 100%; background: white; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        .drift-yes {{ color: #d32f2f; font-weight: bold; }}
        .drift-no {{ color: #388e3c; }}
        .stat-table {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>📊 Drift Detection Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Reference samples:</strong> {len(self.reference):,}</p>
        <p><strong>Current samples:</strong> {len(current_data):,}</p>
        <p><strong>Features monitored:</strong> {len(self.feature_columns)}</p>
        <p><strong>Drifted columns:</strong> {n_drifted}</p>
        <p><strong>Drift share:</strong> {drift_share:.1%}</p>
        {"<div class='alert'>⚠️ DRIFT DETECTED - Action may be required</div>" if drift_share > 0.1 else "<div class='no-alert'>✓ No significant drift detected</div>"}
    </div>
    
    <h2>Column Analysis</h2>
    <table>
        <tr>
            <th>Column</th>
            <th>Drift Detected</th>
            <th>KS p-value</th>
            <th>Reference Mean</th>
            <th>Current Mean</th>
        </tr>
"""
        
        for col_analysis in column_analysis:
            drift_class = "drift-yes" if col_analysis['drift_detected'] else "drift-no"
            drift_text = "⚠️ YES" if col_analysis['drift_detected'] else "✓ NO"
            
            html += f"""
        <tr>
            <td><strong>{col_analysis['column']}</strong></td>
            <td class="{drift_class}">{drift_text}</td>
            <td>{col_analysis['ks_pvalue']:.4f}</td>
            <td>{col_analysis['ref_stats']['mean']:.2f}</td>
            <td>{col_analysis['cur_stats']['mean']:.2f}</td>
        </tr>
"""
        
        html += """
    </table>
    
    <h2>Detailed Statistics</h2>
"""
        
        for col_analysis in column_analysis:
            html += f"""
    <div class="stat-table">
        <h3>{col_analysis['column']}</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Reference</th>
                <th>Current</th>
                <th>Change</th>
            </tr>
            <tr>
                <td>Mean</td>
                <td>{col_analysis['ref_stats']['mean']:.2f}</td>
                <td>{col_analysis['cur_stats']['mean']:.2f}</td>
                <td>{col_analysis['cur_stats']['mean'] - col_analysis['ref_stats']['mean']:+.2f}</td>
            </tr>
            <tr>
                <td>Std Dev</td>
                <td>{col_analysis['ref_stats']['std']:.2f}</td>
                <td>{col_analysis['cur_stats']['std']:.2f}</td>
                <td>{col_analysis['cur_stats']['std'] - col_analysis['ref_stats']['std']:+.2f}</td>
            </tr>
            <tr>
                <td>Min</td>
                <td>{col_analysis['ref_stats']['min']:.2f}</td>
                <td>{col_analysis['cur_stats']['min']:.2f}</td>
                <td>-</td>
            </tr>
            <tr>
                <td>Max</td>
                <td>{col_analysis['ref_stats']['max']:.2f}</td>
                <td>{col_analysis['cur_stats']['max']:.2f}</td>
                <td>-</td>
            </tr>
            <tr>
                <td>Median</td>
                <td>{col_analysis['ref_stats']['median']:.2f}</td>
                <td>{col_analysis['cur_stats']['median']:.2f}</td>
                <td>{col_analysis['cur_stats']['median'] - col_analysis['ref_stats']['median']:+.2f}</td>
            </tr>
        </table>
    </div>
"""
        
        html += """
    <h2>Interpretation</h2>
    <ul>
        <li><strong>KS test:</strong> Kolmogorov-Smirnov test compares distributions. p-value < 0.05 indicates drift.</li>
        <li><strong>Drift share:</strong> Fraction of features with statistically significant drift.</li>
        <li><strong>Action:</strong> If drift > 10%, consider retraining your model with recent data.</li>
    </ul>
</body>
</html>"""
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write HTML file
        with open(output_path, 'w') as f:
            f.write(html)
        
        print(f"Drift report saved to {output_path}")
        print(f"Open this file in a browser to view detailed analysis.")
    
    def get_alerts(self, threshold: float = 0.1) -> List[Dict[str, Any]]:
        """
        Get all historical alerts where drift exceeded threshold.
        
        Useful for:
        - Reviewing past incidents
        - Finding patterns in when drift occurs
        - Generating audit logs
        
        Args:
            threshold: Minimum drift_share to include in alerts
        
        Returns:
            List of alert dictionaries with timestamp, severity, and details
        
        Example:
            >>> alerts = monitor.get_alerts(threshold=0.2)
            >>> for alert in alerts:
            ...     print(f"{alert['timestamp']}: {alert['message']}")
        """
        return [
            {
                'timestamp': r['timestamp'],
                'severity': 'HIGH' if r['drift_share'] > 0.3 else 'MEDIUM',
                'drift_share': r['drift_share'],
                'message': f"Drift detected: {r['drift_share']:.1%} of features drifted",
                'drifted_columns': r['drifted_columns']
            }
            for r in self.history
            if r['drift_share'] > threshold
        ]
    
    def summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of all drift checks.
        
        Returns:
            Dictionary with aggregated metrics:
            - total_checks: How many times drift was checked
            - total_alerts: How many times alert threshold was exceeded
            - avg_drift_share: Average drift across all checks
            - max_drift_share: Maximum drift observed
        
        Example:
            >>> summary = monitor.summary()
            >>> print(f"Total alerts: {summary['total_alerts']}")
        """
        if not self.history:
            return {"message": "No drift checks performed yet"}
        
        drift_shares = [r['drift_share'] for r in self.history]
        alerts = [r for r in self.history if r['alert']]
        
        return {
            'total_checks': len(self.history),
            'total_alerts': len(alerts),
            'avg_drift_share': np.mean(drift_shares),
            'max_drift_share': np.max(drift_shares),
            'first_check': self.history[0]['timestamp'],
            'last_check': self.history[-1]['timestamp']
        }


def simulate_drift_scenarios():
    """
    Demonstrate drift detection with different simulated scenarios.
    
    This function shows how DriftMonitor works in practice by running
    several scenarios that might occur in production:
    
    1. Normal test data (should have low/no drift)
    2. Fraud spike (more fraud than training)
    3. Amount inflation (transactions got larger)
    4. Time shift (different hour patterns)
    
    Each scenario demonstrates a different type of drift that can
    affect model performance in production.
    """
    import sys
    import os
    # Add project root to path for imports
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from generate_data import generate_transactions
    
    print("="*70)
    print("DRIFT DETECTION SIMULATION")
    print("="*70)
    
    # Load reference (training) data
    print("\n1. Loading reference data (training set)...")
    reference = pd.read_csv('data/train.csv')
    feature_cols = ['amount', 'hour', 'day_of_week']
    
    # Create drift monitor with training data as reference
    monitor = DriftMonitor(reference, feature_cols)
    
    # Scenario 1: Test data (similar distribution to training)
    print("\n" + "-"*70)
    print("SCENARIO 1: Test data (similar distribution)")
    print("-"*70)
    test_data = pd.read_csv('data/test.csv')
    result = monitor.check_drift(test_data)
    print(f"  Drift detected: {result['drift_detected']}")
    print(f"  Drift share: {result['drift_share']:.1%}")
    print(f"  Drifted columns: {result['drifted_columns']}")
    print(f"  Alert triggered: {result['alert']}")
    
    # Scenario 2: Fraud spike (5x normal fraud rate)
    print("\n" + "-"*70)
    print("SCENARIO 2: Fraud spike (10% fraud rate instead of 2%)")
    print("-"*70)
    fraud_spike = generate_transactions(n_samples=2000, fraud_ratio=0.10, seed=101)
    result = monitor.check_drift(fraud_spike)
    print(f"  Drift detected: {result['drift_detected']}")
    print(f"  Drift share: {result['drift_share']:.1%}")
    print(f"  Drifted columns: {result['drifted_columns']}")
    print(f"  Alert triggered: {result['alert']}")
    
    # Scenario 3: Amount inflation (transactions are larger)
    print("\n" + "-"*70)
    print("SCENARIO 3: Amount inflation (2x multiplier)")
    print("-"*70)
    inflated = test_data.copy()
    inflated['amount'] = inflated['amount'] * 2
    result = monitor.check_drift(inflated)
    print(f"  Drift detected: {result['drift_detected']}")
    print(f"  Drift share: {result['drift_share']:.1%}")
    print(f"  Drifted columns: {result['drifted_columns']}")
    print(f"  Alert triggered: {result['alert']}")
    
    # Scenario 4: Time shift (different hour patterns)
    print("\n" + "-"*70)
    print("SCENARIO 4: Time shift (mostly late-night transactions)")
    print("-"*70)
    night_shift = test_data.copy()
    night_shift['hour'] = np.random.choice([0, 1, 2, 3, 22, 23], size=len(night_shift))
    result = monitor.check_drift(night_shift)
    print(f"  Drift detected: {result['drift_detected']}")
    print(f"  Drift share: {result['drift_share']:.1%}")
    print(f"  Drifted columns: {result['drifted_columns']}")
    print(f"  Alert triggered: {result['alert']}")
    
    # Generate detailed HTML report
    print("\n" + "-"*70)
    print("GENERATING DETAILED REPORT")
    print("-"*70)
    monitor.generate_report(night_shift, "drift_report.html")
    
    # Print summary
    print("\n" + "-"*70)
    print("MONITORING SUMMARY")
    print("-"*70)
    summary = monitor.summary()
    print(f"  Total checks: {summary['total_checks']}")
    print(f"  Total alerts: {summary['total_alerts']}")
    print(f"  Average drift share: {summary['avg_drift_share']:.1%}")
    print(f"  Maximum drift share: {summary['max_drift_share']:.1%}")
    
    # Print any alerts
    alerts = monitor.get_alerts()
    if alerts:
        print(f"\n  Alerts ({len(alerts)}):")
        for alert in alerts:
            print(f"    [{alert['severity']}] {alert['message']}")
    
    print("\n" + "="*70)
    print("DRIFT DETECTION SIMULATION COMPLETE")
    print("="*70)
    print("\nOpen drift_report.html in your browser to see detailed visualizations!")


if __name__ == "__main__":
    simulate_drift_scenarios()