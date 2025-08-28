import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import logging
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation with risk assessment and stakeholder communication.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        
    def calculate_comprehensive_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                                      model_name: str = "") -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            y_true (pd.Series): True values
            y_pred (np.ndarray): Predicted values
            model_name (str): Model name for logging
            
        Returns:
            Dict[str, float]: Comprehensive metrics
        """
        residuals = y_pred - y_true
        
        metrics = {
            # Basic metrics
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            
            # Error distribution
            'mean_error': np.mean(residuals),
            'std_error': np.std(residuals),
            'median_error': np.median(residuals),
            
            # Percentile metrics
            'mae_p95': np.percentile(np.abs(residuals), 95),
            'mae_p99': np.percentile(np.abs(residuals), 99),
            
            # Relative metrics
            'mape': np.mean(np.abs(residuals / y_true)) * 100,
            'median_ape': np.median(np.abs(residuals / y_true)) * 100,
        }
        
        logger.info(f"{model_name} Metrics - MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def bootstrap_confidence_intervals(self, y_true: pd.Series, y_pred: np.ndarray, 
                                     n_bootstrap: int = 1000, confidence: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """
        Calculate bootstrap confidence intervals for key metrics.
        
        Args:
            y_true (pd.Series): True values
            y_pred (np.ndarray): Predicted values
            n_bootstrap (int): Number of bootstrap samples
            confidence (float): Confidence level
            
        Returns:
            Dict[str, Tuple[float, float]]: Confidence intervals
        """
        logger.info(f"Computing bootstrap CI with {n_bootstrap} samples...")
        
        n_samples = len(y_true)
        bootstrap_metrics = {'mae': [], 'rmse': [], 'r2': []}
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true.iloc[indices]
            y_pred_boot = y_pred[indices]
            
            # Calculate metrics
            bootstrap_metrics['mae'].append(mean_absolute_error(y_true_boot, y_pred_boot))
            bootstrap_metrics['rmse'].append(np.sqrt(mean_squared_error(y_true_boot, y_pred_boot)))
            bootstrap_metrics['r2'].append(r2_score(y_true_boot, y_pred_boot))
        
        # Calculate confidence intervals
        alpha = 1 - confidence
        confidence_intervals = {}
        
        for metric, values in bootstrap_metrics.items():
            lower = np.percentile(values, (alpha/2) * 100)
            upper = np.percentile(values, (1 - alpha/2) * 100)
            confidence_intervals[metric] = (lower, upper)
        
        logger.info(f"Bootstrap CI complete. MAE CI: {confidence_intervals['mae']}")
        
        return confidence_intervals
    
    def scenario_analysis(self, X_test: pd.DataFrame, y_test: pd.Series, 
                         models: Dict[str, Any], scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Perform scenario analysis with different assumptions.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            models (Dict[str, Any]): Trained models
            scenarios (Dict[str, Dict[str, Any]]): Scenario configurations
            
        Returns:
            Dict[str, Dict[str, Any]]: Scenario results
        """
        scenario_results = {}
        
        for scenario_name, scenario_config in scenarios.items():
            logger.info(f"Running scenario: {scenario_name}")
            
            # Apply scenario modifications to test data
            X_test_scenario = X_test.copy()
            
            if 'volatility_shock' in scenario_config:
                if 'implied_volatility' in X_test_scenario.columns:
                    X_test_scenario['implied_volatility'] *= scenario_config['volatility_shock']
            
            if 'time_decay' in scenario_config:
                if 'time_to_expiry' in X_test_scenario.columns:
                    X_test_scenario['time_to_expiry'] *= scenario_config['time_decay']
            
            # Get predictions for each model
            scenario_results[scenario_name] = {}
            
            for model_name, model in models.items():
                try:
                    y_pred_scenario = model.predict(X_test_scenario)
                    metrics = self.calculate_comprehensive_metrics(y_test, y_pred_scenario, f"{model_name}_{scenario_name}")
                    scenario_results[scenario_name][model_name] = metrics
                except Exception as e:
                    logger.error(f"Error in scenario {scenario_name} for {model_name}: {e}")
                    scenario_results[scenario_name][model_name] = {'error': str(e)}
        
        return scenario_results
    
    def subgroup_analysis(self, X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray, 
                         subgroup_col: str = 'contract_type') -> Dict[str, Dict[str, float]]:
        """
        Analyze model performance across different subgroups.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            y_pred (np.ndarray): Predictions
            subgroup_col (str): Column to group by
            
        Returns:
            Dict[str, Dict[str, float]]: Subgroup metrics
        """
        if subgroup_col not in X_test.columns:
            logger.warning(f"Column {subgroup_col} not found for subgroup analysis")
            return {}
        
        subgroup_results = {}
        
        for subgroup in X_test[subgroup_col].unique():
            mask = X_test[subgroup_col] == subgroup
            y_true_sub = y_test[mask]
            y_pred_sub = y_pred[mask]
            
            if len(y_true_sub) > 0:
                metrics = self.calculate_comprehensive_metrics(y_true_sub, y_pred_sub, f"subgroup_{subgroup}")
                subgroup_results[str(subgroup)] = metrics
        
        return subgroup_results
    
    def create_diagnostic_plots(self, y_true: pd.Series, y_pred: np.ndarray, 
                               model_name: str = "Model") -> plt.Figure:
        """
        Create comprehensive diagnostic plots.
        
        Args:
            y_true (pd.Series): True values
            y_pred (np.ndarray): Predicted values
            model_name (str): Model name
            
        Returns:
            plt.Figure: Diagnostic plots figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name} Diagnostic Plots', fontsize=16)
        
        residuals = y_pred - y_true
        
        # 1. Predicted vs Actual
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Predicted vs Actual')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals vs Predicted
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Predicted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residuals Distribution
        axes[1, 0].hist(residuals, bins=50, alpha=0.7, density=True)
        axes[1, 0].axvline(x=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Residuals Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Q-Q Plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normal)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_comparison_plots(self, results: Dict[str, Dict[str, float]], 
                               confidence_intervals: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None) -> plt.Figure:
        """
        Create model comparison plots.
        
        Args:
            results (Dict[str, Dict[str, float]]): Model results
            confidence_intervals (Optional[Dict]): Bootstrap confidence intervals
            
        Returns:
            plt.Figure: Comparison plots figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        models = list(results.keys())
        metrics = ['mae', 'rmse', 'r2']
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in models]
            bars = axes[i].bar(models, values, alpha=0.7)
            
            # Add confidence intervals if available
            if confidence_intervals:
                for j, model in enumerate(models):
                    if model in confidence_intervals and metric in confidence_intervals[model]:
                        ci_lower, ci_upper = confidence_intervals[model][metric]
                        err_lower = values[j] - ci_lower
                        err_upper = ci_upper - values[j]
                        axes[i].errorbar(j, values[j], yerr=[[err_lower], [err_upper]], 
                                       fmt='none', color='black', capsize=5)
            
            axes[i].set_title(f'{metric.upper()}')
            axes[i].set_ylabel(metric.upper())
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i].text(j, v, f'{v:.4f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    def create_scenario_comparison(self, scenario_results: Dict[str, Dict[str, Any]], 
                                  metric: str = 'mae') -> plt.Figure:
        """
        Create scenario comparison visualization.
        
        Args:
            scenario_results (Dict): Scenario analysis results
            metric (str): Metric to compare
            
        Returns:
            plt.Figure: Scenario comparison figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        scenarios = list(scenario_results.keys())
        models = list(scenario_results[scenarios[0]].keys())
        
        x = np.arange(len(scenarios))
        width = 0.25
        
        for i, model in enumerate(models):
            values = []
            for scenario in scenarios:
                if 'error' not in scenario_results[scenario][model]:
                    values.append(scenario_results[scenario][model][metric])
                else:
                    values.append(np.nan)
            
            ax.bar(x + i * width, values, width, label=model, alpha=0.7)
        
        ax.set_xlabel('Scenarios')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'Model Performance Across Scenarios ({metric.upper()})')
        ax.set_xticks(x + width)
        ax.set_xticklabels(scenarios)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    

    
    def save_evaluation_results(self, results: Dict[str, Any], output_dir: str = '../data/processed/') -> str:
        """
        Save evaluation results to files.
        
        Args:
            results (Dict[str, Any]): All evaluation results
            output_dir (str): Output directory
            
        Returns:
            str: Saved file path
        """
        import os
        import json
        from datetime import datetime
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'model_evaluation_results_{timestamp}.json'
        filepath = os.path.join(output_dir, filename)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        results_serializable = convert_numpy(results)
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        logger.info(f"Evaluation results saved to {filepath}")
        return filepath
