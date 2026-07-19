"""
Model Evaluation Service

Provides comprehensive model evaluation including:
- K-Fold Cross-Validation
- Stratified evaluation for regression
- Learning curves
- Residual analysis
- Performance visualization
- Statistical tests
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    cross_validate,
    KFold,
    learning_curve,
    cross_val_predict
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error
)
from typing import Dict, Any, Tuple, Optional, List
import json
import os
from datetime import datetime


class ModelEvaluationService:
    """Comprehensive model evaluation with cross-validation"""

    def __init__(self, n_folds: int = 5, random_state: int = 42):
        """
        Initialize evaluation service

        Args:
            n_folds: Number of folds for cross-validation
            random_state: Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Dictionary of metric names and values
        """
        metrics = {}

        # Error metrics
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['median_ae'] = median_absolute_error(y_true, y_pred)

        # Percentage error metrics
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        metrics['mape'] = float(mape)

        # R-squared and adjusted R-squared
        metrics['r2'] = r2_score(y_true, y_pred)

        # Additional metrics
        metrics['max_error'] = float(np.max(np.abs(y_true - y_pred)))
        metrics['mean_error'] = float(np.mean(y_true - y_pred))  # Bias
        metrics['std_error'] = float(np.std(y_true - y_pred))

        # Percentage of predictions within X% of actual
        residuals_pct = np.abs((y_true - y_pred) / y_true) * 100
        metrics['pct_within_5pct'] = float(np.mean(residuals_pct <= 5) * 100)
        metrics['pct_within_10pct'] = float(np.mean(residuals_pct <= 10) * 100)
        metrics['pct_within_20pct'] = float(np.mean(residuals_pct <= 20) * 100)

        return metrics

    def cross_validate_model(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation

        Args:
            model: Scikit-learn compatible model
            X: Feature matrix
            y: Target vector
            return_predictions: Whether to return out-of-fold predictions

        Returns:
            Dictionary containing CV results and metrics
        """
        # Define scoring metrics
        scoring = {
            'neg_mean_squared_error': 'neg_mean_squared_error',
            'neg_mean_absolute_error': 'neg_mean_absolute_error',
            'r2': 'r2',
            'neg_root_mean_squared_error': 'neg_root_mean_squared_error'
        }

        # Require at least 10 samples per fold; skip CV for tiny datasets
        min_samples = self.n_folds * 10
        if len(X) < min_samples:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                f"Dataset too small for {self.n_folds}-fold CV "
                f"({len(X)} samples < {min_samples} required). Skipping CV."
            )
            stub_metrics = {
                'rmse_mean': float('nan'), 'rmse_std': 0.0,
                'mae_mean': float('nan'), 'mae_std': 0.0,
                'r2_mean': float('nan'), 'r2_std': 0.0,
                'train_rmse_mean': float('nan'), 'train_r2_mean': float('nan'),
                'overfitting_gap_rmse': 0.0, 'overfitting_gap_r2': 0.0,
            }
            result = {'cv_folds': self.n_folds, 'metrics': stub_metrics, 'fold_results': []}
            if return_predictions:
                result['oof_predictions'] = []
                result['oof_metrics'] = {'rmse': float('nan'), 'mae': float('nan'), 'r2': float('nan')}
            return result

        # Perform cross-validation
        cv_results = cross_validate(
            model,
            X,
            y,
            cv=self.cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,  # Use all available cores
            verbose=0
        )

        # Calculate mean and std for each metric
        results = {
            'cv_folds': self.n_folds,
            'metrics': {}
        }

        # Process test scores
        results['metrics']['rmse_mean'] = float(-cv_results['test_neg_root_mean_squared_error'].mean())
        results['metrics']['rmse_std'] = float(cv_results['test_neg_root_mean_squared_error'].std())

        results['metrics']['mae_mean'] = float(-cv_results['test_neg_mean_absolute_error'].mean())
        results['metrics']['mae_std'] = float(cv_results['test_neg_mean_absolute_error'].std())

        results['metrics']['r2_mean'] = float(cv_results['test_r2'].mean())
        results['metrics']['r2_std'] = float(cv_results['test_r2'].std())

        # Process training scores (to detect overfitting)
        results['metrics']['train_rmse_mean'] = float(-cv_results['train_neg_root_mean_squared_error'].mean())
        results['metrics']['train_r2_mean'] = float(cv_results['train_r2'].mean())

        # Calculate overfitting indicators
        results['metrics']['overfitting_gap_rmse'] = float(
            results['metrics']['train_rmse_mean'] - results['metrics']['rmse_mean']
        )
        results['metrics']['overfitting_gap_r2'] = float(
            results['metrics']['train_r2_mean'] - results['metrics']['r2_mean']
        )

        # Get out-of-fold predictions for further analysis
        if return_predictions:
            y_pred_cv = cross_val_predict(model, X, y, cv=self.cv, n_jobs=-1)
            results['oof_predictions'] = y_pred_cv.tolist()
            results['oof_metrics'] = self.calculate_metrics(y.values, y_pred_cv)

        # Add individual fold results
        results['fold_results'] = []
        for fold_idx in range(self.n_folds):
            fold_result = {
                'fold': fold_idx + 1,
                'test_rmse': float(-cv_results['test_neg_root_mean_squared_error'][fold_idx]),
                'test_mae': float(-cv_results['test_neg_mean_absolute_error'][fold_idx]),
                'test_r2': float(cv_results['test_r2'][fold_idx]),
                'train_rmse': float(-cv_results['train_neg_root_mean_squared_error'][fold_idx]),
                'train_r2': float(cv_results['train_r2'][fold_idx])
            }
            results['fold_results'].append(fold_result)

        return results

    def calculate_learning_curves(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        train_sizes: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculate learning curves to diagnose bias/variance

        Args:
            model: Scikit-learn compatible model
            X: Feature matrix
            y: Target vector
            train_sizes: Array of training set sizes to evaluate

        Returns:
            Dictionary containing learning curve data
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        # Learning curves require at least as many samples as folds
        if len(X) < self.n_folds * 2:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                f"Dataset too small for learning curves ({len(X)} samples). Skipping."
            )
            stub_sizes = [len(X)]
            return {
                'train_sizes': stub_sizes,
                'train_scores': {'mean': [float('nan')], 'std': [0.0]},
                'val_scores': {'mean': [float('nan')], 'std': [0.0]},
                'diagnosis': {
                    'final_train_rmse': float('nan'),
                    'final_val_rmse': float('nan'),
                    'generalization_gap': 0.0,
                    'status': 'insufficient_data',
                },
            }

        train_sizes_abs, train_scores, val_scores = learning_curve(
            model,
            X,
            y,
            train_sizes=train_sizes,
            cv=self.cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            shuffle=True,
            random_state=self.random_state
        )

        results = {
            'train_sizes': train_sizes_abs.tolist(),
            'train_scores': {
                'mean': (-train_scores.mean(axis=1)).tolist(),
                'std': train_scores.std(axis=1).tolist()
            },
            'val_scores': {
                'mean': (-val_scores.mean(axis=1)).tolist(),
                'std': val_scores.std(axis=1).tolist()
            }
        }

        # Diagnose learning behavior
        final_train_error = results['train_scores']['mean'][-1]
        final_val_error = results['val_scores']['mean'][-1]
        gap = final_val_error - final_train_error

        results['diagnosis'] = {
            'final_train_rmse': final_train_error,
            'final_val_rmse': final_val_error,
            'generalization_gap': gap,
            'status': 'overfitting' if gap > 5.0 else 'good_fit'
        }

        return results

    def analyze_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze prediction residuals

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Dictionary containing residual analysis
        """
        residuals = y_true - y_pred
        residuals_pct = (residuals / y_true) * 100

        analysis = {
            'residual_stats': {
                'mean': float(np.mean(residuals)),
                'std': float(np.std(residuals)),
                'min': float(np.min(residuals)),
                'max': float(np.max(residuals)),
                'q25': float(np.percentile(residuals, 25)),
                'median': float(np.median(residuals)),
                'q75': float(np.percentile(residuals, 75)),
                'skewness': float(pd.Series(residuals).skew()),
                'kurtosis': float(pd.Series(residuals).kurtosis())
            },
            'residual_pct_stats': {
                'mean': float(np.mean(residuals_pct)),
                'std': float(np.std(residuals_pct)),
                'median': float(np.median(residuals_pct))
            },
            'largest_errors': {
                'indices': np.argsort(np.abs(residuals))[-10:].tolist(),
                'values': residuals[np.argsort(np.abs(residuals))[-10:]].tolist(),
                'predictions': y_pred[np.argsort(np.abs(residuals))[-10:]].tolist(),
                'actuals': y_true[np.argsort(np.abs(residuals))[-10:]].tolist()
            }
        }

        # Normality test approximation
        # If residuals are approximately normal, skewness should be near 0 and kurtosis near 3
        analysis['normality_check'] = {
            'skewness_near_zero': abs(analysis['residual_stats']['skewness']) < 0.5,
            'kurtosis_near_three': abs(analysis['residual_stats']['kurtosis'] - 3) < 1.0,
            'likely_normal': (
                abs(analysis['residual_stats']['skewness']) < 0.5 and
                abs(analysis['residual_stats']['kurtosis'] - 3) < 1.0
            )
        }

        return analysis

    def evaluate_comprehensive(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        Perform comprehensive model evaluation

        Args:
            model: Trained model
            X_train: Training features
            y_train: Training target
            X_test: Optional test features
            y_test: Optional test target
            model_name: Name of the model

        Returns:
            Complete evaluation report
        """
        # Skip all evaluation for datasets too small to be meaningful.
        # This avoids hundreds of LightGBM/sklearn warnings and hanging CV loops.
        min_samples = self.n_folds * 10
        if len(X_train) < min_samples:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                f"Dataset too small for comprehensive evaluation "
                f"({len(X_train)} samples < {min_samples} required). "
                "Returning stub report."
            )
            nan = float('nan')
            stub_metrics = {k: nan for k in [
                'rmse_mean', 'rmse_std', 'mae_mean', 'mae_std',
                'r2_mean', 'r2_std', 'train_rmse_mean', 'train_r2_mean',
                'overfitting_gap_rmse', 'overfitting_gap_r2',
            ]}
            stub_perf = {k: nan for k in [
                'rmse', 'mae', 'mse', 'median_ae', 'mape', 'r2',
                'max_error', 'mean_error', 'std_error',
                'pct_within_5pct', 'pct_within_10pct', 'pct_within_20pct',
            ]}
            return {
                'model_name': model_name,
                'timestamp': datetime.now().isoformat(),
                'data_shape': {
                    'train_samples': len(X_train),
                    'train_features': len(X_train.columns),
                    'test_samples': len(X_test) if X_test is not None else 0,
                },
                'cross_validation': {'cv_folds': self.n_folds, 'metrics': stub_metrics, 'fold_results': []},
                'learning_curves': {
                    'train_sizes': [], 'train_scores': {'mean': [], 'std': []},
                    'val_scores': {'mean': [], 'std': []},
                    'diagnosis': {'final_train_rmse': nan, 'final_val_rmse': nan,
                                  'generalization_gap': 0.0, 'status': 'insufficient_data'},
                },
                'test_set': {'metrics': stub_perf, 'residual_analysis': {}},
                'train_set': {'metrics': stub_perf},
                'summary': {
                    'cross_validation_rmse': 'N/A (insufficient data)',
                    'cross_validation_r2': 'N/A (insufficient data)',
                    'overfitting_status': 'N/A',
                    'recommendation': f'Insufficient data for evaluation ({len(X_train)} samples).',
                },
            }

        report = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'data_shape': {
                'train_samples': len(X_train),
                'train_features': len(X_train.columns),
                'test_samples': len(X_test) if X_test is not None else 0
            }
        }

        # Cross-validation evaluation
        print(f"\nPerforming {self.n_folds}-fold cross-validation...")
        cv_results = self.cross_validate_model(model, X_train, y_train, return_predictions=True)
        report['cross_validation'] = cv_results

        # Learning curves
        print("Calculating learning curves...")
        learning_curves = self.calculate_learning_curves(model, X_train, y_train)
        report['learning_curves'] = learning_curves

        # Fit the model on training data for test/train set predictions.
        # CV and learning_curve use internal clones so the original is still unfitted.
        model.fit(X_train, y_train)

        # Test set evaluation (if provided)
        if X_test is not None and y_test is not None:
            print("Evaluating on test set...")
            y_pred_test = model.predict(X_test)
            test_metrics = self.calculate_metrics(y_test.values, y_pred_test)
            residual_analysis = self.analyze_residuals(y_test.values, y_pred_test)

            report['test_set'] = {
                'metrics': test_metrics,
                'residual_analysis': residual_analysis
            }

        # Training set evaluation (for comparison)
        y_pred_train = model.predict(X_train)
        train_metrics = self.calculate_metrics(y_train.values, y_pred_train)
        report['train_set'] = {
            'metrics': train_metrics
        }

        # Generate summary
        report['summary'] = self._generate_summary(report)

        return report

    def _generate_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate human-readable summary of evaluation"""
        cv = report['cross_validation']['metrics']

        summary = {
            'cross_validation_rmse': f"{cv['rmse_mean']:.2f} ± {cv['rmse_std']:.2f}",
            'cross_validation_r2': f"{cv['r2_mean']:.4f} ± {cv['r2_std']:.4f}",
            'overfitting_status': 'Likely Overfitting' if cv['overfitting_gap_r2'] > 0.05 else 'Good Generalization',
            'recommendation': self._get_recommendation(report)
        }

        if 'test_set' in report:
            test = report['test_set']['metrics']
            summary['test_rmse'] = f"{test['rmse']:.2f}"
            summary['test_r2'] = f"{test['r2']:.4f}"
            summary['test_mape'] = f"{test['mape']:.2f}%"
            summary['pct_within_10pct'] = f"{test['pct_within_10pct']:.1f}%"

        return summary

    def _get_recommendation(self, report: Dict[str, Any]) -> str:
        """Generate recommendation based on evaluation"""
        cv = report['cross_validation']['metrics']
        lc = report['learning_curves']['diagnosis']

        recommendations = []

        # Check for overfitting
        if cv['overfitting_gap_r2'] > 0.05:
            recommendations.append("Reduce model complexity or add regularization (overfitting detected)")

        # Check for underfitting
        if cv['r2_mean'] < 0.7:
            recommendations.append("Consider more complex model or better features (low R²)")

        # Check generalization gap
        if lc['generalization_gap'] > 5.0:
            recommendations.append("Collect more training data to reduce variance")

        # Check variance
        if cv['rmse_std'] > cv['rmse_mean'] * 0.2:
            recommendations.append("High variance across folds - consider stratification or more data")

        if not recommendations:
            return "Model performance looks good. Ready for production consideration."

        return " | ".join(recommendations)

    def save_report(self, report: Dict[str, Any], output_path: str = None):
        """Save evaluation report to JSON file"""
        if output_path is None:
            os.makedirs("evaluation_reports", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"evaluation_reports/{report['model_name']}_{timestamp}.json"

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nEvaluation report saved to: {output_path}")
        return output_path

    def print_summary(self, report: Dict[str, Any]):
        """Print formatted summary of evaluation results"""
        print("\n" + "="*70)
        print(f"Model Evaluation Summary: {report['model_name']}")
        print("="*70)

        cv = report['cross_validation']['metrics']
        print(f"\n{self.n_folds}-Fold Cross-Validation Results:")
        print(f"  RMSE: {cv['rmse_mean']:.2f} ± {cv['rmse_std']:.2f}")
        print(f"  MAE:  {cv['mae_mean']:.2f} ± {cv['mae_std']:.2f}")
        print(f"  R²:   {cv['r2_mean']:.4f} ± {cv['r2_std']:.4f}")

        print(f"\nOverfitting Analysis:")
        print(f"  Train RMSE: {cv['train_rmse_mean']:.2f}")
        print(f"  Val RMSE:   {cv['rmse_mean']:.2f}")
        print(f"  Gap:        {cv['overfitting_gap_rmse']:.2f}")
        print(f"  Status:     {report['summary']['overfitting_status']}")

        if 'test_set' in report:
            test = report['test_set']['metrics']
            print(f"\nTest Set Performance:")
            print(f"  RMSE:  {test['rmse']:.2f}")
            print(f"  MAE:   {test['mae']:.2f}")
            print(f"  R²:    {test['r2']:.4f}")
            print(f"  MAPE:  {test['mape']:.2f}%")
            print(f"\nPrediction Accuracy:")
            print(f"  Within 5%:  {test['pct_within_5pct']:.1f}%")
            print(f"  Within 10%: {test['pct_within_10pct']:.1f}%")
            print(f"  Within 20%: {test['pct_within_20pct']:.1f}%")

        print(f"\nRecommendation:")
        print(f"  {report['summary']['recommendation']}")

        print("="*70 + "\n")
