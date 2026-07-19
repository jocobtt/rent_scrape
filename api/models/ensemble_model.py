"""
Ensemble model implementation for combining multiple ML models.

This module provides various ensemble methods to combine predictions from
the challenger (LightGBM) and passed (Linear) models for improved performance.
"""

import os
import numpy as np
import pandas as pd
import joblib
import mlflow
import logging
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("s3_path")


class EnsembleModel:
    """
    Ensemble model class that combines multiple base models using various strategies.
    """
    
    def __init__(self, method='weighted_average'):
        """
        Initialize ensemble model.
        
        Args:
            method: Ensemble method ('weighted_average', 'stacking', 'voting', 'dynamic_weights')
        """
        self.method = method
        self.base_models = {}
        self.weights = {}
        self.meta_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def add_model(self, name: str, model, weight: float = 1.0):
        """Add a base model to the ensemble."""
        self.base_models[name] = model
        self.weights[name] = weight
        
    def _weighted_average_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using weighted average ensemble."""
        predictions = []
        total_weight = sum(self.weights.values())
        
        for name, model in self.base_models.items():
            pred = model.predict(X)
            weighted_pred = pred * (self.weights[name] / total_weight)
            predictions.append(weighted_pred)
            
        return np.sum(predictions, axis=0)
    
    def _stacking_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using stacking ensemble."""
        # Generate base model predictions
        base_predictions = np.column_stack([
            model.predict(X) for model in self.base_models.values()
        ])
        
        # Use meta-model to make final prediction
        if self.meta_model is None:
            raise ValueError("Meta-model not trained for stacking")
            
        return self.meta_model.predict(base_predictions)
    
    def _voting_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using voting ensemble."""
        predictions = [model.predict(X) for model in self.base_models.values()]
        return np.mean(predictions, axis=0)
    
    def _dynamic_weights_predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using dynamic weights based on feature characteristics.
        Different models might perform better on different types of input.
        """
        predictions = []
        
        for i, row in X.iterrows():
            model_predictions = []
            confidence_weights = []
            
            for name, model in self.base_models.items():
                pred = model.predict(row.to_frame().T)[0]
                
                # Calculate confidence based on feature characteristics
                # Higher rent values might favor one model over another
                rent_factor = row.get('rei_price', 0)
                size_factor = row.get('sqr_m', 0)
                
                if name == 'challenger':  # LightGBM - better for complex patterns
                    confidence = 0.6 if rent_factor > 100000 else 0.4
                    confidence += 0.1 if size_factor > 50 else -0.1
                else:  # Linear model - better for simpler patterns
                    confidence = 0.4 if rent_factor > 100000 else 0.6
                    confidence += -0.1 if size_factor > 50 else 0.1
                
                model_predictions.append(pred)
                confidence_weights.append(max(0.1, min(0.9, confidence)))
            
            # Normalize weights
            total_weight = sum(confidence_weights)
            normalized_weights = [w / total_weight for w in confidence_weights]
            
            # Calculate weighted prediction
            weighted_pred = sum(p * w for p, w in zip(model_predictions, normalized_weights))
            predictions.append(weighted_pred)
            
        return np.array(predictions)
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """
        Train the ensemble model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (for stacking)
            y_val: Validation targets (for stacking)
        """
        if self.method == 'stacking':
            if X_val is None or y_val is None:
                # Use cross-validation to generate meta-features
                from sklearn.model_selection import KFold
                
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                meta_features = np.zeros((len(X_train), len(self.base_models)))
                
                for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                    X_fold_train = X_train.iloc[train_idx]
                    y_fold_train = y_train.iloc[train_idx]
                    X_fold_val = X_train.iloc[val_idx]
                    
                    for i, (name, model) in enumerate(self.base_models.items()):
                        # Clone and fit model on fold
                        fold_model = self._clone_model(model)
                        fold_model.fit(X_fold_train, y_fold_train)
                        
                        # Predict on validation fold
                        meta_features[val_idx, i] = fold_model.predict(X_fold_val)
                
                # Train meta-model
                self.meta_model = Ridge(alpha=1.0)
                self.meta_model.fit(meta_features, y_train)
            else:
                # Use provided validation set
                base_predictions = np.column_stack([
                    model.predict(X_val) for model in self.base_models.values()
                ])
                
                self.meta_model = Ridge(alpha=1.0)
                self.meta_model.fit(base_predictions, y_val)
        
        # Optimize weights for weighted average method
        if self.method == 'weighted_average':
            self._optimize_weights(X_train, y_train)
            
        self.is_trained = True
        
    def _clone_model(self, model):
        """Create a clone of the model."""
        if hasattr(model, 'get_params'):
            # sklearn-like model
            from sklearn.base import clone
            return clone(model)
        else:
            # For joblib-loaded models, this is trickier
            # For now, return the same model (not ideal for CV)
            return model
            
    def _optimize_weights(self, X: pd.DataFrame, y: pd.Series):
        """Optimize weights using grid search."""
        from sklearn.model_selection import ParameterGrid
        
        best_score = float('inf')
        best_weights = self.weights.copy()
        
        # Create parameter grid for weights
        weight_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        model_names = list(self.base_models.keys())
        
        if len(model_names) == 2:
            for w1 in weight_options:
                w2 = 1.0 - w1
                test_weights = {model_names[0]: w1, model_names[1]: w2}
                
                # Temporarily set weights
                old_weights = self.weights.copy()
                self.weights = test_weights
                
                # Evaluate
                predictions = self._weighted_average_predict(X)
                score = mean_squared_error(y, predictions)
                
                if score < best_score:
                    best_score = score
                    best_weights = test_weights.copy()
                
                # Restore old weights
                self.weights = old_weights
        
        self.weights = best_weights
        logger.info(f"Optimized weights: {self.weights}")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the ensemble."""
        if not self.is_trained:
            raise ValueError("Ensemble model must be trained before making predictions")
            
        if self.method == 'weighted_average':
            return self._weighted_average_predict(X)
        elif self.method == 'stacking':
            return self._stacking_predict(X)
        elif self.method == 'voting':
            return self._voting_predict(X)
        elif self.method == 'dynamic_weights':
            return self._dynamic_weights_predict(X)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate the ensemble model."""
        predictions = self.predict(X_test)
        
        return {
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions),
            'r2': r2_score(y_test, predictions),
            'mse': mean_squared_error(y_test, predictions)
        }
    
    def save(self, filepath: str):
        """Save the ensemble model."""
        ensemble_data = {
            'method': self.method,
            'weights': self.weights,
            'meta_model': self.meta_model,
            'is_trained': self.is_trained
        }
        joblib.dump(ensemble_data, filepath)
        
    def load(self, filepath: str, base_models: Dict):
        """Load the ensemble model."""
        ensemble_data = joblib.load(filepath)
        self.method = ensemble_data['method']
        self.weights = ensemble_data['weights']
        self.meta_model = ensemble_data['meta_model']
        self.is_trained = ensemble_data['is_trained']
        self.base_models = base_models


def load_data_for_ensemble():
    """Load and prepare data for ensemble training."""
    try:
        from datasets import load_dataset
        data = load_dataset("jbrazzy/tokyo_rent", split="train")
        df = data.to_pandas()
    except:
        # Fallback to local data
        df = pd.read_csv("../data/tokyo_model.csv")
    
    # Preprocessing (same as individual models)
    df['ku_name'] = df['ku_name'].astype('category')
    df = pd.get_dummies(df, columns=['ku_name'])
    df['apartment_type'] = df['apartment_type'].astype('category')
    df = pd.get_dummies(df, columns=['apartment_type'])
    df['house_type'] = df['house_type'].astype('category')
    df = pd.get_dummies(df, columns=['house_type'])
    
    if 'address' in df.columns:
        df = df.drop(['address'], axis=1)
    
    X = df.drop('rent_price', axis=1)
    y = df['rent_price']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_ensemble_models(ensemble_methods: List[str] = None):
    """
    Train multiple ensemble models and compare their performance.
    
    Args:
        ensemble_methods: List of ensemble methods to train
        
    Returns:
        Dict containing trained ensemble models and their performance
    """
    if ensemble_methods is None:
        ensemble_methods = ['weighted_average', 'stacking', 'voting', 'dynamic_weights']
    
    # Load data
    X_train, X_test, y_train, y_test = load_data_for_ensemble()
    
    # Load base models
    models_dir = os.path.join(os.path.dirname(__file__), '..')
    challenger_model = joblib.load(os.path.join(models_dir, 'challenger-model.joblib'))
    passed_model = joblib.load(os.path.join(models_dir, 'passed-model.joblib'))
    
    results = {}
    
    with mlflow.start_run(run_name="ensemble_comparison") as run:
        mlflow.log_param("ensemble_methods", ensemble_methods)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        
        for method in ensemble_methods:
            print(f"\n{'='*60}")
            print(f"Training {method.upper()} Ensemble")
            print(f"{'='*60}")
            
            # Create ensemble
            ensemble = EnsembleModel(method=method)
            ensemble.add_model('challenger', challenger_model, weight=0.6)
            ensemble.add_model('passed', passed_model, weight=0.4)
            
            # Train ensemble
            try:
                ensemble.fit(X_train, y_train, X_test, y_test)
                
                # Evaluate ensemble
                metrics = ensemble.evaluate(X_test, y_test)
                
                # Evaluate individual models for comparison
                challenger_pred = challenger_model.predict(X_test)
                passed_pred = passed_model.predict(X_test)
                
                challenger_rmse = np.sqrt(mean_squared_error(y_test, challenger_pred))
                passed_rmse = np.sqrt(mean_squared_error(y_test, passed_pred))
                
                print(f"\n{method.upper()} Results:")
                print(f"  Ensemble RMSE: {metrics['rmse']:.2f}")
                print(f"  Challenger RMSE: {challenger_rmse:.2f}")
                print(f"  Passed RMSE: {passed_rmse:.2f}")
                print(f"  Ensemble R²: {metrics['r2']:.4f}")
                
                # Log metrics to MLflow
                with mlflow.start_run(run_name=f"ensemble_{method}", nested=True):
                    mlflow.log_params({
                        "method": method,
                        "base_models": "challenger,passed",
                        "weights": str(ensemble.weights)
                    })
                    
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(f"ensemble_{metric_name}", value)
                    
                    mlflow.log_metric("challenger_rmse", challenger_rmse)
                    mlflow.log_metric("passed_rmse", passed_rmse)
                    mlflow.log_metric("improvement_vs_challenger", challenger_rmse - metrics['rmse'])
                    mlflow.log_metric("improvement_vs_passed", passed_rmse - metrics['rmse'])
                    
                    # Save ensemble model
                    ensemble_path = f"ensemble_{method}_model.joblib"
                    ensemble.save(ensemble_path)
                    mlflow.log_artifact(ensemble_path)
                
                results[method] = {
                    'model': ensemble,
                    'metrics': metrics,
                    'improvement_vs_best_base': min(challenger_rmse, passed_rmse) - metrics['rmse']
                }
                
            except Exception as e:
                logger.error(f"Error training {method} ensemble: {e}")
                continue
        
        # Find best ensemble method
        if results:
            best_method = min(results.keys(), key=lambda k: results[k]['metrics']['rmse'])
            best_ensemble = results[best_method]
            
            print(f"\n{'='*60}")
            print(f"BEST ENSEMBLE: {best_method.upper()}")
            print(f"RMSE: {best_ensemble['metrics']['rmse']:.2f}")
            print(f"Improvement: {best_ensemble['improvement_vs_best_base']:.2f}")
            print(f"{'='*60}")
            
            mlflow.log_metric("best_ensemble_rmse", best_ensemble['metrics']['rmse'])
            mlflow.log_param("best_method", best_method)
    
    return results


if __name__ == "__main__":
    # Train all ensemble methods
    results = train_ensemble_models()
    
    # Print summary
    print("\nEnsemble Model Training Complete!")
    for method, result in results.items():
        metrics = result['metrics']
        print(f"{method}: RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.4f}")