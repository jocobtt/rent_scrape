"""
Enhanced Prediction Service with Ensemble Model Support

This service extends the original prediction service to include ensemble methods
for combining multiple models and provides advanced prediction capabilities.
"""

import logging
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Any, List, Optional

from .prediction_service import PredictionService
from models.ensemble_model import EnsembleModel

logger = logging.getLogger(__name__)


class EnsemblePredictionService(PredictionService):
    """Enhanced prediction service with ensemble model capabilities."""
    
    def __init__(self, passed_model, challenger_model, ensemble_models: Dict[str, EnsembleModel] = None):
        """
        Initialize the ensemble prediction service.
        
        Args:
            passed_model: The passed (production) model
            challenger_model: The challenger model
            ensemble_models: Dictionary of ensemble models by method name
        """
        super().__init__(passed_model, challenger_model)
        self.ensemble_models = ensemble_models or {}
        self._load_ensemble_models()
    
    def _load_ensemble_models(self):
        """Load pre-trained ensemble models from disk."""
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        ensemble_methods = ['weighted_average', 'stacking', 'voting', 'dynamic_weights']
        
        for method in ensemble_methods:
            ensemble_path = os.path.join(models_dir, f'ensemble_{method}_model.joblib')
            if os.path.exists(ensemble_path):
                try:
                    ensemble = EnsembleModel(method=method)
                    base_models = {
                        'challenger': self.challenger_model,
                        'passed': self.passed_model
                    }
                    ensemble.load(ensemble_path, base_models)
                    self.ensemble_models[method] = ensemble
                    logger.info(f"Loaded {method} ensemble model")
                except Exception as e:
                    logger.warning(f"Failed to load {method} ensemble: {e}")
    
    def predict_with_ensemble(self, input_data: Dict[str, float], method: str = 'weighted_average') -> Dict[str, Any]:
        """
        Make prediction using ensemble model.
        
        Args:
            input_data: Dictionary containing apartment features
            method: Ensemble method to use
            
        Returns:
            Dict containing prediction and metadata
        """
        try:
            if method not in self.ensemble_models:
                raise ValueError(f"Ensemble method '{method}' not available. Available methods: {list(self.ensemble_models.keys())}")
            
            ensemble = self.ensemble_models[method]
            
            # Convert input to DataFrame (handle both formats)
            if self._is_challenger_format(input_data):
                # Direct feature array format for challenger
                features_df = pd.DataFrame([{
                    'rei_price': input_data['rei_price'],
                    'shikikin': input_data['shikikin'],
                    'maintenence_price': input_data['maintenence_price'],
                    'year_built': input_data['year_built'],
                    'floor': input_data['floor'],
                    'eki_walk': input_data['eki_walk'],
                    'sqr_m': input_data['sqr_m']
                }])
            else:
                # Full feature format for passed model
                features_df = pd.DataFrame([input_data])
            
            prediction = ensemble.predict(features_df)
            
            return {
                "prediction": float(prediction[0]) if isinstance(prediction, np.ndarray) else float(prediction),
                "model_type": f"ensemble_{method}",
                "ensemble_weights": ensemble.weights if hasattr(ensemble, 'weights') else None,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Ensemble prediction error ({method}): {str(e)}")
            raise
    
    def _is_challenger_format(self, input_data: Dict[str, float]) -> bool:
        """Check if input data is in challenger model format (7 features)."""
        challenger_features = ['rei_price', 'shikikin', 'maintenence_price', 'year_built', 'floor', 'eki_walk', 'sqr_m']
        return all(feature in input_data for feature in challenger_features) and len(input_data) == len(challenger_features)
    
    def compare_all_predictions(self, input_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare predictions from all available models including ensembles.
        
        Args:
            input_data: Dictionary containing apartment features
            
        Returns:
            Dict containing predictions from all models and comparison metrics
        """
        try:
            results = {}
            
            # Get base model predictions
            base_results = super().compare_predictions(input_data)
            results.update(base_results)
            
            # Get ensemble predictions
            ensemble_predictions = {}
            for method in self.ensemble_models.keys():
                try:
                    ensemble_result = self.predict_with_ensemble(input_data, method)
                    ensemble_predictions[f"ensemble_{method}"] = ensemble_result["prediction"]
                except Exception as e:
                    logger.warning(f"Failed to get {method} ensemble prediction: {e}")
                    continue
            
            results["ensemble_predictions"] = ensemble_predictions
            
            # Calculate ensemble statistics
            if ensemble_predictions:
                ensemble_values = list(ensemble_predictions.values())
                results["ensemble_statistics"] = {
                    "mean": float(np.mean(ensemble_values)),
                    "std": float(np.std(ensemble_values)),
                    "min": float(np.min(ensemble_values)),
                    "max": float(np.max(ensemble_values)),
                    "median": float(np.median(ensemble_values))
                }
                
                # Find best and worst performing ensemble relative to base models
                passed_pred = base_results["predictions"]["passed_model"]
                challenger_pred = base_results["predictions"]["challenger_model"]
                base_mean = (passed_pred + challenger_pred) / 2
                
                ensemble_distances = {
                    method: abs(pred - base_mean) 
                    for method, pred in ensemble_predictions.items()
                }
                
                results["ensemble_analysis"] = {
                    "closest_to_base_mean": min(ensemble_distances, key=ensemble_distances.get),
                    "farthest_from_base_mean": max(ensemble_distances, key=ensemble_distances.get),
                    "consensus_prediction": results["ensemble_statistics"]["median"]
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error comparing all predictions: {str(e)}")
            raise
    
    def get_prediction_confidence(self, input_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate prediction confidence based on model agreement.
        
        Args:
            input_data: Dictionary containing apartment features
            
        Returns:
            Dict containing confidence metrics
        """
        try:
            all_predictions = self.compare_all_predictions(input_data)
            
            # Collect all predictions
            predictions = []
            predictions.append(all_predictions["predictions"]["passed_model"])
            predictions.append(all_predictions["predictions"]["challenger_model"])
            
            if "ensemble_predictions" in all_predictions:
                predictions.extend(all_predictions["ensemble_predictions"].values())
            
            predictions = np.array(predictions)
            
            # Calculate confidence metrics
            std_dev = np.std(predictions)
            cv = std_dev / np.mean(predictions) if np.mean(predictions) != 0 else float('inf')
            
            # Define confidence levels based on coefficient of variation
            if cv < 0.05:
                confidence_level = "very_high"
            elif cv < 0.10:
                confidence_level = "high"
            elif cv < 0.20:
                confidence_level = "medium"
            elif cv < 0.35:
                confidence_level = "low"
            else:
                confidence_level = "very_low"
            
            return {
                "confidence_level": confidence_level,
                "coefficient_of_variation": float(cv),
                "standard_deviation": float(std_dev),
                "prediction_range": {
                    "min": float(np.min(predictions)),
                    "max": float(np.max(predictions)),
                    "mean": float(np.mean(predictions))
                },
                "model_agreement": {
                    "total_models": len(predictions),
                    "within_5_percent": int(np.sum(np.abs(predictions - np.mean(predictions)) / np.mean(predictions) <= 0.05)),
                    "within_10_percent": int(np.sum(np.abs(predictions - np.mean(predictions)) / np.mean(predictions) <= 0.10))
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating prediction confidence: {str(e)}")
            raise
    
    def get_model_recommendations(self, input_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Provide model recommendations based on input characteristics.
        
        Args:
            input_data: Dictionary containing apartment features
            
        Returns:
            Dict containing model recommendations and reasoning
        """
        try:
            # Analyze input characteristics
            rent_price = input_data.get('rei_price', 0)
            apartment_size = input_data.get('sqr_m', 0)
            year_built = input_data.get('year_built', 2000)
            
            recommendations = []
            
            # Recommend based on property characteristics
            if rent_price > 150000:  # High-end properties
                recommendations.append({
                    "model": "ensemble_stacking",
                    "reason": "Stacking ensemble performs well on high-value properties with complex patterns",
                    "confidence": "high"
                })
            
            if apartment_size > 60:  # Large apartments
                recommendations.append({
                    "model": "challenger_model",
                    "reason": "LightGBM handles large apartment pricing patterns effectively",
                    "confidence": "medium"
                })
            
            if year_built < 1990:  # Older properties
                recommendations.append({
                    "model": "ensemble_weighted_average",
                    "reason": "Weighted ensemble balances linear and non-linear patterns for older properties",
                    "confidence": "medium"
                })
            
            # Default recommendation
            if not recommendations:
                recommendations.append({
                    "model": "ensemble_weighted_average",
                    "reason": "Weighted ensemble provides reliable predictions for typical properties",
                    "confidence": "high"
                })
            
            # Get confidence analysis
            confidence_analysis = self.get_prediction_confidence(input_data)
            
            return {
                "recommended_models": recommendations,
                "input_analysis": {
                    "rent_category": "high" if rent_price > 150000 else "medium" if rent_price > 80000 else "low",
                    "size_category": "large" if apartment_size > 60 else "medium" if apartment_size > 30 else "small",
                    "age_category": "new" if year_built > 2010 else "medium" if year_built > 1990 else "old"
                },
                "confidence_analysis": confidence_analysis
            }
            
        except Exception as e:
            logger.error(f"Error generating model recommendations: {str(e)}")
            raise
    
    def batch_predict(self, input_data_list: List[Dict[str, float]], method: str = 'ensemble_weighted_average') -> List[Dict[str, Any]]:
        """
        Make batch predictions using specified method.
        
        Args:
            input_data_list: List of input dictionaries
            method: Prediction method to use
            
        Returns:
            List of prediction results
        """
        results = []
        
        for input_data in input_data_list:
            try:
                if method.startswith('ensemble_'):
                    ensemble_method = method.replace('ensemble_', '')
                    result = self.predict_with_ensemble(input_data, ensemble_method)
                elif method == 'passed':
                    result = self.predict_with_passed_model(input_data)
                elif method == 'challenger':
                    result = self.predict_with_challenger_model(input_data)
                else:
                    result = self.compare_all_predictions(input_data)
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    "error": str(e),
                    "status": "failed",
                    "input_data": input_data
                })
        
        return results
    
    def get_available_methods(self) -> Dict[str, Any]:
        """Get information about all available prediction methods."""
        return {
            "base_models": ["passed", "challenger"],
            "ensemble_methods": list(self.ensemble_models.keys()),
            "total_methods": 2 + len(self.ensemble_models),
            "ensemble_details": {
                method: {
                    "weights": ensemble.weights if hasattr(ensemble, 'weights') else None,
                    "method_type": ensemble.method
                }
                for method, ensemble in self.ensemble_models.items()
            }
        }