"""
Enhanced Prediction Service with Ensemble Model Support

This service extends the original prediction service to include ensemble methods
for combining multiple models and provides advanced prediction capabilities.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional

from .prediction_service import PredictionService
from models.ensemble_model import METHODS, EnsembleModel, ensemble_path

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
        """Load the trained ensembles (models/ensemble_<method>_model.joblib) that exist."""
        for method in METHODS:
            path = ensemble_path(method)
            if not path.exists():
                continue
            try:
                self.ensemble_models[method] = EnsembleModel.load(path)
                logger.info(f"Loaded {method} ensemble model")
            except Exception as e:
                logger.warning(f"Failed to load {method} ensemble: {e}")

    def _base_predictions(self, input_data: Dict[str, float]) -> Dict[str, float]:
        """What each base model predicts for this input (shared models: deposit / key money given)."""
        return {
            "challenger": self.predict_with_challenger_model(input_data)["prediction"],
            "passed": self.predict_with_passed_model(input_data)["prediction"],
        }

    def predict_with_ensemble(self, input_data: Dict[str, float], method: str = 'weighted_average',
                              base: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Make prediction using ensemble model.

        Args:
            input_data: Dictionary containing apartment features
            method: Ensemble method to use
            base: base-model predictions if the caller already has them

        Returns:
            Dict containing prediction and metadata
        """
        try:
            if method not in self.ensemble_models:
                raise ValueError(f"Ensemble method '{method}' not available. Available methods: {list(self.ensemble_models.keys())}")

            ensemble = self.ensemble_models[method]
            base = base or self._base_predictions(input_data)
            prediction = ensemble.combine({name: np.array([value]) for name, value in base.items()})

            return {
                "prediction": float(prediction[0]),
                "model_type": f"ensemble_{method}",
                "ensemble_weights": ensemble.describe()["weights"],
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Ensemble prediction error ({method}): {str(e)}")
            raise

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
            
            # Get ensemble predictions (the base models are asked once, not once per method)
            ensemble_predictions = {}
            base = {"challenger": base_results["predictions"]["challenger_model"],
                    "passed": base_results["predictions"]["passed_model"]}
            for method in self.ensemble_models.keys():
                try:
                    ensemble_result = self.predict_with_ensemble(input_data, method, base=base)
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
    
    def get_prediction_confidence(self, input_data: Dict[str, float],
                                  extra_predictions: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Calculate prediction confidence based on model agreement.
        
        Args:
            input_data: Dictionary containing apartment features
            extra_predictions: other models' predictions (the deep models) to include in the agreement
            
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
            
            if extra_predictions:
                predictions.extend(extra_predictions.values())

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
    
    def get_model_recommendations(self, input_data: Dict[str, float],
                                  extra_predictions: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Recommend a model from how each performed on buildings it was not fitted on.

        Args:
            input_data: Dictionary containing apartment features

        Returns:
            Dict containing model recommendations and reasoning
        """
        try:
            confidence_analysis = self.get_prediction_confidence(input_data, extra_predictions)
            consensus = confidence_analysis["prediction_range"]["mean"]  # 万円
            size = input_data.get('sqr_m', 0)
            age = input_data.get('year_built', 0)  # years since construction

            scored = sorted(
                ((m.meta.get("eval_rmse"), name) for name, m in self.ensemble_models.items()
                 if m.meta.get("eval_rmse") is not None),
            )
            recommendations = []
            if scored:
                best_rmse, best = scored[0]
                recommendations.append({
                    "model": f"ensemble_{best}",
                    "reason": f"Lowest held-out error of the ensembles (RMSE {best_rmse:.2f} 万円, "
                              f"against the linear and LightGBM models it combines)",
                    "confidence": "high" if confidence_analysis["confidence_level"] in ("very_high", "high") else "medium",
                })
                recommendations += [
                    {"model": f"ensemble_{name}", "reason": f"Held-out RMSE {rmse:.2f} 万円", "confidence": "medium"}
                    for rmse, name in scored[1:]
                ]
            else:
                recommendations.append({
                    "model": "challenger_model",
                    "reason": "No ensembles are trained yet; LightGBM is the strongest single model",
                    "confidence": "medium",
                })

            return {
                "recommended_models": recommendations,
                "input_analysis": {
                    "rent_category": "high" if consensus > 25 else "medium" if consensus > 12 else "low",
                    "size_category": "large" if size > 60 else "medium" if size > 30 else "small",
                    "age_category": "new" if age < 5 else "medium" if age < 25 else "old",
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
                method: ensemble.describe()
                for method, ensemble in self.ensemble_models.items()
            }
        }