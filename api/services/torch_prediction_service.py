"""
PyTorch Prediction Service for Neural Network Models

This service extends the prediction capabilities to include PyTorch models
and integrates with the existing API infrastructure.
"""

import logging
import torch
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union

from .prediction_service import PredictionService
from models.torch_model import TorchRentPredictor
from models.transformer_model import TransformerRentPredictor

logger = logging.getLogger(__name__)


class PyTorchPredictionService(PredictionService):
    """Enhanced prediction service with PyTorch model support."""
    
    def __init__(self, passed_model, challenger_model, torch_models: Dict = None):
        """
        Initialize PyTorch prediction service.
        
        Args:
            passed_model: Standard passed model (sklearn)
            challenger_model: Standard challenger model (sklearn) 
            torch_models: Dictionary of PyTorch models by name
        """
        super().__init__(passed_model, challenger_model)
        self.torch_models = torch_models or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initialized PyTorchPredictionService with {len(self.torch_models)} PyTorch models")
    
    def add_torch_model(self, name: str, model: Union[TorchRentPredictor, TransformerRentPredictor]):
        """Add a PyTorch model to the service."""
        self.torch_models[name] = model
        logger.info(f"Added PyTorch model: {name}")
    
    def predict_with_torch_model(self, input_data: Dict[str, float], 
                                model_name: str) -> Dict[str, Any]:
        """
        Make prediction using a specific PyTorch model.
        
        Args:
            input_data: Dictionary containing apartment features
            model_name: Name of the PyTorch model to use
            
        Returns:
            Dict containing prediction and metadata
        """
        try:
            if model_name not in self.torch_models:
                available_models = list(self.torch_models.keys())
                raise ValueError(f"PyTorch model '{model_name}' not available. "
                               f"Available models: {available_models}")
            
            model = self.torch_models[model_name]
            
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            
            # Handle different input formats for different models
            if hasattr(model, 'predict_single'):
                # Custom prediction method
                prediction = model.predict_single(input_data)
            elif hasattr(model, 'model') and model.model is not None:
                # PyTorch model with proper predictor wrapper
                prediction = self._predict_with_torch_predictor(model, input_df)
            else:
                # Raw PyTorch model
                prediction = self._predict_with_raw_torch_model(model, input_df)
            
            return {
                "prediction": float(prediction),
                "model_type": f"torch_{model_name}",
                "framework": "PyTorch",
                "device": str(self.device),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"PyTorch prediction error ({model_name}): {str(e)}")
            raise
    
    def _predict_with_torch_predictor(self, predictor, input_df: pd.DataFrame) -> float:
        """Predict using a TorchRentPredictor or TransformerRentPredictor."""
        try:
            # Use the predictor's evaluation method
            if hasattr(predictor, 'predict_dataframe'):
                return predictor.predict_dataframe(input_df)[0]
            else:
                # Manually prepare data and predict
                from models.torch_model import RentDataset
                
                # Create dataset
                dataset = RentDataset(
                    input_df, pd.Series([0]), # Dummy target
                    predictor.numerical_cols, 
                    predictor.categorical_cols,
                    scaler=predictor.scaler,
                    fit_scaler=False
                )
                
                # Get features
                features = torch.tensor(dataset.all_features[0]).unsqueeze(0).to(self.device)
                
                # Predict
                predictor.model.eval()
                with torch.no_grad():
                    if predictor.model_type == "hybrid":
                        num_features = features[:, :len(predictor.numerical_cols)]
                        cat_features = features[:, len(predictor.numerical_cols):]
                        output = predictor.model(num_features, cat_features)
                    else:
                        output = predictor.model(features)
                
                return output.cpu().numpy()[0]
                
        except Exception as e:
            logger.error(f"Error in PyTorch predictor: {e}")
            raise
    
    def _predict_with_raw_torch_model(self, model, input_df: pd.DataFrame) -> float:
        """Predict using a raw PyTorch model (fallback method)."""
        try:
            # This is a fallback for raw torch models without proper wrapper
            # Convert input to tensor (simplified approach)
            
            # Get numerical columns (basic assumption)
            numerical_cols = ['rei_price', 'shikikin', 'maintenence_price', 
                            'year_built', 'floor', 'eki_walk', 'sqr_m']
            
            # Extract available numerical features
            features = []
            for col in numerical_cols:
                if col in input_df.columns:
                    features.append(input_df[col].iloc[0])
                else:
                    features.append(0.0)  # Default value
            
            # Convert to tensor
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Predict
            model.eval()
            with torch.no_grad():
                output = model(features_tensor)
                
            return output.cpu().numpy()[0]
            
        except Exception as e:
            logger.error(f"Error in raw PyTorch model prediction: {e}")
            raise
    
    def compare_all_predictions_with_torch(self, input_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare predictions from all models including PyTorch models.
        
        Args:
            input_data: Dictionary containing apartment features
            
        Returns:
            Dict containing predictions from all models
        """
        try:
            # Get base model predictions
            base_results = super().compare_predictions(input_data)
            
            # Get PyTorch predictions
            torch_predictions = {}
            for model_name in self.torch_models.keys():
                try:
                    torch_result = self.predict_with_torch_model(input_data, model_name)
                    torch_predictions[model_name] = torch_result["prediction"]
                except Exception as e:
                    logger.warning(f"Failed to get {model_name} prediction: {e}")
                    torch_predictions[model_name] = None
            
            # Add PyTorch predictions to results
            base_results["torch_predictions"] = torch_predictions
            
            # Calculate comprehensive statistics
            all_predictions = []
            
            # Add base model predictions
            if "predictions" in base_results:
                all_predictions.extend([
                    base_results["predictions"]["passed_model"],
                    base_results["predictions"]["challenger_model"]
                ])
            
            # Add successful PyTorch predictions
            successful_torch_preds = [pred for pred in torch_predictions.values() if pred is not None]
            all_predictions.extend(successful_torch_preds)
            
            if all_predictions:
                all_predictions = np.array(all_predictions)
                
                base_results["comprehensive_statistics"] = {
                    "mean": float(np.mean(all_predictions)),
                    "std": float(np.std(all_predictions)),
                    "min": float(np.min(all_predictions)),
                    "max": float(np.max(all_predictions)),
                    "median": float(np.median(all_predictions)),
                    "total_models": len(all_predictions),
                    "torch_models_successful": len(successful_torch_preds)
                }
            
            return base_results
            
        except Exception as e:
            logger.error(f"Error comparing all predictions: {str(e)}")
            raise
    
    def get_torch_model_recommendations(self, input_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Get recommendations for which PyTorch model to use based on input characteristics.
        
        Args:
            input_data: Dictionary containing apartment features
            
        Returns:
            Dict containing model recommendations
        """
        try:
            recommendations = []
            
            # Analyze input characteristics
            rent_price = input_data.get('rei_price', 0)
            apartment_size = input_data.get('sqr_m', 0)
            year_built = input_data.get('year_built', 2000)
            floor_level = input_data.get('floor', 1)
            
            # Recommend based on data complexity
            if len(self.torch_models) == 0:
                return {
                    "recommendations": [],
                    "message": "No PyTorch models available"
                }
            
            # High-value, complex properties
            if rent_price > 200000 and apartment_size > 80:
                if "ensemble" in self.torch_models:
                    recommendations.append({
                        "model": "ensemble",
                        "reason": "Ensemble model best for high-value, large properties with complex pricing patterns",
                        "confidence": "high",
                        "priority": 1
                    })
                
                if "transformer" in self.torch_models or any("bert" in name.lower() for name in self.torch_models):
                    transformer_models = [name for name in self.torch_models 
                                        if "transformer" in name or "bert" in name.lower()]
                    if transformer_models:
                        recommendations.append({
                            "model": transformer_models[0],
                            "reason": "Transformer models capture complex feature interactions for premium properties",
                            "confidence": "medium",
                            "priority": 2
                        })
            
            # Standard properties
            elif rent_price < 150000 and apartment_size < 60:
                if "tabular" in self.torch_models:
                    recommendations.append({
                        "model": "tabular",
                        "reason": "Tabular neural network efficient for standard residential properties",
                        "confidence": "high",
                        "priority": 1
                    })
            
            # Modern, mid-range properties
            elif year_built > 2010:
                attention_models = [name for name in self.torch_models 
                                  if "attention" in name.lower() or "tabular" in name]
                if attention_models:
                    recommendations.append({
                        "model": attention_models[0], 
                        "reason": "Attention mechanisms effective for modern properties with diverse features",
                        "confidence": "medium",
                        "priority": 1
                    })
            
            # Default recommendation
            if not recommendations and self.torch_models:
                default_model = list(self.torch_models.keys())[0]
                recommendations.append({
                    "model": default_model,
                    "reason": f"Default PyTorch model for general rent prediction",
                    "confidence": "medium",
                    "priority": 1
                })
            
            # Sort by priority
            recommendations.sort(key=lambda x: x["priority"])
            
            return {
                "recommendations": recommendations,
                "available_models": list(self.torch_models.keys()),
                "input_analysis": {
                    "rent_category": "high" if rent_price > 150000 else "medium" if rent_price > 80000 else "low",
                    "size_category": "large" if apartment_size > 60 else "medium" if apartment_size > 30 else "small",
                    "age_category": "new" if year_built > 2010 else "medium" if year_built > 1990 else "old",
                    "complexity_score": self._calculate_complexity_score(input_data)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating PyTorch model recommendations: {str(e)}")
            raise
    
    def _calculate_complexity_score(self, input_data: Dict[str, float]) -> float:
        """Calculate a complexity score for the input to help with model selection."""
        try:
            score = 0.0
            
            # High rent increases complexity
            rent_price = input_data.get('rei_price', 0)
            if rent_price > 200000:
                score += 0.3
            elif rent_price > 150000:
                score += 0.2
            elif rent_price > 100000:
                score += 0.1
            
            # Large size increases complexity  
            size = input_data.get('sqr_m', 0)
            if size > 80:
                score += 0.3
            elif size > 60:
                score += 0.2
            elif size > 40:
                score += 0.1
            
            # High floors can be complex
            floor = input_data.get('floor', 1)
            if floor > 15:
                score += 0.2
            elif floor > 10:
                score += 0.1
            
            # Very new or very old buildings are complex
            year_built = input_data.get('year_built', 2000)
            current_year = 2026
            age = current_year - year_built
            if age < 5 or age > 40:
                score += 0.2
            
            return min(score, 1.0)  # Cap at 1.0
            
        except Exception:
            return 0.5  # Default medium complexity
    
    def batch_predict_torch(self, input_data_list: List[Dict[str, float]], 
                           model_name: str) -> List[Dict[str, Any]]:
        """
        Make batch predictions using a PyTorch model.
        
        Args:
            input_data_list: List of input dictionaries
            model_name: Name of PyTorch model to use
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i, input_data in enumerate(input_data_list):
            try:
                result = self.predict_with_torch_model(input_data, model_name)
                result["batch_index"] = i
                results.append(result)
                
            except Exception as e:
                results.append({
                    "batch_index": i,
                    "error": str(e),
                    "status": "failed",
                    "input_data": input_data
                })
        
        return results
    
    def get_available_torch_models(self) -> Dict[str, Any]:
        """Get information about available PyTorch models."""
        model_info = {}
        
        for name, model in self.torch_models.items():
            try:
                info = {
                    "name": name,
                    "type": getattr(model, 'model_type', 'unknown'),
                    "framework": "PyTorch",
                    "device": str(getattr(model, 'device', 'unknown')),
                    "has_model": hasattr(model, 'model') and model.model is not None,
                    "ready_for_inference": hasattr(model, 'model') and hasattr(model, 'scaler')
                }
                
                if hasattr(model, 'model') and model.model is not None:
                    # Count parameters
                    total_params = sum(p.numel() for p in model.model.parameters())
                    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
                    
                    info.update({
                        "total_parameters": total_params,
                        "trainable_parameters": trainable_params,
                        "model_size_mb": total_params * 4 / (1024 * 1024)  # Approximate size in MB
                    })
                
                model_info[name] = info
                
            except Exception as e:
                model_info[name] = {
                    "name": name,
                    "error": str(e),
                    "status": "error"
                }
        
        return {
            "models": model_info,
            "total_models": len(self.torch_models),
            "device": str(self.device)
        }