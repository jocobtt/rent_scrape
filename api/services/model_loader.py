"""
Enhanced model loading service for managing various model artifacts including PyTorch models.
"""

import os
import joblib
import torch
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ModelLoader:
    """Enhanced service class for loading and managing various model artifacts."""
    
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            self.models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        else:
            self.models_dir = models_dir
        
        # Set device for PyTorch models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self, model_type: str) -> Any:
        """
        Load a model from various formats (joblib, PyTorch, etc.).
        
        Args:
            model_type: Type of model to load ("challenger", "passed", "torch_tabular", 
                       "torch_ensemble", "torch_transformer", etc.)
            
        Returns:
            Loaded model object
        """
        try:
            # Try different file extensions and formats
            model_files = [
                f"{model_type}-model.joblib",      # Standard scikit-learn models
                f"{model_type}_model.pth",         # PyTorch models
                f"torch_{model_type}_model.pth",   # PyTorch models with prefix
                f"{model_type}.pth",               # Simple PyTorch models
                f"best_torch_model_{model_type}.pth",  # Best PyTorch models
            ]
            
            for model_file in model_files:
                model_path = os.path.join(self.models_dir, model_file)
                
                if os.path.exists(model_path):
                    if model_file.endswith('.joblib'):
                        return self._load_joblib_model(model_path)
                    elif model_file.endswith('.pth'):
                        return self._load_torch_model(model_path, model_type)
            
            raise FileNotFoundError(f"No model found for type: {model_type}")
                
        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {str(e)}")
            raise
    
    def _load_joblib_model(self, model_path: str) -> Any:
        """Load a joblib model."""
        logger.info(f"Loading joblib model from {model_path}")
        return joblib.load(model_path)
    
    def _load_torch_model(self, model_path: str, model_type: str) -> Any:
        """
        Load a PyTorch model.
        
        Args:
            model_path: Path to the model file
            model_type: Type of PyTorch model
            
        Returns:
            Loaded PyTorch model
        """
        logger.info(f"Loading PyTorch model from {model_path}")
        
        try:
            # Load the saved model data
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                # New format with metadata
                model_class = checkpoint.get('model_type', 'tabular')
                
                # Import the appropriate model class
                if 'torch' in model_type or 'tabular' in model_type:
                    from models.torch_model import TorchRentPredictor
                    predictor = TorchRentPredictor(model_type=model_class, device=self.device)
                    
                    # We need to recreate the model architecture
                    # This is a simplified version - in practice you'd save architecture info
                    input_dim = checkpoint.get('input_dim', 100)  # Default fallback
                    predictor.model = predictor.create_model(input_dim)
                    predictor.model.load_state_dict(checkpoint['model_state_dict'])
                    predictor.scaler = checkpoint.get('scaler')
                    predictor.numerical_cols = checkpoint.get('numerical_cols', [])
                    predictor.categorical_cols = checkpoint.get('categorical_cols', [])
                    
                    return predictor
                
                elif 'transformer' in model_type:
                    from models.transformer_model import TransformerRentPredictor
                    predictor = TransformerRentPredictor(
                        model_type=checkpoint.get('transformer_type', 'tabtransformer'),
                        device=self.device
                    )
                    # Similar process for transformer models
                    return predictor
                
                else:
                    # Legacy format - just the state dict
                    logger.warning(f"Loading legacy PyTorch model format: {model_path}")
                    # Return the state dict - caller needs to handle model creation
                    return checkpoint
            
            else:
                # Very old format - just the model object
                logger.warning(f"Loading very old PyTorch model format: {model_path}")
                return checkpoint
                
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            # Fallback - return None so caller can handle
            return None
    
    def get_model_info(self, model) -> Dict[str, Any]:
        """
        Get information about a loaded model.
        
        Args:
            model: Loaded model object
            
        Returns:
            Dict containing model information
        """
        try:
            model_info = {
                "loaded": model is not None,
            }
            
            if model is None:
                return model_info
            
            # Check model type
            model_type_str = str(type(model))
            
            if 'torch' in model_type_str.lower():
                model_info.update({
                    "type": "pytorch",
                    "framework": "PyTorch",
                    "has_predict": hasattr(model, 'model') and hasattr(model.model, 'forward'),
                    "device": str(getattr(model, 'device', 'unknown')),
                    "model_type": getattr(model, 'model_type', 'unknown')
                })
            elif 'sklearn' in model_type_str or hasattr(model, 'predict'):
                model_info.update({
                    "type": "sklearn",
                    "framework": "scikit-learn",
                    "has_predict": hasattr(model, 'predict'),
                    "has_fit": hasattr(model, 'fit')
                })
            else:
                model_info.update({
                    "type": model_type_str.split("'")[1] if "'" in model_type_str else "unknown",
                    "framework": "unknown",
                    "has_predict": hasattr(model, 'predict'),
                    "has_fit": hasattr(model, 'fit')
                })
            
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            return {
                "type": "unknown",
                "loaded": False,
                "error": str(e)
            }
    
    def model_exists(self, model_type: str) -> bool:
        """
        Check if a model file exists for any supported format.
        
        Args:
            model_type: Type of model to check
            
        Returns:
            True if model exists, False otherwise
        """
        model_files = [
            f"{model_type}-model.joblib",
            f"{model_type}_model.pth",
            f"torch_{model_type}_model.pth",
            f"{model_type}.pth",
            f"best_torch_model_{model_type}.pth",
        ]
        
        for model_file in model_files:
            model_path = os.path.join(self.models_dir, model_file)
            if os.path.exists(model_path):
                return True
        
        return False
    
    def list_available_models(self) -> Dict[str, list]:
        """
        List all available models in the models directory.
        
        Returns:
            Dict with model types and their available formats
        """
        available_models = {
            "joblib": [],
            "pytorch": [],
            "other": []
        }
        
        if not os.path.exists(self.models_dir):
            return available_models
        
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.joblib'):
                model_name = filename.replace('-model.joblib', '')
                available_models["joblib"].append(model_name)
            elif filename.endswith('.pth'):
                model_name = filename.replace('.pth', '')
                available_models["pytorch"].append(model_name)
            elif filename.endswith(('.pkl', '.pickle')):
                model_name = filename.split('.')[0]
                available_models["other"].append(model_name)
        
        return available_models
    
    def load_torch_predictor_for_inference(self, model_type: str) -> Optional[Any]:
        """
        Load a PyTorch predictor ready for inference.
        
        Args:
            model_type: Type of PyTorch model to load
            
        Returns:
            Loaded predictor ready for inference, or None if loading fails
        """
        try:
            model = self.load_model(model_type)
            
            # If it's already a predictor, return it
            if hasattr(model, 'predict') and hasattr(model, 'model'):
                return model
            
            # Otherwise, try to create a predictor wrapper
            logger.warning(f"Model {model_type} is not a predictor class, creating wrapper...")
            # This would need to be implemented based on your specific needs
            return None
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch predictor {model_type}: {e}")
            return None