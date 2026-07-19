"""
Model definitions and training utilities for Tokyo rent prediction.
"""

# Import modules for direct access
from . import challenger_model
from . import reg_model

# Import specific functions (may fail if dependencies not installed)
try:
    from .challenger_model import train_and_log_model as train_challenger_model, load_data as load_challenger_data
    from .reg_model import train_model as train_regression_models, load_data as load_regression_data
    
    __all__ = [
        "challenger_model",
        "reg_model", 
        "train_challenger_model",
        "train_regression_models", 
        "load_regression_data",
        "load_challenger_data"
    ]
except ImportError:
    # If dependencies aren't installed, just export the modules
    __all__ = ["challenger_model", "reg_model"]