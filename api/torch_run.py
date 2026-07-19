"""
Updated PyTorch model runner using advanced neural network implementations.

This script demonstrates both custom neural networks and transformer models
for Tokyo rent prediction with proper MLflow integration.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import mlflow
import mlflow.pytorch
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split

# Import our advanced models
from models.torch_model import TorchRentPredictor, train_and_compare_torch_models
from models.transformer_model import TransformerRentPredictor, train_transformer_models

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set MLflow tracking URI (uses environment variable or defaults to local)
import os
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(mlflow_uri)
logger.info(f"MLflow tracking URI: {mlflow_uri}")


def run_comprehensive_torch_training():
    """
    Run comprehensive PyTorch model training including:
    1. Advanced tabular neural networks
    2. Ensemble models
    3. Transformer-based models
    4. Model comparison and selection
    """
    
    logger.info("Starting comprehensive PyTorch model training...")
    
    with mlflow.start_run(run_name="comprehensive_torch_training") as main_run:
        all_results = {}
        
        # 1. Train advanced PyTorch models
        logger.info("="*60)
        logger.info("TRAINING ADVANCED PYTORCH MODELS")
        logger.info("="*60)
        
        torch_results = train_and_compare_torch_models()
        all_results.update(torch_results)
        
        # 2. Train transformer models
        logger.info("="*60)
        logger.info("TRAINING TRANSFORMER MODELS")  
        logger.info("="*60)
        
        transformer_results = train_transformer_models()
        all_results.update(transformer_results)
        
        # 3. Compare all models
        logger.info("="*60)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("="*60)
        
        best_model = None
        best_rmse = float('inf')
        
        for model_name, result in all_results.items():
            if 'metrics' in result:
                rmse = result['metrics']['rmse']
                r2 = result['metrics']['r2']
                mae = result['metrics']['mae']
                
                logger.info(f"{model_name:20s}: RMSE={rmse:7.2f}, MAE={mae:7.2f}, R²={r2:6.4f}")
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model_name
        
        # Log overall results
        if best_model:
            logger.info(f"\n🏆 BEST OVERALL MODEL: {best_model} (RMSE: {best_rmse:.2f})")
            
            mlflow.log_param("best_overall_model", best_model)
            mlflow.log_metric("best_overall_rmse", best_rmse)
            mlflow.log_metric("best_overall_r2", all_results[best_model]['metrics']['r2'])
            
            # Register best model
            try:
                best_predictor = all_results[best_model]['predictor']
                model_path = f"best_torch_model_{best_model}.pth"
                
                if hasattr(best_predictor, 'save_model'):
                    best_predictor.save_model(model_path)
                    mlflow.log_artifact(model_path)
                    
                    # Register in MLflow model registry
                    model_uri = f"runs:/{main_run.info.run_id}/{model_path}"
                    mlflow.register_model(model_uri, "tokyo_rent_torch_best")
                    
                    logger.info(f"✅ Best model registered: tokyo_rent_torch_best")
                
            except Exception as e:
                logger.error(f"Failed to register best model: {e}")
        
        # Log model count and types
        mlflow.log_metric("total_models_trained", len(all_results))
        mlflow.log_param("model_types", list(all_results.keys()))
        
    return all_results


def quick_torch_demo():
    """
    Quick demonstration of a single PyTorch model for testing.
    """
    logger.info("Running quick PyTorch demo...")
    
    # Initialize predictor
    predictor = TorchRentPredictor(model_type="tabular")
    
    # Load data
    X, y = predictor.load_and_prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    with mlflow.start_run(run_name="quick_torch_demo") as run:
        # Quick hyperparameter setup
        hyperparams = {
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'batch_size': 128,
            'patience': 10,
            'dropout_rate': 0.3,
            'hidden_dims': [512, 256, 128, 64],
            'use_attention': True
        }
        
        # Train model
        logger.info("Training tabular neural network...")
        history = predictor.train_model(X_train, y_train, X_val, y_val, 
                                      hyperparams, num_epochs=50)
        
        # Evaluate
        metrics = predictor.evaluate_model(X_test, y_test)
        
        # Log results
        mlflow.log_params(hyperparams)
        mlflow.log_metrics(metrics)
        
        # Save model
        model_path = "quick_demo_torch_model.pth"
        predictor.save_model(model_path)
        mlflow.log_artifact(model_path)
        
        logger.info(f"Demo Results - RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.4f}")
        
    return predictor, metrics


if __name__ == "__main__":
    # Choose training mode
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Quick demo mode
        predictor, metrics = quick_torch_demo()
        print(f"Quick Demo Complete! RMSE: {metrics['rmse']:.2f}")
        
    else:
        # Comprehensive training mode
        results = run_comprehensive_torch_training()
        print("Comprehensive PyTorch Training Complete!")
        
        # Print summary
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        
        for model_name, result in results.items():
            if 'metrics' in result:
                metrics = result['metrics']
                print(f"{model_name:20s}: RMSE={metrics['rmse']:7.2f}, R²={metrics['r2']:6.4f}")
        
        print("\n🚀 All PyTorch models trained and logged to MLflow!")
        print("Check your MLflow UI to compare model performance and artifacts.")