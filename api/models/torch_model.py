"""
Advanced PyTorch Neural Network Models for Tokyo Rent Prediction

This module implements multiple neural network architectures:
1. Custom Tabular Neural Network with attention mechanisms
2. Fine-tuned transformer models from HuggingFace
3. Hybrid models combining embeddings and numerical features
4. Advanced training with early stopping, regularization, and hyperparameter tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torch.optim as optim

import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch
import joblib
import logging
import os
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datasets import load_dataset
import optuna
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    TrainingArguments, Trainer, 
    EarlyStoppingCallback
)

logger = logging.getLogger(__name__)

# MLflow tracking URI is set globally by the calling script
# Don't set it here to avoid conflicts


class RentDataset(Dataset):
    """Enhanced dataset class for rent prediction with proper preprocessing."""
    
    def __init__(self, features: pd.DataFrame, targets: pd.Series, 
                 numerical_cols: List[str], categorical_cols: List[str],
                 scaler: StandardScaler = None, fit_scaler: bool = True):
        """
        Initialize dataset with proper feature preprocessing.
        
        Args:
            features: Feature DataFrame
            targets: Target Series
            numerical_cols: List of numerical column names
            categorical_cols: List of categorical column names
            scaler: StandardScaler for numerical features
            fit_scaler: Whether to fit the scaler (True for train, False for test)
        """
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        
        # Handle numerical features
        if scaler is None:
            self.scaler = StandardScaler()
        else:
            self.scaler = scaler
            
        self.numerical_features = features[numerical_cols].fillna(0).values.astype(np.float32)
        
        if fit_scaler:
            self.numerical_features = self.scaler.fit_transform(self.numerical_features)
        else:
            self.numerical_features = self.scaler.transform(self.numerical_features)
        
        # Handle categorical features (already one-hot encoded)
        self.categorical_features = features[categorical_cols].fillna(0).values.astype(np.float32)
        
        # Combine all features
        self.all_features = np.concatenate([self.numerical_features, self.categorical_features], axis=1)
        
        # Handle targets
        self.targets = targets.fillna(targets.median()).values.astype(np.float32)
        
        # Store dimensions
        self.num_numerical = len(numerical_cols)
        self.num_categorical = len(categorical_cols)
        self.total_features = self.all_features.shape[1]
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return torch.tensor(self.all_features[idx]), torch.tensor(self.targets[idx])


class AttentionBlock(nn.Module):
    """Attention mechanism for tabular data."""
    
    def __init__(self, input_dim: int, attention_dim: int = 64):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # For tabular data, we treat features as sequence
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        attention_weights = self.attention(x)  # (batch_size, 1, 1)
        attended_features = x * attention_weights  # (batch_size, 1, input_dim)
        
        return attended_features.squeeze(1), attention_weights.squeeze(-1)


class TabularNeuralNetwork(nn.Module):
    """Advanced neural network for tabular data with attention and regularization."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256, 128, 64],
                 dropout_rate: float = 0.3, use_attention: bool = True,
                 use_batch_norm: bool = True):
        super(TabularNeuralNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.use_attention = use_attention
        
        # Input layer
        layers = []
        current_dim = input_dim
        
        # Attention mechanism
        if use_attention:
            self.attention = AttentionBlock(input_dim)
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        if self.use_attention:
            x, attention_weights = self.attention(x)
            
        output = self.network(x)
        return output.squeeze(-1)  # Remove last dimension for regression


class HybridTransformerModel(nn.Module):
    """Hybrid model combining transformer embeddings with tabular features."""
    
    def __init__(self, transformer_model_name: str = "microsoft/DialoGPT-medium",
                 numerical_dim: int = 7, categorical_dim: int = 50, 
                 hidden_dim: int = 256, dropout_rate: float = 0.3):
        super(HybridTransformerModel, self).__init__()
        
        # Load pre-trained transformer for feature extraction
        self.transformer = AutoModel.from_pretrained(transformer_model_name)
        
        # Freeze transformer parameters for feature extraction
        for param in self.transformer.parameters():
            param.requires_grad = False
        
        # Get transformer hidden size
        transformer_hidden_size = self.transformer.config.hidden_size
        
        # Feature processing layers
        self.numerical_processor = nn.Sequential(
            nn.Linear(numerical_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.categorical_processor = nn.Sequential(
            nn.Linear(categorical_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Combine features
        combined_dim = hidden_dim + transformer_hidden_size
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, numerical_features, categorical_features, transformer_input_ids=None):
        # Process tabular features
        num_features = self.numerical_processor(numerical_features)
        cat_features = self.categorical_processor(categorical_features)
        tabular_features = torch.cat([num_features, cat_features], dim=1)
        
        # If transformer input is provided, use it
        if transformer_input_ids is not None:
            transformer_output = self.transformer(transformer_input_ids)
            transformer_features = transformer_output.last_hidden_state.mean(dim=1)  # Pool over sequence
            combined_features = torch.cat([tabular_features, transformer_features], dim=1)
        else:
            combined_features = tabular_features
        
        output = self.fusion_layer(combined_features)
        return output.squeeze(-1)


class EnsembleNeuralNetwork(nn.Module):
    """Ensemble of multiple neural networks for improved performance."""
    
    def __init__(self, input_dim: int, num_models: int = 3):
        super(EnsembleNeuralNetwork, self).__init__()
        
        self.models = nn.ModuleList([
            TabularNeuralNetwork(
                input_dim, 
                hidden_dims=[512, 256, 128] if i == 0 else
                           [256, 128, 64] if i == 1 else
                           [384, 192, 96],
                dropout_rate=0.2 + i * 0.1,
                use_attention=i % 2 == 0
            ) for i in range(num_models)
        ])
        
        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(num_models) / num_models)
    
    def forward(self, x):
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred.unsqueeze(-1))
        
        # Stack predictions and apply weights
        stacked_preds = torch.cat(predictions, dim=-1)  # (batch_size, num_models)
        weights = F.softmax(self.ensemble_weights, dim=0)
        
        # Weighted average
        ensemble_pred = torch.sum(stacked_preds * weights, dim=-1)
        return ensemble_pred


class TorchRentPredictor:
    """Main class for training and managing PyTorch models for rent prediction."""
    
    def __init__(self, model_type: str = "tabular", 
                 device: str = None, random_seed: int = 42):
        """
        Initialize the PyTorch rent predictor.
        
        Args:
            model_type: Type of model ("tabular", "hybrid", "ensemble")
            device: Device to use for training
            random_seed: Random seed for reproducibility
        """
        self.model_type = model_type
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.random_seed = random_seed
        
        # Set random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        self.model = None
        self.scaler = None
        self.numerical_cols = ['rei_price', 'shikikin', 'maintenence_price', 
                              'year_built', 'floor', 'eki_walk', 'sqr_m']
        self.categorical_cols = []
        
        logger.info(f"Initialized TorchRentPredictor with device: {self.device}")
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare the rent dataset."""
        try:
            # Load dataset
            ds = load_dataset("jbrazzy/tokyo_rent", split="train")
            df = ds.to_pandas()
        except:
            # Fallback to local data
            df = pd.read_csv("../data/tokyo_model.csv")
        
        # Preprocessing
        df['ku_name'] = df['ku_name'].astype('category')
        df = pd.get_dummies(df, columns=['ku_name'], prefix='ku')
        
        df['apartment_type'] = df['apartment_type'].astype('category')
        df = pd.get_dummies(df, columns=['apartment_type'], prefix='apt')
        
        df['house_type'] = df['house_type'].astype('category')
        df = pd.get_dummies(df, columns=['house_type'], prefix='house')
        
        if 'address' in df.columns:
            df = df.drop(['address'], axis=1)
        
        # Update categorical columns
        self.categorical_cols = [col for col in df.columns 
                               if col.startswith(('ku_', 'apt_', 'house_')) and col != 'rent_price']
        
        X = df.drop('rent_price', axis=1)
        y = df['rent_price']
        
        logger.info(f"Data shape: {X.shape}, Numerical features: {len(self.numerical_cols)}, "
                   f"Categorical features: {len(self.categorical_cols)}")
        
        return X, y
    
    def create_model(self, input_dim: int, hyperparams: Dict = None) -> nn.Module:
        """Create the appropriate model based on model_type."""
        if hyperparams is None:
            hyperparams = {}
        
        if self.model_type == "tabular":
            model = TabularNeuralNetwork(
                input_dim=input_dim,
                hidden_dims=hyperparams.get('hidden_dims', [512, 256, 128, 64]),
                dropout_rate=hyperparams.get('dropout_rate', 0.3),
                use_attention=hyperparams.get('use_attention', True),
                use_batch_norm=hyperparams.get('use_batch_norm', True)
            )
        elif self.model_type == "ensemble":
            model = EnsembleNeuralNetwork(
                input_dim=input_dim,
                num_models=hyperparams.get('num_models', 3)
            )
        elif self.model_type == "hybrid":
            num_numerical = len(self.numerical_cols)
            num_categorical = len(self.categorical_cols)
            model = HybridTransformerModel(
                numerical_dim=num_numerical,
                categorical_dim=num_categorical,
                hidden_dim=hyperparams.get('hidden_dim', 256),
                dropout_rate=hyperparams.get('dropout_rate', 0.3)
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model.to(self.device)
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series,
                   hyperparams: Dict = None, num_epochs: int = 100) -> Dict:
        """
        Train the PyTorch model with early stopping and advanced techniques.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            hyperparams: Hyperparameters for the model
            num_epochs: Maximum number of epochs
            
        Returns:
            Training history and metrics
        """
        if hyperparams is None:
            hyperparams = {
                'learning_rate': 0.001,
                'weight_decay': 1e-5,
                'batch_size': 128,
                'patience': 15
            }
        
        # Create datasets
        train_dataset = RentDataset(X_train, y_train, self.numerical_cols, 
                                   self.categorical_cols, fit_scaler=True)
        val_dataset = RentDataset(X_val, y_val, self.numerical_cols, 
                                 self.categorical_cols, scaler=train_dataset.scaler, 
                                 fit_scaler=False)
        
        self.scaler = train_dataset.scaler
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], 
                                 shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], 
                               shuffle=False, num_workers=2)
        
        # Create model
        self.model = self.create_model(train_dataset.total_features, hyperparams)
        
        # Setup training
        optimizer = optim.AdamW(self.model.parameters(), 
                               lr=hyperparams['learning_rate'],
                               weight_decay=hyperparams['weight_decay'])
        
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                     patience=5, verbose=True)
        
        criterion = nn.MSELoss()
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            num_batches = 0
            
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                
                if self.model_type == "hybrid":
                    # Split features for hybrid model
                    num_features = batch_features[:, :len(self.numerical_cols)]
                    cat_features = batch_features[:, len(self.numerical_cols):]
                    outputs = self.model(num_features, cat_features)
                else:
                    outputs = self.model(batch_features)
                
                loss = criterion(outputs, batch_targets)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            train_loss /= num_batches
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            num_val_batches = 0
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    if self.model_type == "hybrid":
                        num_features = batch_features[:, :len(self.numerical_cols)]
                        cat_features = batch_features[:, len(self.numerical_cols):]
                        outputs = self.model(num_features, cat_features)
                    else:
                        outputs = self.model(batch_features)
                    
                    loss = criterion(outputs, batch_targets)
                    val_loss += loss.item()
                    num_val_batches += 1
            
            val_loss /= num_val_batches
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_torch_model.pth')
            else:
                patience_counter += 1
            
            # Print progress
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                logger.info(f"Epoch {epoch}/{num_epochs}: "
                           f"Train Loss: {train_loss:.4f}, "
                           f"Val Loss: {val_loss:.4f}, "
                           f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if patience_counter >= hyperparams['patience']:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_torch_model.pth'))
        
        return history
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate the trained model."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Create test dataset
        test_dataset = RentDataset(X_test, y_test, self.numerical_cols, 
                                  self.categorical_cols, scaler=self.scaler, 
                                  fit_scaler=False)
        
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(self.device)
                
                if self.model_type == "hybrid":
                    num_features = batch_features[:, :len(self.numerical_cols)]
                    cat_features = batch_features[:, len(self.numerical_cols):]
                    outputs = self.model(num_features, cat_features)
                else:
                    outputs = self.model(batch_features)
                
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_targets.numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions,
            'actuals': actuals
        }
        
        logger.info(f"Test Results - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
        
        return metrics
    
    def hyperparameter_optimization(self, X_train: pd.DataFrame, y_train: pd.Series,
                                   n_trials: int = 50) -> Dict:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X_train, y_train: Training data
            n_trials: Number of optimization trials
            
        Returns:
            Best hyperparameters
        """
        def objective(trial):
            # Suggest hyperparameters
            hyperparams = {
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
                'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-3),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                'dropout_rate': trial.suggest_uniform('dropout_rate', 0.1, 0.5),
                'hidden_dims': [
                    trial.suggest_categorical('hidden_1', [256, 512, 768, 1024]),
                    trial.suggest_categorical('hidden_2', [128, 256, 384, 512]),
                    trial.suggest_categorical('hidden_3', [64, 128, 192, 256]),
                    trial.suggest_categorical('hidden_4', [32, 64, 96, 128])
                ],
                'use_attention': trial.suggest_categorical('use_attention', [True, False]),
                'patience': 10  # Reduced for faster optimization
            }
            
            # Split training data for validation
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.2, random_state=self.random_seed
            )
            
            # Train model
            try:
                history = self.train_model(X_train_split, y_train_split, 
                                         X_val_split, y_val_split,
                                         hyperparams, num_epochs=50)
                
                # Return best validation loss
                return min(history['val_loss'])
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float('inf')
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best hyperparameters: {study.best_params}")
        logger.info(f"Best validation loss: {study.best_value}")
        
        return study.best_params
    
    def save_model(self, filepath: str):
        """Save the trained model and scaler."""
        if self.model is None:
            raise ValueError("No model to save")
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'scaler': self.scaler,
            'numerical_cols': self.numerical_cols,
            'categorical_cols': self.categorical_cols,
            'device': str(self.device)
        }
        
        torch.save(save_dict, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str, input_dim: int):
        """Load a trained model and scaler."""
        save_dict = torch.load(filepath, map_location=self.device)
        
        self.model_type = save_dict['model_type']
        self.scaler = save_dict['scaler']
        self.numerical_cols = save_dict['numerical_cols']
        self.categorical_cols = save_dict['categorical_cols']
        
        # Recreate model architecture
        self.model = self.create_model(input_dim)
        self.model.load_state_dict(save_dict['model_state_dict'])
        
        logger.info(f"Model loaded from {filepath}")


def train_and_compare_torch_models():
    """
    Train multiple PyTorch models and compare their performance.
    """
    # Initialize predictors for different model types
    model_types = ["tabular", "ensemble"]  # "hybrid" requires more setup
    results = {}
    
    with mlflow.start_run(run_name="pytorch_models_comparison") as run:
        for model_type in model_types:
            logger.info(f"Training {model_type} model...")
            
            # Initialize predictor
            predictor = TorchRentPredictor(model_type=model_type)
            
            # Load and prepare data
            X, y = predictor.load_and_prepare_data()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            try:
                # Train model
                with mlflow.start_run(run_name=f"torch_{model_type}", nested=True):
                    # Hyperparameter optimization (limited for demo)
                    logger.info("Running hyperparameter optimization...")
                    best_params = predictor.hyperparameter_optimization(
                        X_train, y_train, n_trials=20
                    )
                    
                    # Train final model with best parameters
                    logger.info("Training final model...")
                    history = predictor.train_model(X_train, y_train, X_val, y_val, 
                                                  best_params, num_epochs=100)
                    
                    # Evaluate model
                    metrics = predictor.evaluate_model(X_test, y_test)
                    
                    # Log to MLflow
                    mlflow.log_params(best_params)
                    mlflow.log_metrics({
                        'rmse': metrics['rmse'],
                        'mae': metrics['mae'],
                        'r2': metrics['r2'],
                        'mse': metrics['mse']
                    })
                    
                    # Save model
                    model_path = f"torch_{model_type}_model.pth"
                    predictor.save_model(model_path)
                    mlflow.log_artifact(model_path)
                    
                    # Log model to MLflow
                    mlflow.pytorch.log_model(
                        predictor.model, 
                        f"torch_{model_type}_model"
                    )
                    
                    results[model_type] = {
                        'predictor': predictor,
                        'metrics': metrics,
                        'best_params': best_params,
                        'history': history
                    }
                    
                    logger.info(f"{model_type} model - RMSE: {metrics['rmse']:.2f}, "
                              f"R²: {metrics['r2']:.4f}")
            
            except Exception as e:
                logger.error(f"Error training {model_type} model: {e}")
                continue
        
        # Compare results
        if results:
            best_model = min(results.keys(), key=lambda k: results[k]['metrics']['rmse'])
            best_rmse = results[best_model]['metrics']['rmse']
            
            logger.info(f"Best PyTorch model: {best_model} (RMSE: {best_rmse:.2f})")
            
            mlflow.log_param("best_torch_model", best_model)
            mlflow.log_metric("best_torch_rmse", best_rmse)
    
    return results


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Train and compare models
    results = train_and_compare_torch_models()
    
    print("PyTorch Model Training Complete!")
    for model_type, result in results.items():
        metrics = result['metrics']
        print(f"{model_type}: RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.4f}")