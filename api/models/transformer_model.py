"""
HuggingFace Transformer Models for Tabular Rent Prediction

This module implements pre-trained transformer models fine-tuned for tabular data:
1. TabTransformer architecture
2. FT-Transformer (Feature Tokenizer Transformer)  
3. Fine-tuned BERT/RoBERTa for numerical data
4. Tabular data preprocessing for transformer inputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import mlflow
import logging
from typing import Dict, List, Tuple, Optional, Any

from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    Trainer, TrainingArguments,
    EarlyStoppingCallback, DataCollatorWithPadding
)
from transformers.modeling_outputs import BaseModelOutput
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from datasets import Dataset as HFDataset

logger = logging.getLogger(__name__)


class TabularTransformerDataset(Dataset):
    """Dataset class for transformer-based tabular models."""
    
    def __init__(self, features: pd.DataFrame, targets: pd.Series,
                 numerical_cols: List[str], categorical_cols: List[str],
                 tokenizer=None, max_length: int = 512):
        """
        Initialize dataset for transformer models.
        
        Args:
            features: Feature DataFrame
            targets: Target Series  
            numerical_cols: Numerical column names
            categorical_cols: Categorical column names
            tokenizer: HuggingFace tokenizer (optional)
            max_length: Maximum sequence length
        """
        self.features = features
        self.targets = targets.values.astype(np.float32)
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Prepare numerical features
        self.scaler = QuantileTransformer(n_quantiles=1000, random_state=42)
        self.numerical_features = self.scaler.fit_transform(
            features[numerical_cols].fillna(0)
        ).astype(np.float32)
        
        # Prepare categorical features
        self.categorical_features = features[categorical_cols].fillna(0).values.astype(np.float32)
        
        # Create text representations for transformer input
        if tokenizer is not None:
            self.text_inputs = self._create_text_inputs()
    
    def _create_text_inputs(self) -> List[str]:
        """Convert tabular data to text format for transformer input."""
        text_inputs = []
        
        for idx in range(len(self.features)):
            row = self.features.iloc[idx]
            
            # Create descriptive text from features
            text_parts = []
            
            # Add numerical features as text
            for col in self.numerical_cols:
                value = row[col]
                if not pd.isna(value):
                    if col == 'rei_price':
                        text_parts.append(f"rent fee {int(value)} yen")
                    elif col == 'shikikin':
                        text_parts.append(f"deposit {int(value)} yen")
                    elif col == 'maintenence_price':
                        text_parts.append(f"maintenance {int(value)} yen")
                    elif col == 'year_built':
                        text_parts.append(f"built in {int(value)}")
                    elif col == 'floor':
                        text_parts.append(f"floor {int(value)}")
                    elif col == 'eki_walk':
                        text_parts.append(f"{int(value)} minutes walk to station")
                    elif col == 'sqr_m':
                        text_parts.append(f"area {value:.1f} square meters")
            
            # Add categorical features
            for col in self.categorical_cols:
                if row[col] == 1:  # One-hot encoded
                    if col.startswith('ku_'):
                        ward = col.replace('ku_', '').replace('_', ' ')
                        text_parts.append(f"located in {ward}")
                    elif col.startswith('apt_'):
                        apt_type = col.replace('apt_', '').replace('_', ' ')
                        text_parts.append(f"apartment type {apt_type}")
                    elif col.startswith('house_'):
                        house_type = col.replace('house_', '').replace('_', ' ')
                        text_parts.append(f"house type {house_type}")
            
            # Combine all parts
            text = " ".join(text_parts)
            text_inputs.append(text)
        
        return text_inputs
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        if self.tokenizer is not None:
            # Tokenize text input
            text = self.text_inputs[idx]
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'numerical_features': torch.tensor(self.numerical_features[idx]),
                'categorical_features': torch.tensor(self.categorical_features[idx]),
                'labels': torch.tensor(self.targets[idx])
            }
        else:
            return {
                'numerical_features': torch.tensor(self.numerical_features[idx]),
                'categorical_features': torch.tensor(self.categorical_features[idx]),
                'labels': torch.tensor(self.targets[idx])
            }


class TabTransformer(nn.Module):
    """TabTransformer architecture for tabular data."""
    
    def __init__(self, numerical_features: int, categorical_features: int,
                 d_model: int = 256, nhead: int = 8, num_layers: int = 6,
                 dim_feedforward: int = 1024, dropout: float = 0.1):
        super(TabTransformer, self).__init__()
        
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.d_model = d_model
        
        # Numerical feature processing
        self.numerical_processor = nn.Sequential(
            nn.Linear(numerical_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Categorical feature embeddings
        self.categorical_embeddings = nn.Linear(categorical_features, d_model)
        
        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, 2, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Final prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, numerical_features, categorical_features):
        batch_size = numerical_features.size(0)
        
        # Process features
        num_embed = self.numerical_processor(numerical_features)  # (B, d_model)
        cat_embed = self.categorical_embeddings(categorical_features)  # (B, d_model)
        
        # Stack embeddings as sequence
        embeddings = torch.stack([num_embed, cat_embed], dim=1)  # (B, 2, d_model)
        
        # Add positional embeddings
        embeddings = embeddings + self.pos_embedding
        
        # Apply transformer
        transformer_output = self.transformer(embeddings)  # (B, 2, d_model)
        
        # Flatten for prediction
        flattened = transformer_output.view(batch_size, -1)  # (B, 2 * d_model)
        
        # Final prediction
        output = self.prediction_head(flattened)
        return output.squeeze(-1)


class HybridBERTTabular(nn.Module):
    """BERT-based model for tabular data with numerical feature integration."""
    
    def __init__(self, bert_model_name: str = "bert-base-uncased",
                 numerical_features: int = 7, categorical_features: int = 50,
                 hidden_dim: int = 256, dropout: float = 0.3):
        super(HybridBERTTabular, self).__init__()
        
        # Load pre-trained BERT
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # Freeze BERT for feature extraction (can be unfrozen for fine-tuning)
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Numerical feature processor
        self.numerical_processor = nn.Sequential(
            nn.Linear(numerical_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Categorical feature processor
        self.categorical_processor = nn.Sequential(
            nn.Linear(categorical_features, hidden_dim),
            nn.LayerNorm(hidden_dim),  
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism for feature fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Final fusion and prediction
        combined_dim = self.bert_hidden_size + hidden_dim * 2
        
        self.fusion_layers = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, input_ids, attention_mask, numerical_features, categorical_features):
        # Get BERT features
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_features = bert_output.pooler_output  # (B, bert_hidden_size)
        
        # Process tabular features
        num_features = self.numerical_processor(numerical_features)  # (B, hidden_dim)
        cat_features = self.categorical_processor(categorical_features)  # (B, hidden_dim)
        
        # Apply attention between numerical and categorical features
        tabular_features = torch.stack([num_features, cat_features], dim=1)  # (B, 2, hidden_dim)
        attended_features, _ = self.attention(tabular_features, tabular_features, tabular_features)
        attended_features = attended_features.mean(dim=1)  # (B, hidden_dim)
        
        # Combine all features
        combined_features = torch.cat([
            bert_features, 
            num_features, 
            attended_features
        ], dim=1)
        
        # Final prediction
        output = self.fusion_layers(combined_features)
        return output.squeeze(-1)


class TransformerRentPredictor:
    """Transformer-based rent predictor with HuggingFace integration."""
    
    def __init__(self, model_type: str = "tabtransformer", 
                 bert_model_name: str = "bert-base-uncased",
                 device: str = None):
        """
        Initialize transformer rent predictor.
        
        Args:
            model_type: Type of model ("tabtransformer", "bert_hybrid")
            bert_model_name: Pre-trained model name for BERT-based models
            device: Device for training
        """
        self.model_type = model_type
        self.bert_model_name = bert_model_name
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.tokenizer = None
        self.numerical_cols = ['rei_price', 'shikikin', 'maintenence_price', 
                              'year_built', 'floor', 'eki_walk', 'sqr_m']
        self.categorical_cols = []
        
        # Initialize tokenizer for BERT-based models
        if model_type == "bert_hybrid":
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Initialized TransformerRentPredictor: {model_type}")
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare data (same as TabularNN)."""
        try:
            from datasets import load_dataset
            ds = load_dataset("jbrazzy/tokyo_rent", split="train")
            df = ds.to_pandas()
        except:
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
        
        self.categorical_cols = [col for col in df.columns 
                               if col.startswith(('ku_', 'apt_', 'house_')) and col != 'rent_price']
        
        X = df.drop('rent_price', axis=1)
        y = df['rent_price']
        
        return X, y
    
    def create_model(self) -> nn.Module:
        """Create the appropriate transformer model."""
        num_numerical = len(self.numerical_cols)
        num_categorical = len(self.categorical_cols)
        
        if self.model_type == "tabtransformer":
            model = TabTransformer(
                numerical_features=num_numerical,
                categorical_features=num_categorical,
                d_model=256,
                nhead=8,
                num_layers=4,
                dropout=0.1
            )
        elif self.model_type == "bert_hybrid":
            model = HybridBERTTabular(
                bert_model_name=self.bert_model_name,
                numerical_features=num_numerical,
                categorical_features=num_categorical,
                hidden_dim=256,
                dropout=0.3
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model.to(self.device)
    
    def train_with_huggingface_trainer(self, X_train: pd.DataFrame, y_train: pd.Series,
                                      X_val: pd.DataFrame, y_val: pd.Series,
                                      output_dir: str = "./transformer_results",
                                      num_epochs: int = 10) -> Dict:
        """
        Train using HuggingFace Trainer API.
        """
        # Create datasets
        train_dataset = TabularTransformerDataset(
            X_train, y_train, self.numerical_cols, self.categorical_cols,
            tokenizer=self.tokenizer
        )
        
        val_dataset = TabularTransformerDataset(
            X_val, y_val, self.numerical_cols, self.categorical_cols,
            tokenizer=self.tokenizer
        )
        
        # Create model
        self.model = self.create_model()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb/tensorboard
            dataloader_num_workers=2,
            remove_unused_columns=False
        )
        
        # Custom trainer for regression
        class RegressionTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.pop("labels")
                
                if self.model_type == "bert_hybrid":
                    outputs = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        numerical_features=inputs["numerical_features"],
                        categorical_features=inputs["categorical_features"]
                    )
                else:
                    outputs = model(
                        numerical_features=inputs["numerical_features"],
                        categorical_features=inputs["categorical_features"]
                    )
                
                loss_fct = nn.MSELoss()
                loss = loss_fct(outputs, labels)
                
                return (loss, outputs) if return_outputs else loss
        
        # Initialize trainer
        trainer = RegressionTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train model
        logger.info("Starting transformer training with HuggingFace Trainer...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        
        return {"trainer": trainer, "training_args": training_args}
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate the transformer model."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Create test dataset  
        test_dataset = TabularTransformerDataset(
            X_test, y_test, self.numerical_cols, self.categorical_cols,
            tokenizer=self.tokenizer
        )
        
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Move to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Forward pass
                if self.model_type == "bert_hybrid":
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        numerical_features=batch["numerical_features"],
                        categorical_features=batch["categorical_features"]
                    )
                else:
                    outputs = self.model(
                        numerical_features=batch["numerical_features"],
                        categorical_features=batch["categorical_features"]
                    )
                
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch["labels"].cpu().numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        logger.info(f"Transformer Results - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
        
        return metrics


def train_transformer_models():
    """Train and compare transformer models."""
    model_types = ["tabtransformer", "bert_hybrid"]
    results = {}
    
    with mlflow.start_run(run_name="transformer_models") as run:
        for model_type in model_types:
            logger.info(f"Training {model_type} model...")
            
            try:
                # Initialize predictor
                predictor = TransformerRentPredictor(model_type=model_type)
                
                # Load data
                X, y = predictor.load_and_prepare_data()
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42
                )
                
                with mlflow.start_run(run_name=f"transformer_{model_type}", nested=True):
                    # Train model
                    training_results = predictor.train_with_huggingface_trainer(
                        X_train, y_train, X_val, y_val,
                        output_dir=f"./transformer_{model_type}_results",
                        num_epochs=5  # Reduced for demo
                    )
                    
                    # Evaluate
                    metrics = predictor.evaluate_model(X_test, y_test)
                    
                    # Log metrics
                    mlflow.log_metrics(metrics)
                    mlflow.log_param("model_type", model_type)
                    
                    results[model_type] = {
                        'predictor': predictor,
                        'metrics': metrics,
                        'training_results': training_results
                    }
                    
                    logger.info(f"{model_type} - RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.4f}")
            
            except Exception as e:
                logger.error(f"Error training {model_type}: {e}")
                continue
        
        # Log best model
        if results:
            best_model = min(results.keys(), key=lambda k: results[k]['metrics']['rmse'])
            mlflow.log_param("best_transformer_model", best_model)
            mlflow.log_metric("best_transformer_rmse", results[best_model]['metrics']['rmse'])
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Train transformer models
    results = train_transformer_models()
    
    print("Transformer Model Training Complete!")
    for model_type, result in results.items():
        metrics = result['metrics']
        print(f"{model_type}: RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.4f}")