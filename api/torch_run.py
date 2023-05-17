import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch
import pandas as pd
import numpy as np
# https://towardsdatascience.com/attention-for-time-series-classification-and-forecasting-261723e0006d
# load your dataset and pull out X and y
ds = load_dataset("jbrazzy/tokyo_rent", split="train")
df = ds.to_pandas()

# make categorical variables and one-hot encode them
df['ku_name'] = df['ku_name'].astype('category')
df = pd.get_dummies(df, columns=['ku_name'])
df['apartment_type'] = df['apartment_type'].astype('category')
df = pd.get_dummies(df, columns=['apartment_type'])
df['house_type'] = df['house_type'].astype('category')
df = pd.get_dummies(df, columns=['house_type'])

# drop the address column and define X, y
df = df.drop(['address'], axis=1)
X = df.drop('rent_price', axis=1)
y = df['rent_price']

# Split your dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a custom Dataset class
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.fillna(0)  # replace NaN values with 0
        self.y = y.fillna(0)  # replace NaN values with 0

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_values = self.X.iloc[idx].values.astype('float32')
        y_value = np.array([self.y.iloc[idx]]).astype('float32')  # Keep as 1-element array
        return torch.from_numpy(x_values), torch.from_numpy(y_value)

# Create datasets and dataloaders
train_data = MyDataset(X_train, y_train)
test_data = MyDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

# Load the pre-trained model and tokenizer from Hugging Face
model_name = "bert-base-uncased"  # Replace with a suitable transformer model for your task
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize the model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

# Freeze some layers if necessary (optional, for fine-tuning)
for param in model.base_model.parameters():
    param.requires_grad = False

# Define optimizer and loss function for regression
optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.MSELoss()

# Training loop
model.train()
for epoch in range(10):  # Number of epochs
    for batch in train_loader:
        optimizer.zero_grad()
        features, labels = batch
        output = model(features).logits  # Get logits for regression
        loss = loss_fn(output.squeeze(), labels.squeeze())
        loss.backward()
        optimizer.step()

# Evaluate your model
model.eval()
predictions = []
with torch.no_grad():
    for batch in test_loader:
        features, labels = batch
        output = model(features).logits
        predictions.extend(output.squeeze().tolist())

# Convert predictions and test labels to tensors
predictions = torch.tensor(predictions)
y_test_tensor = torch.tensor(y_test.values.astype('float32'))

# Calculate MSE
mse = torch.nn.functional.mse_loss(predictions, y_test_tensor)
print(f"Test MSE: {mse}")

# Log your model and metrics with MLflow
mlflow.set_tracking_uri("s3_path")
with mlflow.start_run() as run:
    mlflow.log_metric("mse", mse.item())
    mlflow.pytorch.log_model(model, "model")

    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/torch"
    
    # Register the model in MLflow
    registered_model_name = "tokyo_rent_transformer"
    mlflow.register_model(model_uri, registered_model_name)

# Save the model locally
torch.save(model.state_dict(), 'torch_model.pth')


# loss could be better.. 
# could regularize my model, early stop, or tune hyperparameters, feature engineer or use a different model