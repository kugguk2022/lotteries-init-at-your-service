import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# File paths - using relative paths (create these files or adjust paths as needed)
sequence_a_path = "../data/g1.csv"  # Adjust this path to your actual data location
sequence_b_path = "../data/poi.csv"  # Adjust this path to your actual data location

# Load sequences from CSV files with error handling
try:
    sequence_a = pd.read_csv(sequence_a_path, header=None).squeeze("columns")
    sequence_b = pd.read_csv(sequence_b_path, header=None).squeeze("columns")
except FileNotFoundError as e:
    print(f"Error: Could not find data files. Please check paths: {e}")
    print("Creating sample data for demonstration...")
    # Create sample data if files don't exist
    np.random.seed(42)
    sequence_a = pd.Series(np.random.randn(1000).cumsum())
    sequence_b = pd.Series(0.8 * sequence_a + 0.2 * np.random.randn(1000))

# Ensure both sequences are the same length
min_length = min(len(sequence_a), len(sequence_b))
sequence_a = sequence_a[:min_length]
sequence_b = sequence_b[:min_length]

# Combine into a DataFrame and ensure numeric data
df = pd.DataFrame(
    {
        "sequence_a": pd.to_numeric(sequence_a, errors="coerce"),
        "sequence_b": pd.to_numeric(sequence_b, errors="coerce"),
    }
)

# Fix deprecated fillna method
df["sequence_a"] = df["sequence_a"].ffill()
df["sequence_b"] = df["sequence_b"].ffill()

# Normalize sequences for better stability
scaler_a = StandardScaler()
scaler_b = StandardScaler()
df["sequence_a"] = scaler_a.fit_transform(df[["sequence_a"]])
df["sequence_b"] = scaler_b.fit_transform(df[["sequence_b"]])

# Feature Engineering
# 1. Rolling correlation
window_size = 10
df["rolling_corr"] = df["sequence_a"].rolling(window=window_size).corr(df["sequence_b"])

# 2. Lagged features
max_lag = 5
for lag in range(1, max_lag + 1):
    df[f"sequence_a_lag_{lag}"] = df["sequence_a"].shift(lag)
    df[f"sequence_b_lag_{lag}"] = df["sequence_b"].shift(lag)

# Drop rows with NaN values created by lagging
df = df.dropna().reset_index(drop=True)


# Define Dataset with both sequences as inputs
class DualSequenceDataset(Dataset):
    def __init__(self, df, target_col, seq_len=10):
        self.data = df
        self.target_col = target_col
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x_a = self.data.iloc[idx : idx + self.seq_len].drop(columns=[self.target_col]).values
        y = self.data.iloc[idx + self.seq_len][self.target_col]
        return torch.tensor(x_a, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# Calculate input_dim and adjust n_heads accordingly
input_dim = len(df.columns) - 1  # Exclude target column
n_heads = 1  # Default to 1 head if input_dim is not large enough

# Find the largest `n_heads` that divides `input_dim`
for i in range(2, input_dim + 1):
    if input_dim % i == 0:
        n_heads = i


# Dual-Input Transformer Model
class DualInputTransformer(nn.Module):
    def __init__(self, input_dim, seq_len, n_heads, hidden_dim=64, n_layers=2, dropout=0.2):
        super(DualInputTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=n_heads, dim_feedforward=hidden_dim, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(input_dim * seq_len, 1)

    def forward(self, x_a):
        x = x_a.permute(1, 0, 2)  # Shape: (seq_len, batch_size, input_dim)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2).reshape(x.shape[1], -1)  # Flatten for output
        return self.fc_out(x)


# Instantiate and configure the model
seq_len = 10
model = DualInputTransformer(
    input_dim=input_dim, seq_len=seq_len, n_heads=n_heads, hidden_dim=128, n_layers=4, dropout=0.3
)

# Prepare the data for training
target_column = "sequence_b"
dataset = DualSequenceDataset(df, target_column, seq_len=seq_len)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Training configurations
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
loss_fn = nn.MSELoss()
n_epochs = 50

# Training loop
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    for x_a_batch, y_batch in dataloader:
        optimizer.zero_grad()
        predictions = model(x_a_batch)
        loss = loss_fn(predictions, y_batch.unsqueeze(1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss / len(dataloader)}")

print("Model training complete!")


# Inverse transform the scaler to compare predictions in the original scale
def inverse_transform_prediction(prediction):
    return scaler_b.inverse_transform(prediction.detach().numpy())


# Optional AR Model as baseline
from statsmodels.tsa.ar_model import AutoReg

ar_model = AutoReg(df["sequence_a"], lags=5).fit()
predicted_b = ar_model.predict(start=len(df), end=len(df))
print(
    "AR Model predicted next value of sequence_b (scaled):",
    scaler_b.inverse_transform([[predicted_b.values[0]]])[0, 0],
)
