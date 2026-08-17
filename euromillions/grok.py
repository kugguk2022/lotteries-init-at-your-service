import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.ar_model import AutoReg
from torch.utils.data import DataLoader, Dataset


# Dual-Input Transformer Model
class DualInputTransformer(nn.Module):
    def __init__(self, input_dim, seq_len, n_heads, hidden_dim=64, n_layers=2, dropout=0.2):
        super(DualInputTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=n_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(input_dim * seq_len, 1)

    def forward(self, x_a):
        # x_a shape: (batch_size, seq_len, input_dim)
        x = self.transformer_encoder(x_a)
        x = x.reshape(x.shape[0], -1)  # Flatten for output
        return self.fc_out(x)


# Define Dataset with both sequences as inputs
class DualSequenceDataset(Dataset):
    def __init__(self, df, target_col, seq_len=10):
        self.data = df
        self.target_col = target_col
        self.seq_len = seq_len
        self.features = df.drop(columns=[target_col]).values.astype(np.float32)
        self.targets = df[target_col].values.astype(np.float32)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x_a = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return torch.tensor(x_a, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Train a Dual-Input Transformer on sequence data.")
    parser.add_argument("--g", type=Path, required=True, help="Path to 'g' (sequence A) CSV")
    parser.add_argument("--poi", type=Path, required=True, help="Path to 'poi' (sequence B/target) CSV")
    parser.add_argument("--out-dir", type=Path, default=Path("."), help="Directory to save outputs")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--seq-len", type=int, default=10, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load sequences
    try:
        sequence_a = pd.read_csv(args.g, header=None).squeeze("columns")
        sequence_b = pd.read_csv(args.poi, header=None).squeeze("columns")
    except FileNotFoundError as e:
        print(f"Error: Could not find data files: {e}")
        sys.exit(1)

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

    # Fix NaNs
    df["sequence_a"] = df["sequence_a"].ffill()
    df["sequence_b"] = df["sequence_b"].ffill()

    # Normalize sequences
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

    # Drop rows with NaN values created by lagging/rolling
    df = df.dropna().reset_index(drop=True)

    # Prepare Data
    target_column = "sequence_b"
    dataset = DualSequenceDataset(df, target_column, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model Setup
    input_dim = len(df.columns) - 1
    # Simple logic for n_heads
    n_heads = 1
    for i in range(2, input_dim + 1):
        if input_dim % i == 0:
            n_heads = i

    model = DualInputTransformer(
        input_dim=input_dim,
        seq_len=args.seq_len,
        n_heads=n_heads,
        hidden_dim=128,
        n_layers=4,
        dropout=0.3,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    # Training
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = loss_fn(predictions, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {epoch_loss / len(dataloader):.6f}")

    print("Model training complete!")

    # Optional: Save model
    model_path = args.out_dir / "grok_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    # Baseline AR Model
    try:
        ar_model = AutoReg(df["sequence_a"], lags=5).fit()
        predicted_b = ar_model.predict(start=len(df), end=len(df))
        val_scaled = predicted_b.values[0]
        val_orig = scaler_b.inverse_transform([[val_scaled]])[0, 0]
        print(f"AR Model predicted next POI (scaled): {val_scaled:.4f}")
        print(f"AR Model predicted next POI (original): {val_orig:.4f}")
    except Exception as e:
        print(f"AR Model extraction failed: {e}")

if __name__ == "__main__":
    main()

