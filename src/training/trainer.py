import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.models.pytorch.bilstm import AGTSFNet, FocalLoss


# ── 1. Dataset Class ──────────────────────────────────────────────────────────
class TrafficDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── 2. Trainer Class ──────────────────────────────────────────────────────────
class Trainer:
    def __init__(self, model, device, lr=0.001):
        self.model     = model.to(device)
        self.device    = device
        self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=3, factor=0.5, verbose=True
        )
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits, _ = self.model(X, X)  # same input for both streams
            loss = self.criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def eval_epoch(self, loader):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                logits, _ = self.model(X, X)
                loss = self.criterion(logits, y)
                total_loss += loss.item()
                preds = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.cpu().numpy())

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)
        binary_preds = (all_preds > 0.5).astype(int)

        f1  = f1_score(all_labels, binary_preds, zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except:
            auc = 0.0

        return total_loss / len(loader), f1, auc

    def fit(self, train_loader, val_loader, epochs=20):
        print(f"Training on {self.device} for {epochs} epochs...\n")
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_f1, val_auc = self.eval_epoch(val_loader)
            self.scheduler.step(val_loss)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_f1)

            print(f"Epoch {epoch:02d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val F1: {val_f1:.4f} | "
                  f"Val AUC: {val_auc:.4f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                os.makedirs("experiments/results", exist_ok=True)
                torch.save(self.model.state_dict(),
                           "experiments/results/best_model.pt")
                print(f"  ✅ Best model saved!")

        print("\nTraining complete!")
        return self.history


# ── 3. Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load processed data
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    X_val   = np.load("data/processed/X_val.npy")
    y_val   = np.load("data/processed/y_val.npy")

    print(f"Train: {X_train.shape} | Val: {X_val.shape}")

    # Dataloaders
    train_ds = TrafficDataset(X_train, y_train)
    val_ds   = TrafficDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

    # Model
    model = AGTSFNet(
        temporal_input_size=X_train.shape[2],
        spatial_input_size=X_train.shape[2]
    )

    # Train
    trainer = Trainer(model, device, lr=0.001)
    history = trainer.fit(train_loader, val_loader, epochs=20)

    print("\n🎉 Model training complete!")
    print(f"Best Val Loss: {trainer.best_val_loss:.4f}")