"""
GymForm AI — Improved Training Pipeline (v2)
=============================================
Fixes over the original train_model.py:
  1. Feature normalization (StandardScaler)
  2. Stratified Train/Val/Test split (70/15/15)
  3. Class-weighted CrossEntropyLoss (handles imbalance)
  4. Learning rate scheduler (CosineAnnealingLR)
  5. Early stopping on validation loss
  6. Per-class accuracy + confusion matrix
  7. Saves scaler params (needed for inference in the app)
  8. Optionally uses augmented dataset if available

Run: python train_model_v2.py
Outputs:
  - gym_model_v2.pt          (model weights)
  - scaler_params.json        (mean + std for each feature — needed by the app)
  - training_report.txt       (accuracy, confusion matrix, per-class metrics)

Original train_model.py is NOT modified.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import json
import os
import sys

# ── Config ──────────────────────────────────────────────────────────
AUGMENTED_CSV = 'dataset_augmented.csv'
ORIGINAL_CSV = 'dataset_fullbody.csv'
MODEL_OUTPUT = 'gym_model_v2.pt'
SCALER_OUTPUT = 'scaler_params.json'
REPORT_OUTPUT = 'training_report.txt'

EPOCHS = 300
BATCH_SIZE = 64
LEARNING_RATE = 0.003
PATIENCE = 25           # early stopping patience (epochs)
RANDOM_SEED = 42
# ────────────────────────────────────────────────────────────────────

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

LABELS_MAP = {
    'Bad Curl': 0, 'Good Curl': 1,
    'Bad Squat': 2, 'Good Squat': 3,
    'Bad Raise': 4, 'Good Raise': 5,
    'Bad Shoulder': 6, 'Good Shoulder': 7,
    'Bad Tricep': 8, 'Good Tricep': 9,
}
LABELS_REVERSE = {v: k for k, v in LABELS_MAP.items()}


class GymModelV2(nn.Module):
    """
    Improved architecture over the original:
    - Wider layers (128, 64, 32 vs 64, 32)
    - BatchNorm for training stability
    - LeakyReLU to avoid dead neurons
    - Higher dropout (0.3) for better generalization
    """
    def __init__(self, num_features=8, num_classes=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),

            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def load_data():
    """Load augmented dataset if available, otherwise fall back to original."""
    if os.path.exists(AUGMENTED_CSV):
        print(f"[OK] Using augmented dataset: {AUGMENTED_CSV}")
        df = pd.read_csv(AUGMENTED_CSV)
    else:
        print(f"[WARN] Augmented dataset not found. Using original: {ORIGINAL_CSV}")
        print(f"       Run 'python augment_dataset.py' first for better results!")
        df = pd.read_csv(ORIGINAL_CSV)
    
    df['label_encoded'] = df['label'].map(LABELS_MAP)
    
    # Drop any rows with unmapped labels
    df = df.dropna(subset=['label_encoded'])
    df['label_encoded'] = df['label_encoded'].astype(int)
    
    feature_cols = ['l_elbow', 'r_elbow', 'l_shoulder', 'r_shoulder',
                    'l_hip', 'r_hip', 'l_knee', 'r_knee']
    
    X = df[feature_cols].values
    y = df['label_encoded'].values
    
    return X, y, df['label'].values


def compute_class_weights(y):
    """Inverse frequency weighting — gives more importance to rare classes."""
    class_counts = np.bincount(y, minlength=10)
    # Avoid division by zero for classes with no samples
    class_counts = np.maximum(class_counts, 1)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(weights)  # normalize
    return torch.FloatTensor(weights)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on a dataloader, return loss and accuracy."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * len(y_batch)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += len(y_batch)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # ── Load Data ──
    X, y, labels = load_data()
    print(f"Total samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {len(np.unique(y))}")
    print()
    
    # ── Stratified Split: 70/15/15 ──
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # ── Feature Normalization ──
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Save scaler params for use in the app (mobile inference needs these)
    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'std': scaler.scale_.tolist(),
        'feature_names': ['l_elbow', 'r_elbow', 'l_shoulder', 'r_shoulder',
                          'l_hip', 'r_hip', 'l_knee', 'r_knee'],
    }
    with open(SCALER_OUTPUT, 'w') as f:
        json.dump(scaler_params, f, indent=2)
    print(f"[OK] Scaler params saved to {SCALER_OUTPUT}")
    
    # ── Convert to PyTorch tensors + DataLoaders ──
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # ── Class-Weighted Loss ──
    class_weights = compute_class_weights(y_train)
    print(f"Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # ── Model ──
    model = GymModelV2().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print()
    
    # ── Training Loop with Early Stopping ──
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_state = None
    
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val Acc':>8} | {'LR':>10}")
    print("-" * 60)
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
        
        train_loss /= len(X_train)
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Print every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>5} | {train_loss:>10.4f} | {val_loss:>10.4f} | {val_acc:>7.1%} | {current_lr:>10.6f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n[STOP] Early stopping at epoch {epoch} (best was epoch {best_epoch})")
                break
    
    # Load best model
    model.load_state_dict(best_state)
    print(f"\n[OK] Loaded best model from epoch {best_epoch} (val_loss: {best_val_loss:.4f})")
    
    # ── Final Evaluation on Test Set ──
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    
    print(f"\n{'='*60}")
    print(f"FINAL TEST RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_acc:.1%}")
    print(f"Test Loss:     {test_loss:.4f}")
    
    # Per-class report
    target_names = [LABELS_REVERSE[i] for i in range(10)]
    report = classification_report(test_labels, test_preds, target_names=target_names, zero_division=0)
    print(f"\nPer-Class Report:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    print("Confusion Matrix:")
    print(f"{'':>15}", end='')
    for name in target_names:
        print(f"{name[:8]:>9}", end='')
    print()
    for i, row in enumerate(cm):
        print(f"{target_names[i]:>15}", end='')
        for val in row:
            print(f"{val:>9}", end='')
        print()
    
    # ── Save Model ──
    torch.save(model.state_dict(), MODEL_OUTPUT)
    print(f"\n[OK] Model saved to {MODEL_OUTPUT}")
    
    # ── Save Report ──
    with open(REPORT_OUTPUT, 'w') as f:
        f.write(f"GymForm AI — Training Report\n")
        f.write(f"{'='*60}\n")
        f.write(f"Dataset: {AUGMENTED_CSV if os.path.exists(AUGMENTED_CSV) else ORIGINAL_CSV}\n")
        f.write(f"Total samples: {len(X)}\n")
        f.write(f"Train/Val/Test: {len(X_train)}/{len(X_val)}/{len(X_test)}\n")
        f.write(f"Architecture: GymModelV2 ({total_params:,} params)\n")
        f.write(f"Best epoch: {best_epoch}/{EPOCHS}\n")
        f.write(f"Test accuracy: {test_acc:.1%}\n")
        f.write(f"Test loss: {test_loss:.4f}\n\n")
        f.write(f"Per-Class Report:\n{report}\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"{'':>15}")
        for name in target_names:
            f.write(f"{name[:8]:>9}")
        f.write('\n')
        for i, row in enumerate(cm):
            f.write(f"{target_names[i]:>15}")
            for val in row:
                f.write(f"{val:>9}")
            f.write('\n')
    
    print(f"[OK] Report saved to {REPORT_OUTPUT}")
    
    # ── Comparison with original model ──
    print(f"\n{'='*60}")
    print(f"COMPARISON WITH ORIGINAL")
    print(f"{'='*60}")
    print(f"  Original: 8->64->32->10, no normalization, no class weights, 80/20 split")
    print(f"  V2:       8->128->64->32->10, BatchNorm, StandardScaler, class weights, 70/15/15 split")
    print(f"  V2 test accuracy: {test_acc:.1%}")


if __name__ == '__main__':
    main()
