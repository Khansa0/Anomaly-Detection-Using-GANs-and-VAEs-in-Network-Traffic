# ============================================================
# CICIDS2017 → Preprocessing → VAE → Anomaly Detection
# WITHOUT GAN (baseline)
# ============================================================

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# LOAD DATA
# ============================================================

DATASET_PATH = r"D:\Khanssa\CICIDS-2017/"

csv_files = [f for f in os.listdir(DATASET_PATH) if f.endswith(".csv")]
df = pd.concat([pd.read_csv(os.path.join(DATASET_PATH,f), low_memory=False) for f in csv_files])

df.replace([np.inf,-np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df.columns = df.columns.str.strip()

df['Label'] = LabelEncoder().fit_transform(df['Label'])

X = df.drop("Label", axis=1)
y = df["Label"]

X = MinMaxScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test  = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test  = torch.tensor(y_test.values).long().to(device)

input_dim = X_train.shape[1]

# ============================================================
# VAE ARCHITECTURE
# ============================================================

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim,128), nn.ReLU())
        self.mean = nn.Linear(128,16)
        self.logvar = nn.Linear(128,16)

    def forward(self, x):
        h = self.fc(x)
        return self.mean(h), self.logvar(h)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(16,128), nn.ReLU(),
            nn.Linear(128,input_dim), nn.Sigmoid()
        )
    def forward(self, z):
        return self.fc(z)

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        mean, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        recon = self.decoder(z)
        return recon, mean, logvar

vae = VAE().to(device)
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# ============================================================
# TRAIN VAE
# ============================================================

loader = DataLoader(TensorDataset(X_train), batch_size=512, shuffle=True)

for epoch in range(40):
    loss_total = 0
    for batch in loader:
        x = batch[0]
        recon, mean, logvar = vae(x)

        recon_loss = torch.mean((x - recon)**2)
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total += loss.item()

    print(f"Epoch {epoch+1} | Loss {loss_total/len(loader):.4f}")

# ========================= 
# 8) EVALUATION & LOGS — VAE + LightGBM
# =========================

# Compute reconstruction errors
vae.eval()
with torch.no_grad():
    recon, _, _ = vae(X_test)
    errors = torch.mean((X_test - recon)**2, dim=1).cpu().numpy()

# Convert y_test to binary: 0 = Normal, 1 = Attack
y_binary = (y_test.cpu().numpy() != 0).astype(int)

# ROC-AUC using reconstruction errors
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

roc_auc = roc_auc_score(y_binary, errors)
print(f"ROC-AUC (VAE anomaly score): {roc_auc:.4f}")

# Optional: Choose threshold to compute metrics (e.g., 95th percentile of errors)
threshold = np.percentile(errors, 95)
y_pred_binary = (errors >= threshold).astype(int)

acc  = (y_binary == y_pred_binary).mean()
prec = precision_score(y_binary, y_pred_binary)
rec  = recall_score(y_binary, y_pred_binary)
f1   = f1_score(y_binary, y_pred_binary)

print("\n=== Binary Classification Metrics (Normal vs Attack) ===")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-Score  : {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_binary, y_pred_binary)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Normal","Attack"],
            yticklabels=["Normal","Attack"])
plt.title("Confusion Matrix — VAE Anomaly Detection")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

