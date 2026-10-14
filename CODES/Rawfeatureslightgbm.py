# =======================================================================
# ABLATION C1: LightGBM ON RAW FEATURES
# CICIDS2017 → PREPROCESSING → LIGHTGBM
# =======================================================================

import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

# =======================================================================
# 1. LOAD DATASET
# =======================================================================

DATASET_PATH = r"D:\Khanssa\CICIDS-2017"

csv_files = [f for f in os.listdir(DATASET_PATH) if f.endswith(".csv")]

df_list = []
for file in csv_files:
    path = os.path.join(DATASET_PATH, file)
    print("Loading:", path)
    df_list.append(pd.read_csv(path, low_memory=False))

df = pd.concat(df_list, axis=0)
print("Dataset Loaded:", df.shape)

# =======================================================================
# 2. PREPROCESSING
# =======================================================================

df.drop_duplicates(inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df.columns = df.columns.str.strip()

label_encoder = LabelEncoder()
df["Label"] = label_encoder.fit_transform(df["Label"])
classes = label_encoder.classes_

X = df.drop("Label", axis=1).values
y = df["Label"].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("Preprocessing Done — Shape:", X_scaled.shape)

# =======================================================================
# 3. TRAIN/TEST SPLIT
# =======================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Train/Test Split Completed")

# =======================================================================
# 4. LIGHTGBM ON RAW FEATURES
# =======================================================================

classes_all = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes_all,
    y=y_train
)

weight_dict = dict(zip(classes_all, weights))
sample_weights = np.array([weight_dict[c] for c in y_train], dtype=float)

params = {
    "objective": "multiclass",
    "num_class": len(classes_all),
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "metric": "multi_logloss",
}

train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
test_data = lgb.Dataset(X_test, label=y_test)

start_lgb = time.time()

callbacks = [lgb.log_evaluation(period=10)]

model_lgb_raw = lgb.train(
    params,
    train_data,
    valid_sets=[test_data],
    num_boost_round=200,
    callbacks=callbacks
)

lgb_time = time.time() - start_lgb

print(f"LightGBM Raw Feature Training Finished in {lgb_time:.1f}s")

# =======================================================================
# 5. EVALUATION
# =======================================================================

print("\nEvaluating LightGBM on Raw Features...")

y_pred_probs = model_lgb_raw.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_test, y_pred_classes)
prec = precision_score(y_test, y_pred_classes, average="weighted", zero_division=0)
rec = recall_score(y_test, y_pred_classes, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred_classes, average="weighted", zero_division=0)
macro_f1 = f1_score(y_test, y_pred_classes, average="macro", zero_division=0)

print("\n=== Evaluation Metrics: LightGBM on Raw Features ===")
print(f"Accuracy    : {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision   : {prec:.4f} ({prec*100:.2f}%)")
print(f"Recall      : {rec:.4f} ({rec*100:.2f}%)")
print(f"F1-Score    : {f1:.4f} ({f1*100:.2f}%)")
print(f"Macro F1    : {macro_f1:.4f} ({macro_f1*100:.2f}%)")

print("\n=== Classification Report ===")
print(classification_report(
    y_test,
    y_pred_classes,
    target_names=classes,
    digits=4,
    zero_division=0
))

# =======================================================================
# 6. MINORITY-CLASS F1
# =======================================================================

report = classification_report(
    y_test,
    y_pred_classes,
    target_names=classes,
    output_dict=True,
    zero_division=0
)

minority_classes = [
    "Bot",
    "DoS Slowhttptest",
    "DoS slowloris",
    "FTP-Patator",
    "Heartbleed",
    "Infiltration",
    "SSH-Patator",
    "Web Attack � Brute Force",
    "Web Attack � Sql Injection",
    "Web Attack � XSS"
]

minority_f1_scores = []

for cls in minority_classes:
    if cls in report:
        minority_f1_scores.append(report[cls]["f1-score"])

minority_f1 = np.mean(minority_f1_scores)

print(f"Minority-Class F1 : {minority_f1:.4f} ({minority_f1*100:.2f}%)")

# =======================================================================
# 7. CONFUSION MATRIX
# =======================================================================

cm = confusion_matrix(y_test, y_pred_classes, labels=range(len(classes)))

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes,
    cbar_kws={"label": "Count"}
)

plt.title("Confusion Matrix — LightGBM on Raw Features")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# =======================================================================
# 8. SAVE ABLATION RESULT
# =======================================================================

ablation_result = {
    "Configuration": "C1: LightGBM on Raw Features",
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "Weighted F1": f1,
    "Macro F1": macro_f1,
    "Minority F1": minority_f1,
    "Training Time": lgb_time
}

print("\n=== Ablation Result ===")
print(ablation_result)