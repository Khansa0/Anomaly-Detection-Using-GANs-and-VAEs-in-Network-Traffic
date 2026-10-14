# =======================================================================
# ABLATION C3: GAN + LIGHTGBM ON RAW FEATURES
# CICIDS2017 → PREPROCESSING → GAN SYNTHETIC ATTACKS → LIGHTGBM
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

import tensorflow as tf
from tensorflow.keras import layers, Model

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
# 3. TRAIN / TEST SPLIT
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
# 4. BUILD GAN
# =======================================================================

latent_dim = 32
num_features = X_train.shape[1]

def build_generator():
    inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dense(256, activation="relu")(x)

    # IMPORTANT:
    # X_scaled is in [0,1], therefore sigmoid is more consistent than tanh.
    output = layers.Dense(num_features, activation="sigmoid")(x)

    return Model(inputs, output, name="generator")


def build_discriminator():
    inputs = layers.Input(shape=(num_features,))
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.Dense(128, activation="relu")(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    return Model(inputs, output, name="discriminator")


generator = build_generator()
discriminator = build_discriminator()

discriminator.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
    loss="binary_crossentropy"
)

# Freeze discriminator only inside combined GAN
discriminator.trainable = False

z = layers.Input(shape=(latent_dim,))
fake_data = generator(z)
validity = discriminator(fake_data)

gan = Model(z, validity, name="gan")
gan.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
    loss="binary_crossentropy"
)

print("GAN Built Successfully")

# =======================================================================
# 5. TRAIN GAN ON ATTACK DATA ONLY
# =======================================================================

attack_idx = y_train != 0
attack_data = X_train[attack_idx]
attack_labels = y_train[attack_idx]

print("Attack Data:", attack_data.shape)

gan_epochs = 1500
batch_size = 64

d_losses = []
g_losses = []

start_gan = time.time()

for epoch in range(gan_epochs):

    # -----------------------------
    # Train Discriminator
    # -----------------------------
    idx = np.random.randint(0, attack_data.shape[0], batch_size)
    real = attack_data[idx]

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake = generator.predict(noise, verbose=0)

    discriminator.trainable = True

    d_loss_real = discriminator.train_on_batch(
        real,
        np.ones((batch_size, 1))
    )

    d_loss_fake = discriminator.train_on_batch(
        fake,
        np.zeros((batch_size, 1))
    )

    d_loss = (d_loss_real + d_loss_fake) / 2

    # -----------------------------
    # Train Generator
    # -----------------------------
    discriminator.trainable = False

    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    g_loss = gan.train_on_batch(
        noise,
        np.ones((batch_size, 1))
    )

    d_losses.append(float(d_loss))
    g_losses.append(float(g_loss))

    if epoch % 200 == 0:
        print(
            f"Epoch {epoch} | "
            f"D Loss: {d_loss:.4f} | "
            f"G Loss: {g_loss:.4f}"
        )

gan_time = time.time() - start_gan

print(f"GAN Training Finished in {gan_time:.1f}s")

# =======================================================================
# 6. GENERATE SYNTHETIC ATTACK SAMPLES — PER-CLASS BALANCING
# =======================================================================

classes_attack = np.unique(attack_labels)

counts = {
    c: np.sum(attack_labels == c)
    for c in classes_attack
}

max_count = max(counts.values())

# Cap target per attack class to avoid dataset explosion
TARGET_PER_CLASS = min(max_count, 50000)

synthetic_list = []
y_synthetic_list = []

for cls in classes_attack:

    cls_count = counts[cls]
    need = max(0, TARGET_PER_CLASS - cls_count)

    if need == 0:
        continue

    noise = np.random.normal(0, 1, (need, latent_dim))
    gen_samples = generator.predict(noise, verbose=0)

    synthetic_list.append(gen_samples)
    y_synthetic_list.append(np.full(need, cls, dtype=int))

if len(synthetic_list) > 0:
    synthetic_attacks = np.vstack(synthetic_list)
    y_synthetic = np.hstack(y_synthetic_list)
else:
    synthetic_attacks = np.empty((0, num_features))
    y_synthetic = np.array([], dtype=int)

print("Synthetic attacks generated:", synthetic_attacks.shape)

# =======================================================================
# 7. SHOW SYNTHETIC DISTRIBUTION
# =======================================================================

if synthetic_attacks.shape[0] > 0:

    plt.figure(figsize=(10, 4))
    plt.hist(synthetic_attacks.flatten(), bins=50)
    plt.title("GAN Synthetic Data Distribution")
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    print("\nSample GAN Synthetic Data (first 5 rows):")
    print(synthetic_attacks[:5])

# =======================================================================
# 8. COMBINE REAL TRAINING DATA + GAN SYNTHETIC DATA
# =======================================================================

X_gan_train = np.vstack((X_train, synthetic_attacks))
y_gan_train = np.hstack((y_train, y_synthetic))

print("GAN-Augmented Training Shape:", X_gan_train.shape)

# =======================================================================
# 9. LIGHTGBM ON GAN-AUGMENTED RAW FEATURES
# =======================================================================

classes_all = np.unique(y_gan_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes_all,
    y=y_gan_train
)

weight_dict = dict(zip(classes_all, weights))

sample_weights = np.array(
    [weight_dict[label] for label in y_gan_train],
    dtype=float
)

params = {
    "objective": "multiclass",
    "num_class": len(classes_all),
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "metric": "multi_logloss"
}

train_data = lgb.Dataset(
    X_gan_train,
    label=y_gan_train,
    weight=sample_weights
)

test_data = lgb.Dataset(
    X_test,
    label=y_test
)

start_lgb = time.time()

callbacks = [
    lgb.log_evaluation(period=10)
]

model_lgb_gan = lgb.train(
    params,
    train_data,
    valid_sets=[test_data],
    num_boost_round=200,
    callbacks=callbacks
)

lgb_time = time.time() - start_lgb

print(f"GAN + LightGBM Training Finished in {lgb_time:.1f}s")

# =======================================================================
# 10. EVALUATION — GAN + LIGHTGBM
# =======================================================================

print("\nEvaluating GAN + LightGBM on Raw Features...")

y_pred_probs = model_lgb_gan.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_test, y_pred_classes)
prec = precision_score(y_test, y_pred_classes, average="weighted", zero_division=0)
rec = recall_score(y_test, y_pred_classes, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred_classes, average="weighted", zero_division=0)
macro_f1 = f1_score(y_test, y_pred_classes, average="macro", zero_division=0)

print("\n=== Evaluation Metrics: GAN + LightGBM ===")
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
# 11. MINORITY-CLASS F1
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

print(f"\nMinority-Class F1 : {minority_f1:.4f} ({minority_f1*100:.2f}%)")

# =======================================================================
# 12. CONFUSION MATRIX
# =======================================================================

cm = confusion_matrix(
    y_test,
    y_pred_classes,
    labels=range(len(classes))
)

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

plt.title("Confusion Matrix — GAN + LightGBM on Raw Features")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# =======================================================================
# 13. GAN LOSS CURVE
# =======================================================================

plt.figure(figsize=(8, 5))
plt.plot(d_losses, label="Discriminator Loss")
plt.plot(g_losses, label="Generator Loss")
plt.title("GAN Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

# =======================================================================
# 14. SAVE ABLATION RESULT
# =======================================================================

ablation_result = {
    "Configuration": "C3: GAN + LightGBM",
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "Weighted F1": f1,
    "Macro F1": macro_f1,
    "Minority F1": minority_f1,
    "GAN Training Time": gan_time,
    "LightGBM Training Time": lgb_time,
    "Total Training Time": gan_time + lgb_time
}

print("\n=== Ablation Result ===")
print(ablation_result)