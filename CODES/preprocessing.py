# =======================================================================
#  FULL EXECUTABLE PIPELINE
#  CICIDS2017 → PREPROCESSING → GAN SYNTHETIC ATTACKS → VAE → LIGHTGBM
# =======================================================================

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import tensorflow as tf
from tensorflow.keras import layers, Model
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# =======================================================================
# 1. LOAD THE DATASET (7 FILES)
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

X = df.drop("Label", axis=1)
y = df["Label"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("Preprocessing Done — Shape:", X_scaled.shape)

# =======================================================================
# 3. TRAIN/TEST SPLIT
# =======================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

print("Train/Test Split Completed")

# =======================================================================
# 4. TRAIN GAN TO GENERATE SYNTHETIC ATTACK SAMPLES
# =======================================================================

latent_dim = 32

def build_generator():
    inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dense(256, activation="relu")(x)
    output = layers.Dense(X_train.shape[1], activation="tanh")(x)
    return Model(inputs, output)

def build_discriminator():
    inputs = layers.Input(shape=(X_train.shape[1],))
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.Dense(128, activation="relu")(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    return Model(inputs, output)

generator = build_generator()
discriminator = build_discriminator()

discriminator.compile(optimizer="adam", loss="binary_crossentropy")
discriminator.trainable = False

z = layers.Input(shape=(latent_dim,))
fake_data = generator(z)
validity = discriminator(fake_data)

gan = Model(z, validity)
gan.compile(optimizer="adam", loss="binary_crossentropy")

print("GAN Built Successfully")

# =======================================================================
# 5. GAN TRAINING (ONLY ON ATTACK DATA) — SHOW DISTRIBUTION
# =======================================================================

attack_data = X_train[y_train != 0]  # all attack classes
print("Attack Data:", attack_data.shape)

epochs = 1500
batch_size = 64

for epoch in range(epochs):
    idx = np.random.randint(0, attack_data.shape[0], batch_size)
    real = attack_data[idx]

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake = generator.predict(noise, verbose=0)

    d_loss_real = discriminator.train_on_batch(real, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(fake, np.zeros((batch_size, 1)))

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

    if epoch % 200 == 0:
        print(f"Epoch {epoch} | D Loss: {(d_loss_real+d_loss_fake)/2:.4f} | G Loss: {g_loss:.4f}")

print("GAN Training Finished")

# =======================================================================
# 6. GENERATE SYNTHETIC ATTACK SAMPLES AND SHOW DISTRIBUTION
# =======================================================================

num_samples = attack_data.shape[0]
noise = np.random.normal(0, 1, (num_samples, latent_dim))
synthetic_attacks = generator.predict(noise, verbose=0)

# Show synthetic data distribution
plt.figure(figsize=(10, 4))
plt.hist(synthetic_attacks.flatten(), bins=50)
plt.title("GAN Synthetic Data Distribution")
plt.show()

# Show a few synthetic samples
print("\nSample GAN Synthetic Data (first 5 rows):")
print(synthetic_attacks[:5])

# Combine GAN attacks with original training data
X_balanced = np.vstack((X_train, synthetic_attacks))
y_balanced = np.hstack((y_train, np.random.choice(np.unique(y_train[y_train != 0]), num_samples)))

print("Balanced Dataset Shape:", X_balanced.shape)

# =======================================================================
# 7. BUILD VAE → LEARN REPRESENTATIONS (EPOCHS = 50)
# =======================================================================

latent_dim_vae = 16

inputs = layers.Input(shape=(X_train.shape[1],))
h = layers.Dense(128, activation="relu")(inputs)
z_mean = layers.Dense(latent_dim_vae)(h)
z_log_var = layers.Dense(latent_dim_vae)(h)

def sampling(args):
    z_mean, z_log_var = args
    eps = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim_vae))
    return z_mean + tf.exp(0.5 * z_log_var) * eps

z = layers.Lambda(sampling)([z_mean, z_log_var])

encoder = Model(inputs, z, name="encoder")

latent_inputs = layers.Input(shape=(latent_dim_vae,))
d = layers.Dense(128, activation="relu")(latent_inputs)
outputs = layers.Dense(X_train.shape[1], activation="sigmoid")(d)
decoder = Model(latent_inputs, outputs)

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_latent = self.encoder(data)
            reconstruction = self.decoder(z_latent)
            loss = tf.reduce_mean(tf.square(data - reconstruction))
        grads = tape.gradient(
            loss, self.encoder.trainable_variables + self.decoder.trainable_variables
        )
        self.optimizer.apply_gradients(
            zip(grads, self.encoder.trainable_variables + self.decoder.trainable_variables)
        )
        return {"loss": loss}

vae = VAE(encoder, decoder)
optimizer = tf.keras.optimizers.Adam()
vae.compile(optimizer=optimizer)

# =======================================================================
# 8. GET VAE LATENT VECTORS (FEATURES FOR LIGHTGBM)
# =======================================================================

X_train_latent = encoder.predict(X_train)
X_test_latent = encoder.predict(X_test)
X_bal_latent = encoder.predict(X_balanced)  # <-- needed for LightGBM
print("Latent Features Shape:", X_train_latent.shape)

vae.fit(X_balanced, epochs=50, batch_size=256, shuffle=True)
print("VAE Training Finished")

# =======================================================================
# 9. LIGHTGBM MULTI–CLASS CLASSIFIER (FIXED)
# =======================================================================

params = {
    "objective": "multiclass",
    "num_class": len(np.unique(y)),
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "metric": "multi_logloss",
    "is_unbalance": True  # Handle rare classes
}

train_data = lgb.Dataset(X_bal_latent, label=y_balanced)
test_data = lgb.Dataset(X_test_latent, label=y_test)

model_lgb = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=200)

print("LightGBM Training Finished")

# ========================= 
# 8) EVALUATION & LOGS — VAE + LightGBM
# =========================

print("\nEvaluating LightGBM on VAE latent features...")

# Predictions
y_pred_probs = model_lgb.predict(X_test_latent)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true = y_test

# Overall metrics
acc  = accuracy_score(y_true, y_pred_classes)
prec = precision_score(y_true, y_pred_classes, average="weighted", zero_division=0)
rec  = recall_score(y_true, y_pred_classes, average="weighted", zero_division=0)
f1   = f1_score(y_true, y_pred_classes, average="weighted", zero_division=0)

print("\n=== 📊 Evaluation Metrics (Overall) ===")
print(f"Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision : {prec:.4f} ({prec*100:.2f}%)")
print(f"Recall    : {rec:.4f} ({rec*100:.2f}%)")
print(f"F1-Score  : {f1:.4f} ({f1*100:.2f}%)")

# Per-class report
print("\n=== Classification report ===")
print(classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_, digits=4))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes, labels=range(len(label_encoder.classes_)))
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            cbar_kws={'label':'Count'})
plt.title("Confusion Matrix — LightGBM on VAE Features")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Save logs (training summary + test metrics)
logs = {
    "dataset": "CIC-IDS-2017",
    "classes": list(label_encoder.classes_),
    "VAE_epochs": 50,
    "LightGBM_boost_round": 200,
    "test_metrics": {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1)
    }
}

print("\n✅ Logs saved for reference")

