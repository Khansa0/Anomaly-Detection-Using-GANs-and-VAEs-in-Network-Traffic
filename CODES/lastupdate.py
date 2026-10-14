# =======================================================================
# FINAL PIPELINE (IMPROVED MINOR-CLASS HANDLING)
# CICIDS2017 → PREPROCESSING → GAN SYNTHETIC ATTACKS → β-VAE → LightGBM
# =======================================================================

import os
import time
import numpy as np
import pandas as pd
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
import lightgbm as lgb

# --------------------------
# 1) LOAD DATA
# --------------------------
DATASET_PATH = r"D:\Khanssa\CICIDS-2017"
csv_files = [f for f in os.listdir(DATASET_PATH) if f.endswith(".csv")]
df_list = []
for file in csv_files:
    path = os.path.join(DATASET_PATH, file)
    print("Loading:", path)
    df_list.append(pd.read_csv(path, low_memory=False))
df = pd.concat(df_list, axis=0)
print("Dataset Loaded:", df.shape)

# --------------------------
# 2) PREPROCESSING
# --------------------------
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

# --------------------------
# 3) TRAIN/TEST SPLIT
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)
print("Train/Test Split Completed")

# --------------------------
# 4) BUILD GAN (unchanged)
# --------------------------
latent_dim = 32
num_features = X_train.shape[1]

def build_generator():
    inp = layers.Input(shape=(latent_dim,))
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.Dense(256, activation="relu")(x)
    out = layers.Dense(num_features, activation="tanh")(x)
    return Model(inp, out, name="generator")

def build_discriminator():
    inp = layers.Input(shape=(num_features,))
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return Model(inp, out, name="discriminator")

generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer="adam", loss="binary_crossentropy")
discriminator.trainable = False

z = layers.Input(shape=(latent_dim,))
validity = discriminator(generator(z))
gan = Model(z, validity)
gan.compile(optimizer="adam", loss="binary_crossentropy")
print("GAN Built Successfully")

# --------------------------
# 5) GAN TRAINING (ATTACK DATA)
# --------------------------
attack_idx = y_train != 0
attack_data = X_train[attack_idx]
attack_labels = y_train[attack_idx]
print("Attack Data:", attack_data.shape)

gan_epochs = 1500
batch_size = 64

start_gan = time.time()
for epoch in range(gan_epochs):
    idx = np.random.randint(0, attack_data.shape[0], batch_size)
    real = attack_data[idx]

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake = generator.predict(noise, verbose=0)

    d_loss_real = discriminator.train_on_batch(real, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(fake, np.zeros((batch_size, 1)))

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

    if epoch % 200 == 0:
        print(f"Epoch {epoch} | D Loss: {(d_loss_real + d_loss_fake)/2:.4f} | G Loss: {g_loss:.4f}")
gan_time = time.time() - start_gan
print(f"GAN Training Finished in {gan_time:.1f}s")

# --------------------------
# 6) GENERATE SYNTHETIC ATTACK SAMPLES (per-class)
# --------------------------
# Generate synthetic samples for each attack class to augment rare classes.
classes_attack = np.unique(attack_labels)
counts = {c: np.sum(attack_labels == c) for c in classes_attack}
max_count = max(counts.values())

# To avoid enormous dataset, cap per-class target to a reasonable value (e.g., 50k)
TARGET_PER_CLASS = min(max_count, 50000)

synthetic_list = []
y_synthetic_list = []
for cls in classes_attack:
    cls_count = counts[cls]
    need = max(0, TARGET_PER_CLASS - cls_count)
    if need == 0:
        continue
    noise = np.random.normal(0, 1, (need, latent_dim))
    gen = generator.predict(noise, verbose=0)
    synthetic_list.append(gen)
    y_synthetic_list.append(np.full(need, cls, dtype=int))

if len(synthetic_list) > 0:
    synthetic_attacks = np.vstack(synthetic_list)
    y_synthetic = np.hstack(y_synthetic_list)
else:
    synthetic_attacks = np.empty((0, num_features))
    y_synthetic = np.array([], dtype=int)

print("Synthetic attacks generated:", synthetic_attacks.shape)

# Show distribution & samples
if synthetic_attacks.shape[0] > 0:
    plt.figure(figsize=(10,4))
    plt.hist(synthetic_attacks.flatten(), bins=50)
    plt.title("GAN Synthetic Data Distribution")
    plt.show()
    print("\nSample GAN Synthetic Data (first 5 rows):")
    print(synthetic_attacks[:5])

# Combine synthetic with original training data for final classifier training
X_combined_train = np.vstack((X_train, synthetic_attacks)) if synthetic_attacks.shape[0] > 0 else X_train
y_combined_train = np.hstack((y_train, y_synthetic)) if synthetic_attacks.shape[0] > 0 else y_train
print("Combined train shape (for classifier):", X_combined_train.shape)

# --------------------------
# 7) PREPARE DATA FOR VAE (BALANCED ATTACK DATA)
#    --- key improvement: train VAE on balanced attack subset (real + synthetic)
# --------------------------
# Build per-class pools (real attack samples + generated samples if any)
from collections import defaultdict
per_class_pool = defaultdict(list)

# add real attack samples
for cls in classes_attack:
    per_class_pool[cls].append(attack_data[attack_labels == cls])

# add generated samples to corresponding pools
if synthetic_attacks.shape[0] > 0:
    for cls in classes_attack:
        # append synthetic samples of this cls if present
        sel = (y_synthetic == cls)
        if np.any(sel):
            per_class_pool[cls].append(synthetic_attacks[sel])

# target per-class for VAE training: choose reasonable cap (we used TARGET_PER_CLASS above)
VAE_TARGET_PER_CLASS = TARGET_PER_CLASS

balanced_attack_pieces = []
balanced_attack_labels = []
for cls in classes_attack:
    pool = np.vstack(per_class_pool[cls]) if len(per_class_pool[cls])>0 else np.empty((0, num_features))
    if pool.shape[0] == 0:
        continue
    # If pool smaller than target, sample with replacement; if larger, sample without replacement
    if pool.shape[0] >= VAE_TARGET_PER_CLASS:
        idxs = np.random.choice(pool.shape[0], VAE_TARGET_PER_CLASS, replace=False)
    else:
        idxs = np.random.choice(pool.shape[0], VAE_TARGET_PER_CLASS, replace=True)
    balanced_attack_pieces.append(pool[idxs])
    balanced_attack_labels.append(np.full(VAE_TARGET_PER_CLASS, cls, dtype=int))

# final VAE training data (balanced across attack classes)
if len(balanced_attack_pieces) == 0:
    raise RuntimeError("No attack data available for VAE training.")
X_vae_train = np.vstack(balanced_attack_pieces)
y_vae_train = np.hstack(balanced_attack_labels)
print("VAE training (balanced attacks) shape:", X_vae_train.shape)

# --------------------------
# 8) BUILD β-VAE (encoder outputs z_mean & z_log_var)
# --------------------------
latent_dim_vae = 32 
inputs = layers.Input(shape=(num_features,), name="vae_input")
h = layers.Dense(128, activation="relu")(inputs)
z_mean = layers.Dense(latent_dim_vae, name="z_mean")(h)
z_log_var = layers.Dense(latent_dim_vae, name="z_log_var")(h)

def sampling_fn(zmean, zlog):
    eps = tf.random.normal(shape=tf.shape(zmean))
    return zmean + tf.exp(0.5 * zlog) * eps

# encoder returns z_mean and z_log_var (we will sample in train_step)
encoder = Model(inputs, [z_mean, z_log_var], name="encoder")

latent_inputs = layers.Input(shape=(latent_dim_vae,), name="z_input")
d = layers.Dense(128, activation="relu")(latent_inputs)
outputs = layers.Dense(num_features, activation="sigmoid")(d)
decoder = Model(latent_inputs, outputs, name="decoder")

# --------------------------
# 9) Custom VAE model with β (train_step uses both recon loss and KL)
# --------------------------
class BetaVAE(tf.keras.Model):
    def __init__(self, encoder, decoder, beta=0.1, **kwargs):
        super(BetaVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta

    def compile(self, optimizer):
        super(BetaVAE, self).compile()
        self.optimizer = optimizer

    @tf.function  # speed-up the training on GPU
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean_t, z_log_var_t = self.encoder(data, training=True)
            z = sampling_fn(z_mean_t, z_log_var_t)
            reconstruction = self.decoder(z, training=True)
            # reconstruction loss (MSE)
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(data - reconstruction), axis=1))
            # KL divergence
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var_t - tf.square(z_mean_t) - tf.exp(z_log_var_t), axis=1))
            loss = recon_loss + self.beta * kl_loss
        vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        grads = tape.gradient(loss, vars)
        self.optimizer.apply_gradients(zip(grads, vars))
        return {"loss": loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

    @tf.function
    def call(self, inputs, training=False):
        z_mean_t, z_log_var_t = self.encoder(inputs, training=training)
        z = sampling_fn(z_mean_t, z_log_var_t)
        return self.decoder(z, training=training)

# instantiate and compile
beta = 0.1  # keep KL weight small so reconstruction dominates (preserves class-specific features)
vae = BetaVAE(encoder, decoder, beta=beta)
vae.compile(optimizer=tf.keras.optimizers.Adam())

# Training callbacks (lightweight)
LOG_CSV = "vae_training_log.csv"
class CSVLoggerLite(tf.keras.callbacks.Callback):
    def __init__(self, path):
        super().__init__()
        self.path = path
        with open(self.path, "w") as f:
            f.write("epoch,loss,recon_loss,kl_loss\n")
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        row = [epoch, logs.get("loss", np.nan), logs.get("recon_loss", np.nan), logs.get("kl_loss", np.nan)]
        with open(self.path, "a") as f:
            f.write(",".join(str(x) for x in row) + "\n")

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="loss", patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.3, patience=2, min_lr=1e-6),
    tf.keras.callbacks.TerminateOnNaN(),
    CSVLoggerLite(LOG_CSV),
]

# Train VAE on the balanced attack dataset (this preserves minority-class patterns)
start_vae = time.time()
history = vae.fit(X_vae_train, epochs=50, batch_size=256, shuffle=True, callbacks=callbacks, verbose=1)
vae_time = time.time() - start_vae
print(f"VAE Training Finished in {vae_time:.1f}s")

# --------------------------
# 10) EXTRACT LATENT VECTORS FOR CLASSIFIER (use z_mean as representation)
# --------------------------
# We use encoder.predict to get [z_mean, z_log_var]; take z_mean
z_mean_train, _ = encoder.predict(X_train, verbose=0)
z_mean_test, _ = encoder.predict(X_test, verbose=0)

# If you want to use combined training (original train + synthetic) for classifier:
# compute z_mean for combined train:
if X_combined_train.shape[0] > X_train.shape[0]:
    z_mean_combined, _ = encoder.predict(X_combined_train, verbose=0)
else:
    z_mean_combined = z_mean_train

print("Latent Features Shape:", z_mean_train.shape)

# --------------------------
# 11) LIGHTGBM (class weights from combined training labels)
# --------------------------
classes_all = np.unique(y_combined_train)
weights = compute_class_weight('balanced', classes=classes_all, y=y_combined_train)
weight_dict = dict(zip(classes_all, weights))
sample_weights = np.array([weight_dict[c] for c in y_combined_train], dtype=float)

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

train_data = lgb.Dataset(z_mean_combined, label=y_combined_train, weight=sample_weights)
test_data = lgb.Dataset(z_mean_test, label=y_test)
start_lgb = time.time()
callbacks = [lgb.log_evaluation(period=10)]
model_lgb = lgb.train(
    params,
    train_data,
    valid_sets=[test_data],
    num_boost_round=200,
    callbacks=callbacks
)
lgb_time = time.time() - start_lgb
print(f"LightGBM Training Finished in {lgb_time:.1f}s")

# --------------------------
# 12) EVALUATION (same style)
# --------------------------
print("\nEvaluating LightGBM on VAE latent features...")

y_pred_probs = model_lgb.predict(z_mean_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true = y_test

acc  = accuracy_score(y_true, y_pred_classes)
prec = precision_score(y_true, y_pred_classes, average="weighted", zero_division=0)
rec  = recall_score(y_true, y_pred_classes, average="weighted", zero_division=0)
f1   = f1_score(y_true, y_pred_classes, average="weighted", zero_division=0)

print("\n=== 📊 Evaluation Metrics (Overall) ===")
print(f"Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision : {prec:.4f} ({prec*100:.2f}%)")
print(f"Recall    : {rec:.4f} ({rec*100:.2f}%)")
print(f"F1-Score  : {f1:.4f} ({f1*100:.2f}%)")

print("\n=== Classification report ===")
print(classification_report(y_true, y_pred_classes, target_names=classes, digits=4))

cm = confusion_matrix(y_true, y_pred_classes, labels=range(len(classes)))
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes, cbar_kws={'label':'Count'})
plt.title("Confusion Matrix — LightGBM on β-VAE Features")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# --------------------------
# 13) TRAINING TIME SUMMARY
# --------------------------
print("\n⏱ Training Times Summary:")
print(f"GAN Training Time  : {gan_time:.1f}s")
print(f"VAE Training Time  : {vae_time:.1f}s")
print(f"LightGBM Training  : {lgb_time:.1f}s")
