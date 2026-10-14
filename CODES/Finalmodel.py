# =======================================================================
# FULL MODEL: GAN + β-VAE + LIGHTGBM
# CICIDS2017 → PREPROCESSING → GAN → β-VAE → LIGHTGBM
# =======================================================================

import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import layers, Model
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

latent_dim_gan = 32
num_features = X_train.shape[1]

def build_generator():
    inputs = layers.Input(shape=(latent_dim_gan,))
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dense(256, activation="relu")(x)

    # MinMax data is in [0,1], so sigmoid is used
    outputs = layers.Dense(num_features, activation="sigmoid")(x)

    return Model(inputs, outputs, name="generator")


def build_discriminator():
    inputs = layers.Input(shape=(num_features,))
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return Model(inputs, outputs, name="discriminator")


generator = build_generator()
discriminator = build_discriminator()

discriminator.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
    loss="binary_crossentropy"
)

discriminator.trainable = False

z = layers.Input(shape=(latent_dim_gan,))
fake_sample = generator(z)
validity = discriminator(fake_sample)

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

    idx = np.random.randint(0, attack_data.shape[0], batch_size)
    real = attack_data[idx]

    noise = np.random.normal(0, 1, (batch_size, latent_dim_gan))
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

    d_loss = (d_loss_real + d_loss_fake) / 2.0

    discriminator.trainable = False

    noise = np.random.normal(0, 1, (batch_size, latent_dim_gan))

    g_loss = gan.train_on_batch(
        noise,
        np.ones((batch_size, 1))
    )

    d_losses.append(float(d_loss))
    g_losses.append(float(g_loss))

    if epoch % 200 == 0:
        print(f"Epoch {epoch} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}")

gan_time = time.time() - start_gan
print(f"GAN Training Finished in {gan_time:.1f}s")

# =======================================================================
# 6. GENERATE SYNTHETIC ATTACK SAMPLES
# =======================================================================

classes_attack = np.unique(attack_labels)

counts = {
    c: np.sum(attack_labels == c)
    for c in classes_attack
}

# Controlled augmentation target
TARGET_PER_CLASS = 30000

synthetic_list = []
y_synthetic_list = []

for cls in classes_attack:

    cls_count = counts[cls]
    need = max(0, TARGET_PER_CLASS - cls_count)

    if need == 0:
        continue

    noise = np.random.normal(0, 1, (need, latent_dim_gan))
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

print("\nGAN Synthetic Data Quality Check")
print("Real attack min :", attack_data.min())
print("Real attack max :", attack_data.max())
print("Synthetic min   :", synthetic_attacks.min())
print("Synthetic max   :", synthetic_attacks.max())
print("Synthetic mean  :", synthetic_attacks.mean())
print("Synthetic std   :", synthetic_attacks.std())

plt.figure(figsize=(10, 4))
plt.hist(synthetic_attacks.flatten(), bins=50)
plt.title("GAN Synthetic Data Distribution")
plt.xlabel("Feature Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# =======================================================================
# 7. COMBINE ORIGINAL TRAINING DATA WITH GAN SYNTHETIC DATA
# =======================================================================

X_combined_train = np.vstack((X_train, synthetic_attacks))
y_combined_train = np.hstack((y_train, y_synthetic))

print("Combined Train Shape:", X_combined_train.shape)

# =======================================================================
# 8. PREPARE BALANCED ATTACK DATASET FOR β-VAE TRAINING
# =======================================================================

from collections import defaultdict

per_class_pool = defaultdict(list)

for cls in classes_attack:
    per_class_pool[cls].append(attack_data[attack_labels == cls])

if synthetic_attacks.shape[0] > 0:
    for cls in classes_attack:
        selected = y_synthetic == cls
        if np.any(selected):
            per_class_pool[cls].append(synthetic_attacks[selected])

VAE_TARGET_PER_CLASS = TARGET_PER_CLASS

balanced_attack_pieces = []
balanced_attack_labels = []

for cls in classes_attack:

    pool = np.vstack(per_class_pool[cls])

    if pool.shape[0] >= VAE_TARGET_PER_CLASS:
        idxs = np.random.choice(
            pool.shape[0],
            VAE_TARGET_PER_CLASS,
            replace=False
        )
    else:
        idxs = np.random.choice(
            pool.shape[0],
            VAE_TARGET_PER_CLASS,
            replace=True
        )

    balanced_attack_pieces.append(pool[idxs])
    balanced_attack_labels.append(
        np.full(VAE_TARGET_PER_CLASS, cls, dtype=int)
    )

X_vae_train = np.vstack(balanced_attack_pieces)
y_vae_train = np.hstack(balanced_attack_labels)

print("β-VAE Training Data Shape:", X_vae_train.shape)

# =======================================================================
# 9. BUILD β-VAE
# =======================================================================

latent_dim_vae = 32
beta = 0.1

inputs = layers.Input(shape=(num_features,), name="vae_input")

h = layers.Dense(128, activation="relu")(inputs)

z_mean = layers.Dense(latent_dim_vae, name="z_mean")(h)
z_log_var = layers.Dense(latent_dim_vae, name="z_log_var")(h)

def sampling_fn(zmean, zlog):
    eps = tf.random.normal(shape=tf.shape(zmean))
    return zmean + tf.exp(0.5 * zlog) * eps

encoder = Model(inputs, [z_mean, z_log_var], name="encoder")

latent_inputs = layers.Input(shape=(latent_dim_vae,), name="z_input")
d = layers.Dense(128, activation="relu")(latent_inputs)
outputs = layers.Dense(num_features, activation="sigmoid")(d)

decoder = Model(latent_inputs, outputs, name="decoder")

class BetaVAE(tf.keras.Model):
    def __init__(self, encoder, decoder, beta=0.1, **kwargs):
        super(BetaVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta

    def compile(self, optimizer):
        super(BetaVAE, self).compile()
        self.optimizer = optimizer

    @tf.function
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean_t, z_log_var_t = self.encoder(data, training=True)
            z = sampling_fn(z_mean_t, z_log_var_t)
            reconstruction = self.decoder(z, training=True)

            recon_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(data - reconstruction), axis=1)
            )

            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var_t
                    - tf.square(z_mean_t)
                    - tf.exp(z_log_var_t),
                    axis=1
                )
            )

            total_loss = recon_loss + self.beta * kl_loss

        train_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        grads = tape.gradient(total_loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss
        }

vae = BetaVAE(encoder, decoder, beta=beta)
vae.compile(optimizer=tf.keras.optimizers.Adam())

print("β-VAE Built Successfully")

# =======================================================================
# 10. TRAIN β-VAE
# =======================================================================

LOG_CSV = "vae_training_log_full_model.csv"

class CSVLoggerLite(tf.keras.callbacks.Callback):
    def __init__(self, path):
        super().__init__()
        self.path = path
        with open(self.path, "w") as f:
            f.write("epoch,loss,recon_loss,kl_loss\n")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        row = [
            epoch,
            logs.get("loss", np.nan),
            logs.get("recon_loss", np.nan),
            logs.get("kl_loss", np.nan)
        ]
        with open(self.path, "a") as f:
            f.write(",".join(str(x) for x in row) + "\n")

callbacks_vae = [
    tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=4,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss",
        factor=0.3,
        patience=2,
        min_lr=1e-6
    ),
    tf.keras.callbacks.TerminateOnNaN(),
    CSVLoggerLite(LOG_CSV)
]

start_vae = time.time()

history = vae.fit(
    X_vae_train,
    epochs=50,
    batch_size=256,
    shuffle=True,
    callbacks=callbacks_vae,
    verbose=1
)

vae_time = time.time() - start_vae
print(f"β-VAE Training Finished in {vae_time:.1f}s")

# =======================================================================
# 11. EXTRACT LATENT FEATURES USING z_mean
# =======================================================================

z_mean_train, _ = encoder.predict(X_train, verbose=0)
z_mean_test, _ = encoder.predict(X_test, verbose=0)
z_mean_combined, _ = encoder.predict(X_combined_train, verbose=0)

print("Latent Train Shape:", z_mean_train.shape)
print("Latent Combined Shape:", z_mean_combined.shape)
print("Latent Test Shape:", z_mean_test.shape)

# =======================================================================
# 12. LIGHTGBM ON β-VAE LATENT FEATURES
# =======================================================================

classes_all = np.unique(y_combined_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes_all,
    y=y_combined_train
)

weight_dict = dict(zip(classes_all, weights))

sample_weights = np.array(
    [weight_dict[label] for label in y_combined_train],
    dtype=float
)

params = {
    "objective": "multiclass",
    "num_class": len(classes_all),
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 8,
    "min_data_in_leaf": 100,
    "lambda_l1": 0.1,
    "lambda_l2": 0.5,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "metric": "multi_logloss",
    "force_col_wise": True
}

train_data = lgb.Dataset(
    z_mean_combined,
    label=y_combined_train,
    weight=sample_weights
)

test_data = lgb.Dataset(
    z_mean_test,
    label=y_test
)

start_lgb = time.time()

callbacks_lgb = [
    lgb.log_evaluation(period=10)
]

model_lgb = lgb.train(
    params,
    train_data,
    valid_sets=[test_data],
    num_boost_round=300,
    callbacks=callbacks_lgb
)

lgb_time = time.time() - start_lgb

print(f"LightGBM Training Finished in {lgb_time:.1f}s")

# =======================================================================
# 13. EVALUATION
# =======================================================================

print("\nEvaluating GAN + β-VAE + LightGBM...")

y_pred_probs = model_lgb.predict(z_mean_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_test, y_pred_classes)
prec = precision_score(y_test, y_pred_classes, average="weighted", zero_division=0)
rec = recall_score(y_test, y_pred_classes, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred_classes, average="weighted", zero_division=0)
macro_f1 = f1_score(y_test, y_pred_classes, average="macro", zero_division=0)

print("\n=== Evaluation Metrics: GAN + β-VAE + LightGBM ===")
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
# 14. MINORITY-CLASS F1
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
# 15. CONFUSION MATRIX
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

plt.title("Confusion Matrix — GAN + β-VAE + LightGBM")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# =======================================================================
# 16. TRAINING TIME SUMMARY
# =======================================================================

print("\nTraining Times Summary")
print(f"GAN Training Time      : {gan_time:.1f}s")
print(f"β-VAE Training Time    : {vae_time:.1f}s")
print(f"LightGBM Training Time : {lgb_time:.1f}s")
print(f"Total Training Time    : {gan_time + vae_time + lgb_time:.1f}s")

ablation_result = {
    "Configuration": "C5: GAN + β-VAE + LightGBM",
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "Weighted F1": f1,
    "Macro F1": macro_f1,
    "Minority F1": minority_f1,
    "GAN Training Time": gan_time,
    "β-VAE Training Time": vae_time,
    "LightGBM Training Time": lgb_time,
    "Total Training Time": gan_time + vae_time + lgb_time
}

print("\n=== Full Model Result ===")
print(ablation_result)