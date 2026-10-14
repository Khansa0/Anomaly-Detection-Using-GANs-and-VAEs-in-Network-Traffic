# =======================================================================
# ABLATION C4: β-VAE + LIGHTGBM
# CICIDS2017 → PREPROCESSING → β-VAE LATENT FEATURES → LIGHTGBM
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
# 4. BUILD β-VAE
# =======================================================================

num_features = X_train.shape[1]
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
                    1 + z_log_var_t - tf.square(z_mean_t) - tf.exp(z_log_var_t),
                    axis=1
                )
            )

            loss = recon_loss + self.beta * kl_loss

        vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        grads = tape.gradient(loss, vars)

        self.optimizer.apply_gradients(zip(grads, vars))

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss
        }

    @tf.function
    def call(self, inputs, training=False):
        z_mean_t, z_log_var_t = self.encoder(inputs, training=training)
        z = sampling_fn(z_mean_t, z_log_var_t)
        return self.decoder(z, training=training)

vae = BetaVAE(encoder, decoder, beta=beta)
vae.compile(optimizer=tf.keras.optimizers.Adam())

print("β-VAE Built Successfully")

# =======================================================================
# 5. TRAIN β-VAE ON ORIGINAL TRAINING DATA
# =======================================================================

LOG_CSV = "vae_ablation_no_gan_training_log.csv"

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

callbacks = [
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
    X_train,
    epochs=50,
    batch_size=256,
    shuffle=True,
    callbacks=callbacks,
    verbose=1
)

vae_time = time.time() - start_vae

print(f"β-VAE Training Finished in {vae_time:.1f}s")

# =======================================================================
# 6. EXTRACT LATENT FEATURES USING z_mean
# =======================================================================

z_mean_train, _ = encoder.predict(X_train, verbose=0)
z_mean_test, _ = encoder.predict(X_test, verbose=0)

print("Latent Features Shape:", z_mean_train.shape)

# =======================================================================
# 7. LIGHTGBM ON β-VAE LATENT FEATURES
# =======================================================================

classes_all = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes_all,
    y=y_train
)

weight_dict = dict(zip(classes_all, weights))

sample_weights = np.array(
    [weight_dict[label] for label in y_train],
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
    z_mean_train,
    label=y_train,
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

model_lgb_bvae = lgb.train(
    params,
    train_data,
    valid_sets=[test_data],
    num_boost_round=200,
    callbacks=callbacks_lgb
)

lgb_time = time.time() - start_lgb

print(f"β-VAE + LightGBM Training Finished in {lgb_time:.1f}s")

# =======================================================================
# 8. EVALUATION — β-VAE + LIGHTGBM
# =======================================================================

print("\nEvaluating β-VAE + LightGBM...")

y_pred_probs = model_lgb_bvae.predict(z_mean_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_test, y_pred_classes)
prec = precision_score(y_test, y_pred_classes, average="weighted", zero_division=0)
rec = recall_score(y_test, y_pred_classes, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred_classes, average="weighted", zero_division=0)
macro_f1 = f1_score(y_test, y_pred_classes, average="macro", zero_division=0)

print("\n=== Evaluation Metrics: β-VAE + LightGBM ===")
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
# 9. MINORITY-CLASS F1
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
# 10. CONFUSION MATRIX
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

plt.title("Confusion Matrix — β-VAE + LightGBM")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# =======================================================================
# 11. TRAINING LOSS CURVES
# =======================================================================

if os.path.exists(LOG_CSV):
    log_df = pd.read_csv(LOG_CSV)

    plt.figure(figsize=(8, 5))
    plt.plot(log_df["epoch"], log_df["loss"], marker="o", label="Total Loss")
    plt.plot(log_df["epoch"], log_df["recon_loss"], marker="s", label="Reconstruction Loss")
    plt.plot(log_df["epoch"], log_df["kl_loss"], marker="^", label="KL Loss")
    plt.title("β-VAE Training Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

# =======================================================================
# 12. SAVE ABLATION RESULT
# =======================================================================

ablation_result = {
    "Configuration": "C4: β-VAE + LightGBM",
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "Weighted F1": f1,
    "Macro F1": macro_f1,
    "Minority F1": minority_f1,
    "β-VAE Training Time": vae_time,
    "LightGBM Training Time": lgb_time,
    "Total Training Time": vae_time + lgb_time
}

print("\n=== Ablation Result ===")
print(ablation_result)