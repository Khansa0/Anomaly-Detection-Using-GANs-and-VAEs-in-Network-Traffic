# =======================================================================
# IDS INTRUSION CSV → GAN + β-VAE + LightGBM
# =======================================================================

import os, time
import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import layers, Model
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict

# =======================================================================
# 1. DATASET PATH
# =======================================================================

DATASET_PATH = r"D:\Khanssa\CSE-CIC-IDS2018"   # CHANGE THIS

csv_files = [f for f in os.listdir(DATASET_PATH) if f.endswith(".csv")]

df_list = []
for file in csv_files:
    path = os.path.join(DATASET_PATH, file)
    print("Loading:", path)
    df_list.append(pd.read_csv(path, low_memory=False))

df = pd.concat(df_list, axis=0, ignore_index=True)
print("Dataset Loaded:", df.shape)

# =======================================================================
# 2. CLEANING
# =======================================================================

df.columns = df.columns.str.strip()
df.drop_duplicates(inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

print("After Cleaning:", df.shape)

# =======================================================================
# 3. AUTO-DETECT LABEL COLUMN
# =======================================================================

possible_labels = ["Label", "label", "Class", "class", "Attack", "attack", "target", "Target", "attack_cat"]

label_col = None
for col in possible_labels:
    if col in df.columns:
        label_col = col
        break

if label_col is None:
    raise ValueError(f"No label column found. Available columns are:\n{df.columns.tolist()}")

print("Detected Label Column:", label_col)

df[label_col] = df[label_col].astype(str).str.strip()

# =======================================================================
# 4. DROP NON-USEFUL COLUMNS IF PRESENT
# =======================================================================

drop_cols = [
    label_col,
    "Flow ID", "Src IP", "Source IP", "Dst IP", "Destination IP",
    "Timestamp", "Flow Bytes/s", "Flow Packets/s",
    "Unnamed: 0", "id"
]

X_df = df.drop(columns=drop_cols, errors="ignore")
y_raw = df[label_col]

# Encode categorical features automatically
cat_cols = X_df.select_dtypes(include=["object"]).columns.tolist()

for col in cat_cols:
    le = LabelEncoder()
    X_df[col] = le.fit_transform(X_df[col].astype(str))

X_df = X_df.apply(pd.to_numeric, errors="coerce")
X_df.replace([np.inf, -np.inf], np.nan, inplace=True)
X_df.fillna(0, inplace=True)

# =======================================================================
# 5. LABEL ENCODING
# =======================================================================

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
classes = label_encoder.classes_

print("\nClasses:")
for i, cls in enumerate(classes):
    print(i, cls)

# =======================================================================
# 6. FEATURE SCALING
# =======================================================================

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_df).astype("float32")

print("Feature Shape:", X_scaled.shape)

# =======================================================================
# 7. TRAIN / TEST SPLIT
# =======================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Train Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# =======================================================================
# 8. IDENTIFY NORMAL/BENIGN CLASS
# =======================================================================

normal_keywords = ["BENIGN", "Benign", "benign", "Normal", "normal", "NORMAL"]

normal_label = None
for n in normal_keywords:
    if n in classes:
        normal_label = label_encoder.transform([n])[0]
        break

if normal_label is None:
    normal_label = 0
    print("Warning: Normal/Benign class not detected. Using class 0 as normal.")

attack_idx = y_train != normal_label
attack_data = X_train[attack_idx]
attack_labels = y_train[attack_idx]

print("Attack Data Shape:", attack_data.shape)

# =======================================================================
# 9. GAN MODEL
# =======================================================================

latent_dim_gan = 32
num_features = X_train.shape[1]

def build_generator():
    inp = layers.Input(shape=(latent_dim_gan,))
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.Dense(256, activation="relu")(x)
    out = layers.Dense(num_features, activation="sigmoid")(x)
    return Model(inp, out)

def build_discriminator():
    inp = layers.Input(shape=(num_features,))
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return Model(inp, out)

generator = build_generator()
discriminator = build_discriminator()

bce_loss = tf.keras.losses.BinaryCrossentropy()
d_optimizer = tf.keras.optimizers.Adam(0.0002)
g_optimizer = tf.keras.optimizers.Adam(0.0002)

@tf.function
def train_gan_step(real_batch):
    batch_size_tf = tf.shape(real_batch)[0]
    noise = tf.random.normal((batch_size_tf, latent_dim_gan))

    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        fake_samples = generator(noise, training=True)

        real_output = discriminator(real_batch, training=True)
        fake_output = discriminator(fake_samples, training=True)

        d_loss_real = bce_loss(tf.ones_like(real_output), real_output)
        d_loss_fake = bce_loss(tf.zeros_like(fake_output), fake_output)
        d_loss = d_loss_real + d_loss_fake

        g_loss = bce_loss(tf.ones_like(fake_output), fake_output)

    d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)

    d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
    g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))

    return d_loss, g_loss

# =======================================================================
# 10. TRAIN GAN
# =======================================================================

gan_epochs = 1500
batch_size = 64

d_losses, g_losses = [], []

start_gan = time.time()

for epoch in range(gan_epochs):
    idx = np.random.randint(0, attack_data.shape[0], batch_size)
    real_batch = attack_data[idx].astype("float32")

    d_loss, g_loss = train_gan_step(real_batch)

    d_losses.append(float(d_loss.numpy()))
    g_losses.append(float(g_loss.numpy()))

    if epoch % 200 == 0:
        print(f"Epoch {epoch} | D Loss: {float(d_loss.numpy()):.4f} | G Loss: {float(g_loss.numpy()):.4f}")

gan_time = time.time() - start_gan
print(f"GAN Training Finished in {gan_time:.1f}s")

# =======================================================================
# 11. SYNTHETIC DATA GENERATION
# =======================================================================

classes_attack = np.unique(attack_labels)

TARGET_PER_CLASS = min(30000, max(np.bincount(attack_labels)))

synthetic_list = []
y_synthetic_list = []

for cls in classes_attack:
    cls_count = np.sum(attack_labels == cls)
    need = max(0, TARGET_PER_CLASS - cls_count)

    if need == 0:
        continue

    noise = np.random.normal(0, 1, (need, latent_dim_gan)).astype("float32")
    gen_samples = generator.predict(noise, verbose=0)

    synthetic_list.append(gen_samples)
    y_synthetic_list.append(np.full(need, cls, dtype=int))

if len(synthetic_list) > 0:
    synthetic_attacks = np.vstack(synthetic_list).astype("float32")
    y_synthetic = np.hstack(y_synthetic_list)
else:
    synthetic_attacks = np.empty((0, num_features), dtype="float32")
    y_synthetic = np.array([], dtype=int)

print("Synthetic Attacks Generated:", synthetic_attacks.shape)

X_combined_train = np.vstack((X_train, synthetic_attacks)).astype("float32")
y_combined_train = np.hstack((y_train, y_synthetic))

print("Combined Training Shape:", X_combined_train.shape)

# =======================================================================
# 12. BALANCED ATTACK DATA FOR β-VAE
# =======================================================================

per_class_pool = defaultdict(list)

for cls in classes_attack:
    per_class_pool[cls].append(attack_data[attack_labels == cls])

for cls in classes_attack:
    selected = y_synthetic == cls
    if np.any(selected):
        per_class_pool[cls].append(synthetic_attacks[selected])

balanced_attack_pieces = []

for cls in classes_attack:
    pool = np.vstack(per_class_pool[cls])

    if pool.shape[0] >= TARGET_PER_CLASS:
        idxs = np.random.choice(pool.shape[0], TARGET_PER_CLASS, replace=False)
    else:
        idxs = np.random.choice(pool.shape[0], TARGET_PER_CLASS, replace=True)

    balanced_attack_pieces.append(pool[idxs])

X_vae_train = np.vstack(balanced_attack_pieces).astype("float32")

print("β-VAE Training Data Shape:", X_vae_train.shape)

# =======================================================================
# 13. β-VAE MODEL
# =======================================================================

latent_dim_vae = 32
beta = 0.1

vae_inputs = layers.Input(shape=(num_features,))
h = layers.Dense(128, activation="relu")(vae_inputs)

z_mean = layers.Dense(latent_dim_vae, name="z_mean")(h)
z_log_var = layers.Dense(latent_dim_vae, name="z_log_var")(h)

def sampling_fn(z_mean_input, z_log_var_input):
    eps = tf.random.normal(shape=tf.shape(z_mean_input))
    return z_mean_input + tf.exp(0.5 * z_log_var_input) * eps

encoder = Model(vae_inputs, [z_mean, z_log_var])

latent_inputs = layers.Input(shape=(latent_dim_vae,))
d = layers.Dense(128, activation="relu")(latent_inputs)
vae_outputs = layers.Dense(num_features, activation="sigmoid")(d)

decoder = Model(latent_inputs, vae_outputs)

class BetaVAE(tf.keras.Model):
    def __init__(self, encoder, decoder, beta=0.1):
        super(BetaVAE, self).__init__()
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

            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(data - reconstruction), axis=1))

            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var_t - tf.square(z_mean_t) - tf.exp(z_log_var_t),
                    axis=1
                )
            )

            total_loss = recon_loss + self.beta * kl_loss

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        grads = tape.gradient(total_loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))

        return {"loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

vae = BetaVAE(encoder, decoder, beta=beta)
vae.compile(optimizer=tf.keras.optimizers.Adam())

# =======================================================================
# 14. TRAIN β-VAE
# =======================================================================

start_vae = time.time()

vae.fit(
    X_vae_train,
    epochs=50,
    batch_size=256,
    shuffle=True,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor="loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.3, patience=2, min_lr=1e-6),
        tf.keras.callbacks.TerminateOnNaN()
    ],
    verbose=1
)

vae_time = time.time() - start_vae
print(f"β-VAE Training Finished in {vae_time:.1f}s")

# =======================================================================
# 15. LATENT FEATURE EXTRACTION
# =======================================================================

z_mean_test, _ = encoder.predict(X_test, verbose=0)
z_mean_combined, _ = encoder.predict(X_combined_train, verbose=0)

print("Latent Combined Shape:", z_mean_combined.shape)
print("Latent Test Shape:", z_mean_test.shape)

# =======================================================================
# 16. LIGHTGBM CLASSIFIER
# =======================================================================

classes_all = np.unique(y_combined_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes_all,
    y=y_combined_train
)

weight_dict = dict(zip(classes_all, weights))
sample_weights = np.array([weight_dict[label] for label in y_combined_train], dtype=float)

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

train_data = lgb.Dataset(z_mean_combined, label=y_combined_train, weight=sample_weights)
test_data = lgb.Dataset(z_mean_test, label=y_test)

start_lgb = time.time()

model_lgb = lgb.train(
    params,
    train_data,
    valid_sets=[test_data],
    num_boost_round=300,
    callbacks=[lgb.log_evaluation(period=10)]
)

lgb_time = time.time() - start_lgb

print(f"LightGBM Training Finished in {lgb_time:.1f}s")

# =======================================================================
# 17. EVALUATION
# =======================================================================

print("\nEvaluating IDS Intrusion CSV GAN + β-VAE + LightGBM...")

y_pred_probs = model_lgb.predict(z_mean_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_test, y_pred_classes)
prec = precision_score(y_test, y_pred_classes, average="weighted", zero_division=0)
rec = recall_score(y_test, y_pred_classes, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred_classes, average="weighted", zero_division=0)
macro_f1 = f1_score(y_test, y_pred_classes, average="macro", zero_division=0)

print("\n=== Evaluation Metrics ===")
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

# Minority-F1: all classes except benign/normal
minority_classes = [
    cls for cls in classes
    if cls.lower() not in ["benign", "normal"]
]

report = classification_report(
    y_test,
    y_pred_classes,
    target_names=classes,
    output_dict=True,
    zero_division=0
)

minority_f1_scores = [
    report[cls]["f1-score"]
    for cls in minority_classes
    if cls in report
]

minority_f1 = np.mean(minority_f1_scores)

print(f"\nMinority-Class F1 : {minority_f1:.4f} ({minority_f1*100:.2f}%)")

# =======================================================================
# 18. CONFUSION MATRIX
# =======================================================================

cm = confusion_matrix(y_test, y_pred_classes, labels=range(len(classes)))

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes
)

plt.title("Confusion Matrix — IDS Intrusion CSV GAN + β-VAE + LightGBM")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# =======================================================================
# 19. RESULT SUMMARY
# =======================================================================

print("\nTraining Times Summary")
print(f"GAN Training Time      : {gan_time:.1f}s")
print(f"β-VAE Training Time    : {vae_time:.1f}s")
print(f"LightGBM Training Time : {lgb_time:.1f}s")
print(f"Total Training Time    : {gan_time + vae_time + lgb_time:.1f}s")

result = {
    "Dataset": "IDS Intrusion CSV",
    "Configuration": "GAN + β-VAE + LightGBM",
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "Weighted F1": f1,
    "Macro F1": macro_f1,
    "Minority F1": minority_f1,
    "Total Training Time": gan_time + vae_time + lgb_time
}

print("\n=== Final Result ===")
print(result)