# =======================================================================
# FULL MODEL FOR UNSW-NB15: GAN + β-VAE + LIGHTGBM
# Dataset files:
#   UNSW_NB15_training-set.csv
#   UNSW_NB15_testing-set.csv
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict

# =======================================================================
# 1. DATASET PATH
# =======================================================================

DATASET_PATH = r"D:\Khanssa\UNSW_nb15"

TRAIN_FILE = os.path.join(DATASET_PATH, "UNSW_NB15_training-set.csv")
TEST_FILE = os.path.join(DATASET_PATH, "UNSW_NB15_testing-set.csv")

# =======================================================================
# 2. LOAD DATA
# =======================================================================

train_df = pd.read_csv(TRAIN_FILE, low_memory=False)
test_df = pd.read_csv(TEST_FILE, low_memory=False)

print("Training Dataset Loaded:", train_df.shape)
print("Testing Dataset Loaded :", test_df.shape)

train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

# =======================================================================
# 3. BASIC CLEANING
# =======================================================================

train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df.replace([np.inf, -np.inf], np.nan, inplace=True)

train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

train_df.drop_duplicates(inplace=True)
test_df.drop_duplicates(inplace=True)

print("After Cleaning Train:", train_df.shape)
print("After Cleaning Test :", test_df.shape)

# =======================================================================
# 4. LABEL HANDLING
# =======================================================================

label_col = "attack_cat"

train_df[label_col] = train_df[label_col].astype(str).str.strip()
test_df[label_col] = test_df[label_col].astype(str).str.strip()

label_encoder = LabelEncoder()
label_encoder.fit(
    pd.concat(
        [train_df[label_col], test_df[label_col]],
        axis=0
    )
)

y_train = label_encoder.transform(train_df[label_col])
y_test = label_encoder.transform(test_df[label_col])

classes = label_encoder.classes_

print("\nClasses:")
for i, cls in enumerate(classes):
    print(i, cls)

# =======================================================================
# 5. FEATURE PREPROCESSING
# =======================================================================

drop_cols = ["id", "label", "attack_cat"]

X_train_df = train_df.drop(columns=drop_cols, errors="ignore")
X_test_df = test_df.drop(columns=drop_cols, errors="ignore")

categorical_cols = ["proto", "service", "state"]

for col in categorical_cols:
    if col in X_train_df.columns:
        le = LabelEncoder()
        combined_col = pd.concat(
            [X_train_df[col], X_test_df[col]],
            axis=0
        ).astype(str)

        le.fit(combined_col)

        X_train_df[col] = le.transform(X_train_df[col].astype(str))
        X_test_df[col] = le.transform(X_test_df[col].astype(str))

# Ensure all features are numeric
X_train_df = X_train_df.apply(pd.to_numeric, errors="coerce")
X_test_df = X_test_df.apply(pd.to_numeric, errors="coerce")

X_train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test_df.replace([np.inf, -np.inf], np.nan, inplace=True)

X_train_df.fillna(0, inplace=True)
X_test_df.fillna(0, inplace=True)

# Align columns
X_test_df = X_test_df[X_train_df.columns]

# Scale features
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train_df).astype("float32")
X_test = scaler.transform(X_test_df).astype("float32")

num_features = X_train.shape[1]

print("\nFeature Shape Train:", X_train.shape)
print("Feature Shape Test :", X_test.shape)

# =======================================================================
# 6. IDENTIFY NORMAL AND ATTACK SAMPLES
# =======================================================================

normal_candidates = ["Normal", "normal", "NORMAL"]

normal_label = None

for n in normal_candidates:
    if n in classes:
        normal_label = label_encoder.transform([n])[0]
        break

if normal_label is None:
    raise ValueError("Normal class not found in attack_cat labels.")

attack_idx = y_train != normal_label
attack_data = X_train[attack_idx]
attack_labels = y_train[attack_idx]

print("\nNormal Label:", normal_label)
print("Attack Data Shape:", attack_data.shape)

# =======================================================================
# 7. BUILD GAN
# =======================================================================

latent_dim_gan = 32

def build_generator():
    inp = layers.Input(shape=(latent_dim_gan,))
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.Dense(256, activation="relu")(x)
    out = layers.Dense(num_features, activation="sigmoid")(x)
    return Model(inp, out, name="generator")


def build_discriminator():
    inp = layers.Input(shape=(num_features,))
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return Model(inp, out, name="discriminator")


generator = build_generator()
discriminator = build_discriminator()

discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

bce_loss = tf.keras.losses.BinaryCrossentropy()

print("\nGAN Built Successfully")

# =======================================================================
# 8. CUSTOM GAN TRAINING STEP
# =======================================================================

@tf.function
def train_gan_step(real_batch):
    batch_size_tf = tf.shape(real_batch)[0]

    noise = tf.random.normal((batch_size_tf, latent_dim_gan))

    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        gen_fake = generator(noise, training=True)

        real_output = discriminator(real_batch, training=True)
        fake_output = discriminator(gen_fake, training=True)

        d_loss_real = bce_loss(tf.ones_like(real_output), real_output)
        d_loss_fake = bce_loss(tf.zeros_like(fake_output), fake_output)
        d_loss = d_loss_real + d_loss_fake

        g_loss = bce_loss(tf.ones_like(fake_output), fake_output)

    d_grads = d_tape.gradient(
        d_loss,
        discriminator.trainable_variables
    )

    g_grads = g_tape.gradient(
        g_loss,
        generator.trainable_variables
    )

    discriminator_optimizer.apply_gradients(
        zip(d_grads, discriminator.trainable_variables)
    )

    generator_optimizer.apply_gradients(
        zip(g_grads, generator.trainable_variables)
    )

    return d_loss, g_loss

# =======================================================================
# 9. TRAIN GAN
# =======================================================================

gan_epochs = 1500
batch_size = 64

d_losses = []
g_losses = []

start_gan = time.time()

for epoch in range(gan_epochs):
    idx = np.random.randint(
        0,
        attack_data.shape[0],
        batch_size
    )

    real_batch = attack_data[idx].astype("float32")

    d_loss, g_loss = train_gan_step(real_batch)

    d_losses.append(float(d_loss.numpy()))
    g_losses.append(float(g_loss.numpy()))

    if epoch % 200 == 0:
        print(
            f"Epoch {epoch} | "
            f"D Loss: {float(d_loss.numpy()):.4f} | "
            f"G Loss: {float(g_loss.numpy()):.4f}"
        )

gan_time = time.time() - start_gan
print(f"\nGAN Training Finished in {gan_time:.1f}s")

# =======================================================================
# 10. GENERATE SYNTHETIC ATTACK SAMPLES
# =======================================================================

classes_attack = np.unique(attack_labels)

TARGET_PER_CLASS = 30000

synthetic_list = []
y_synthetic_list = []

for cls in classes_attack:
    cls_count = np.sum(attack_labels == cls)

    need = max(0, TARGET_PER_CLASS - cls_count)

    if need == 0:
        continue

    noise = np.random.normal(
        0,
        1,
        (need, latent_dim_gan)
    ).astype("float32")

    gen_samples = generator.predict(noise, verbose=0)

    synthetic_list.append(gen_samples)
    y_synthetic_list.append(
        np.full(need, cls, dtype=int)
    )

if len(synthetic_list) > 0:
    synthetic_attacks = np.vstack(synthetic_list).astype("float32")
    y_synthetic = np.hstack(y_synthetic_list)
else:
    synthetic_attacks = np.empty((0, num_features), dtype="float32")
    y_synthetic = np.array([], dtype=int)

print("Synthetic Attacks Generated:", synthetic_attacks.shape)

if synthetic_attacks.shape[0] > 0:
    print("\nGAN Quality Check")
    print("Real attack min:", attack_data.min())
    print("Real attack max:", attack_data.max())
    print("Synthetic min:", synthetic_attacks.min())
    print("Synthetic max:", synthetic_attacks.max())
    print("Synthetic mean:", synthetic_attacks.mean())
    print("Synthetic std:", synthetic_attacks.std())

    plt.figure(figsize=(10, 4))
    plt.hist(synthetic_attacks.flatten(), bins=50)
    plt.title("UNSW-NB15 GAN Synthetic Data Distribution")
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# =======================================================================
# 11. COMBINE REAL + SYNTHETIC TRAINING DATA
# =======================================================================

X_combined_train = np.vstack(
    (X_train, synthetic_attacks)
).astype("float32")

y_combined_train = np.hstack(
    (y_train, y_synthetic)
)

print("\nCombined Training Shape:", X_combined_train.shape)

# =======================================================================
# 12. BUILD BALANCED ATTACK DATASET FOR β-VAE
# =======================================================================

per_class_pool = defaultdict(list)

for cls in classes_attack:
    per_class_pool[cls].append(
        attack_data[attack_labels == cls]
    )

for cls in classes_attack:
    selected = y_synthetic == cls

    if np.any(selected):
        per_class_pool[cls].append(
            synthetic_attacks[selected]
        )

VAE_TARGET_PER_CLASS = TARGET_PER_CLASS

balanced_attack_pieces = []

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

X_vae_train = np.vstack(
    balanced_attack_pieces
).astype("float32")

print("β-VAE Training Data Shape:", X_vae_train.shape)

# =======================================================================
# 13. BUILD β-VAE
# =======================================================================

latent_dim_vae = 32
beta = 0.1

vae_inputs = layers.Input(
    shape=(num_features,),
    name="vae_input"
)

h = layers.Dense(
    128,
    activation="relu"
)(vae_inputs)

z_mean = layers.Dense(
    latent_dim_vae,
    name="z_mean"
)(h)

z_log_var = layers.Dense(
    latent_dim_vae,
    name="z_log_var"
)(h)

def sampling_fn(z_mean_input, z_log_var_input):
    eps = tf.random.normal(
        shape=tf.shape(z_mean_input)
    )

    return z_mean_input + tf.exp(
        0.5 * z_log_var_input
    ) * eps

encoder = Model(
    vae_inputs,
    [z_mean, z_log_var],
    name="encoder"
)

latent_inputs = layers.Input(
    shape=(latent_dim_vae,)
)

d = layers.Dense(
    128,
    activation="relu"
)(latent_inputs)

vae_outputs = layers.Dense(
    num_features,
    activation="sigmoid"
)(d)

decoder = Model(
    latent_inputs,
    vae_outputs,
    name="decoder"
)


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
            z_mean_t, z_log_var_t = self.encoder(
                data,
                training=True
            )

            z = sampling_fn(
                z_mean_t,
                z_log_var_t
            )

            reconstruction = self.decoder(
                z,
                training=True
            )

            recon_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(data - reconstruction),
                    axis=1
                )
            )

            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1
                    + z_log_var_t
                    - tf.square(z_mean_t)
                    - tf.exp(z_log_var_t),
                    axis=1
                )
            )

            total_loss = recon_loss + self.beta * kl_loss

        variables = (
            self.encoder.trainable_variables
            + self.decoder.trainable_variables
        )

        grads = tape.gradient(
            total_loss,
            variables
        )

        self.optimizer.apply_gradients(
            zip(grads, variables)
        )

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss
        }


vae = BetaVAE(
    encoder,
    decoder,
    beta=beta
)

vae.compile(
    optimizer=tf.keras.optimizers.Adam()
)

print("\nβ-VAE Built Successfully")

# =======================================================================
# 14. TRAIN β-VAE
# =======================================================================

LOG_CSV = "unsw_vae_training_log.csv"

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

print(f"\nβ-VAE Training Finished in {vae_time:.1f}s")

# =======================================================================
# 15. EXTRACT LATENT FEATURES
# =======================================================================

z_mean_train, _ = encoder.predict(
    X_train,
    verbose=0
)

z_mean_test, _ = encoder.predict(
    X_test,
    verbose=0
)

z_mean_combined, _ = encoder.predict(
    X_combined_train,
    verbose=0
)

print("\nLatent Train Shape:", z_mean_train.shape)
print("Latent Combined Shape:", z_mean_combined.shape)
print("Latent Test Shape:", z_mean_test.shape)

# =======================================================================
# 16. LIGHTGBM ON β-VAE LATENT FEATURES
# =======================================================================

classes_all = np.unique(y_combined_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes_all,
    y=y_combined_train
)

weight_dict = dict(
    zip(classes_all, weights)
)

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

model_lgb = lgb.train(
    params,
    train_data,
    valid_sets=[test_data],
    num_boost_round=300,
    callbacks=[
        lgb.log_evaluation(period=10)
    ]
)

lgb_time = time.time() - start_lgb

print(f"\nLightGBM Training Finished in {lgb_time:.1f}s")

# =======================================================================
# 17. EVALUATION
# =======================================================================

print("\nEvaluating UNSW-NB15 GAN + β-VAE + LightGBM...")

y_pred_probs = model_lgb.predict(
    z_mean_test
)

y_pred_classes = np.argmax(
    y_pred_probs,
    axis=1
)

acc = accuracy_score(
    y_test,
    y_pred_classes
)

prec = precision_score(
    y_test,
    y_pred_classes,
    average="weighted",
    zero_division=0
)

rec = recall_score(
    y_test,
    y_pred_classes,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred_classes,
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_pred_classes,
    average="macro",
    zero_division=0
)

print("\n=== Evaluation Metrics: UNSW-NB15 ===")
print(f"Accuracy    : {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision   : {prec:.4f} ({prec*100:.2f}%)")
print(f"Recall      : {rec:.4f} ({rec*100:.2f}%)")
print(f"F1-Score    : {f1:.4f} ({f1*100:.2f}%)")
print(f"Macro F1    : {macro_f1:.4f} ({macro_f1*100:.2f}%)")

print("\n=== Classification Report ===")
print(
    classification_report(
        y_test,
        y_pred_classes,
        target_names=classes,
        digits=4,
        zero_division=0
    )
)

# =======================================================================
# 18. MINORITY-CLASS F1
# =======================================================================

report = classification_report(
    y_test,
    y_pred_classes,
    target_names=classes,
    output_dict=True,
    zero_division=0
)

# Minority classes are all non-Normal classes with relatively small support
minority_classes = [
    cls for cls in classes
    if cls.lower() != "normal"
]

minority_f1_scores = []

for cls in minority_classes:
    if cls in report:
        minority_f1_scores.append(
            report[cls]["f1-score"]
        )

minority_f1 = np.mean(
    minority_f1_scores
)

print(
    f"\nMinority-Class F1 : "
    f"{minority_f1:.4f} ({minority_f1*100:.2f}%)"
)

# =======================================================================
# 19. CONFUSION MATRIX
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
    cbar_kws={
        "label": "Count"
    }
)

plt.title(
    "Confusion Matrix — UNSW-NB15 GAN + β-VAE + LightGBM"
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# =======================================================================
# 20. TRAINING TIMES
# =======================================================================

print("\nTraining Times Summary")
print(f"GAN Training Time      : {gan_time:.1f}s")
print(f"β-VAE Training Time    : {vae_time:.1f}s")
print(f"LightGBM Training Time : {lgb_time:.1f}s")
print(f"Total Training Time    : {gan_time + vae_time + lgb_time:.1f}s")

result = {
    "Dataset": "UNSW-NB15",
    "Configuration": "GAN + β-VAE + LightGBM",
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

print("\n=== UNSW-NB15 Result ===")
print(result)