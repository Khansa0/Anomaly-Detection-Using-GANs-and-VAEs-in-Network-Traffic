import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess CICIDS2017 dataset"""
        print("Loading dataset...")
        
        # Load dataset
        try:
            df = pd.read_csv(file_path)
            print(f"Successfully loaded: {file_path}")
        except Exception as e:
            print(f"Error loading file: {e}")
            return None, None
        
        print(f"Original dataset shape: {df.shape}")
        
        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()
        
        # Print column names and types to understand the data
        print("\nDataset info:")
        print(f"Data types:\n{df.dtypes.value_counts()}")
        
        # Handle the dataset
        df = self._preprocess_data(df)
        
        if df is None:
            return None, None
            
        # Select features
        X, y = self._select_features(df)
        
        return X, y
    
    def _preprocess_data(self, df):
        """Preprocess the dataset"""
        print("Preprocessing data...")
        
        # Step 1: Handle the Label column properly
        label_column = None
        possible_label_names = ['Label', 'label', 'Attack', 'attack', 'is_anomaly']
        
        for col in possible_label_names:
            if col in df.columns:
                label_column = col
                break
        
        if label_column:
            print(f"Found label column: '{label_column}'")
            
            # Check what labels we have
            unique_labels = df[label_column].unique()
            print(f"Unique labels found: {unique_labels[:10]}...")
            
            # Convert labels to binary (0 for normal/BENIGN, 1 for attacks)
            if df[label_column].dtype == 'object':
                df['is_anomaly'] = df[label_column].apply(
                    lambda x: 0 if str(x).strip().upper() in ['BENIGN', 'NORMAL', '0'] else 1
                )
            else:
                df['is_anomaly'] = (df[label_column] != 0).astype(int)
            
            # Remove the original label column
            df = df.drop(columns=[label_column])
            print(f"Label column '{label_column}' processed and removed")
        else:
            print("No label column found in dataset.")
            return None
        
        # Step 2: Remove any other non-numeric columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            print(f"Removing non-numeric columns: {non_numeric_cols.tolist()}")
            df = df.drop(columns=non_numeric_cols)
        
        # Step 3: Handle infinite values and NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        # Step 4: Remove duplicate rows
        initial_size = len(df)
        df = df.drop_duplicates()
        final_size = len(df)
        print(f"Removed {initial_size - final_size} duplicate rows")
        
        # Step 5: Remove constant columns (zero variance)
        feature_cols = [col for col in df.columns if col != 'is_anomaly']
        constant_cols = []
        for col in feature_cols:
            if df[col].std() == 0:
                constant_cols.append(col)
        
        if constant_cols:
            print(f"Removing constant columns: {constant_cols}")
            df = df.drop(columns=constant_cols)
        
        print(f"Final dataset shape after preprocessing: {df.shape}")
        if 'is_anomaly' in df.columns:
            print(f"Class distribution: {df['is_anomaly'].value_counts()}")
        return df
    
    def _select_features(self, df):
        """Select features for modeling"""
        print("Selecting features...")
        
        if 'is_anomaly' not in df.columns:
            print("Error: 'is_anomaly' column not found!")
            return None, None
        
        feature_cols = [col for col in df.columns if col != 'is_anomaly']
        self.feature_columns = feature_cols
        X = df[feature_cols].values
        y = df['is_anomaly'].values
        
        print(f"Selected {len(feature_cols)} features")
        print(f"Class distribution - Normal: {sum(y==0)}, Attacks: {sum(y==1)}")
        print(f"Attack ratio: {sum(y==1)/len(y):.4f}")
        
        return X, y
    
    def scale_data(self, X_train, X_test):
        """Scale the data"""
        print("Scaling data...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

class ImprovedGAN:
    def __init__(self, input_dim, latent_dim=100):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.gan = self._build_gan()
        
    def _build_generator(self):
        """Build generator model"""
        model = keras.Sequential([
            layers.Dense(128, input_dim=self.latent_dim, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(self.input_dim, activation='tanh')
        ])
        return model
    
    def _build_discriminator(self):
        """Build discriminator model"""
        model = keras.Sequential([
            layers.Dense(512, input_dim=self.input_dim, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def _build_gan(self):
        """Build GAN model"""
        self.discriminator.compile(
            optimizer=keras.optimizers.Adam(0.0002, 0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.discriminator.trainable = False
        
        gan_input = layers.Input(shape=(self.latent_dim,))
        generated_data = self.generator(gan_input)
        gan_output = self.discriminator(generated_data)
        
        gan = keras.Model(gan_input, gan_output)
        gan.compile(
            optimizer=keras.optimizers.Adam(0.0002, 0.5),
            loss='binary_crossentropy'
        )
        
        return gan
    
    def train(self, X_train, epochs=50, batch_size=32):
        """Train the GAN"""
        print(f"Training GAN for {epochs} epochs...")
        
        # Normalize data to [-1, 1] for GAN training
        X_min = X_train.min(axis=0)
        X_max = X_train.max(axis=0)
        X_train_gan = (X_train - X_min) / (X_max - X_min + 1e-8) * 2 - 1
        
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            # Train discriminator
            idx = np.random.randint(0, X_train_gan.shape[0], batch_size)
            real_samples = X_train_gan[idx]
            
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            generated_samples = self.generator.predict(noise, verbose=0)
            
            d_loss_real = self.discriminator.train_on_batch(real_samples, real)
            d_loss_fake = self.discriminator.train_on_batch(generated_samples, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, real)
            
            if epoch % 25 == 0:
                print(f"Epoch {epoch}, D Loss: {d_loss[0]:.4f}, G Loss: {g_loss:.4f}")
    
    def generate_samples(self, n_samples):
        """Generate synthetic samples"""
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        generated_samples = self.generator.predict(noise, verbose=0)
        return generated_samples

class SimpleVAE:
    def __init__(self, input_dim, latent_dim=20):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.model = self._build_vae()
        
    def _build_vae(self):
        """Build a simple VAE using Sequential model to avoid KerasTensor issues"""
        # Encoder
        encoder_inputs = keras.Input(shape=(self.input_dim,))
        
        # Encoder network
        x = layers.Dense(256, activation='relu')(encoder_inputs)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        
        # Latent space
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        
        # Use the reparameterization trick
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        z = layers.Lambda(sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])
        
        # Decoder
        decoder_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(64, activation='relu')(decoder_inputs)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(256, activation='relu')(x)
        decoder_outputs = layers.Dense(self.input_dim, activation='sigmoid')(x)
        
        decoder = keras.Model(decoder_inputs, decoder_outputs, name='decoder')
        
        # VAE outputs
        outputs = decoder(z)
        
        # Build VAE model
        vae = keras.Model(encoder_inputs, outputs, name='vae')
        
        # Add KL divergence loss and reconstruction loss
        reconstruction_loss = keras.losses.mse(encoder_inputs, outputs)
        reconstruction_loss *= self.input_dim
        
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')
        
        return vae
    
    def train(self, X_train, epochs=20, batch_size=128):
        """Train the VAE"""
        print(f"Training VAE for {epochs} epochs...")
        
        # Normalize data to [0, 1] for VAE
        self.X_min = X_train.min(axis=0)
        self.X_max = X_train.max(axis=0)
        X_train_vae = (X_train - self.X_min) / (self.X_max - self.X_min + 1e-8)
        
        history = self.model.fit(
            X_train_vae, X_train_vae,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            shuffle=True,
            verbose=1
        )
        
        return history
    
    def detect_anomalies(self, X_test, threshold_percentile=95):
        """Detect anomalies using reconstruction error"""
        # Normalize test data
        X_test_norm = (X_test - self.X_min) / (self.X_max - self.X_min + 1e-8)
        
        # Get reconstructions
        reconstructions = self.model.predict(X_test_norm, verbose=0)
        
        # Calculate reconstruction error
        reconstruction_error = np.mean(np.square(X_test_norm - reconstructions), axis=1)
        
        # Set threshold based on percentile
        threshold = np.percentile(reconstruction_error, threshold_percentile)
        
        # Predict anomalies
        predictions = (reconstruction_error > threshold).astype(int)
        
        return predictions, reconstruction_error, threshold

class GANVAEAnomalyDetector:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.gan = ImprovedGAN(input_dim)
        self.vae = SimpleVAE(input_dim)
        
    def train_models(self, X_train, y_train, gan_epochs=50, vae_epochs=20):
        """Train both GAN and VAE models"""
        
        # Separate normal and attack data
        normal_data = X_train[y_train == 0]
        attack_data = X_train[y_train == 1]
        
        print(f"Normal samples: {len(normal_data)}, Attack samples: {len(attack_data)}")
        
        # Train GAN on attack data to generate more attack samples
        if len(attack_data) > 50:
            print("Training GAN on attack data...")
            self.gan.train(attack_data, epochs=gan_epochs)
        else:
            print("Not enough attack samples for GAN training")
        
        # Train VAE on normal data only for anomaly detection
        print("Training VAE on normal data...")
        self.vae.train(normal_data, epochs=vae_epochs)
        
        return normal_data, attack_data
    
    def evaluate(self, X_test, y_test, threshold_percentile=95):
        """Evaluate the model"""
        print("Evaluating model...")
        
        # Use VAE for anomaly detection
        predictions, reconstruction_error, threshold = self.vae.detect_anomalies(
            X_test, threshold_percentile
        )
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        auc_score = roc_auc_score(y_test, reconstruction_error)
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC Score: {auc_score:.4f}")
        print(f"Anomaly Threshold: {threshold:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, target_names=['Normal', 'Anomaly']))
        
        # Confusion matrix
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # ROC Curve
        plt.subplot(1, 3, 2)
        fpr, tpr, _ = roc_curve(y_test, reconstruction_error)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        # Precision-Recall Curve
        plt.subplot(1, 3, 3)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, reconstruction_error)
        pr_auc = auc(recall_curve, precision_curve)
        
        plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        plt.tight_layout()
        plt.show()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score,
            'predictions': predictions,
            'reconstruction_error': reconstruction_error,
            'threshold': threshold
        }

def main():
    """Main function to run the complete pipeline"""
    print("GAN-VAE Anomaly Detection for Network Traffic")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Use just one file for testing
    file_path = "D:\\Khanssa\\CICIDS-2017\\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    
    try:
        print(f"Loading data from: {file_path}")
        X, y = preprocessor.load_and_preprocess_data(file_path)
        
        if X is None or y is None:
            print("Failed to load and preprocess data")
            return
        
        print(f"Final dataset shape: {X.shape}")
        print(f"Anomaly ratio: {y.mean():.4f}")
        
        # For large datasets, we can sample to make it manageable
        if len(X) > 50000:
            print("Dataset is large, sampling 50000 samples for faster training...")
            sample_indices = np.random.choice(len(X), 50000, replace=False)
            X = X[sample_indices]
            y = y[sample_indices]
            print(f"After sampling - X shape: {X.shape}, y shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale data
        X_train_scaled, X_test_scaled = preprocessor.scale_data(X_train, X_test)
        
        # Initialize GAN-VAE detector
        detector = GANVAEAnomalyDetector(input_dim=X_train_scaled.shape[1])
        
        # Train models
        normal_data, attack_data = detector.train_models(
            X_train_scaled, y_train, 
            gan_epochs=50,
            vae_epochs=20
        )
        
        # Evaluate model
        results = detector.evaluate(X_test_scaled, y_test, threshold_percentile=95)
        
        # Print final summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(f"Best Accuracy: {results['accuracy']:.4f}")
        print(f"Best F1-Score: {results['f1_score']:.4f}")
        print(f"AUC Score: {results['auc_score']:.4f}")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()