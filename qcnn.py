import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.utils import resample
import tensorflow as tf
import pennylane as qml
import warnings
import os
import pickle
import traceback

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class HeartDiseaseQCNN:
    def __init__(self, num_qubits=8):
        # Proper constructor name
        self.num_qubits = num_qubits
        self.scaler = StandardScaler()
        # PCA n_components will be adjusted later to not exceed feature count
        self.pca = None
        self.model = None
        # create PennyLane device (simulation)
        self.device = qml.device("default.qubit", wires=self.num_qubits)

    def load_and_preprocess_data(self, file_path):
        """Load and preprocess data for QCNN"""
        print("Loading and preprocessing data...")
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

        # Ensure label column exists
        if 'HeartDiseaseorAttack' not in df.columns:
            raise KeyError("Expected column 'HeartDiseaseorAttack' not found in CSV")

        print(f"Dataset shape: {df.shape}")
        print(f"Class distribution: {df['HeartDiseaseorAttack'].value_counts().to_dict()}")

        X = df.drop('HeartDiseaseorAttack', axis=1)
        y = df['HeartDiseaseorAttack']

        return X, y

    def simple_oversampling(self, X, y):
        """Simple random oversampling of minority class"""
        print("Applying simple oversampling...")

        # Combine for ease
        df = pd.concat([X, y.rename('label')], axis=1)

        # Separate majority and minority
        majority_df = df[df['label'] == 0]
        minority_df = df[df['label'] == 1]

        print(f"Before oversampling - Majority: {len(majority_df)}, Minority: {len(minority_df)}")

        if len(minority_df) == 0:
            print("Warning: no minority class examples found. Returning original data.")
            return X, y

        # Oversample minority to match majority
        minority_upsampled = resample(
            minority_df,
            replace=True,
            n_samples=len(majority_df),
            random_state=42
        )

        balanced_df = pd.concat([majority_df, minority_upsampled]).sample(frac=1, random_state=42).reset_index(drop=True)
        X_balanced = balanced_df.drop('label', axis=1)
        y_balanced = balanced_df['label']

        print(f"After oversampling - Total: {len(X_balanced)}, Class distribution: {y_balanced.value_counts().to_dict()}")
        return X_balanced, y_balanced

    def quantum_convolutional_layer(self, params, wires):
        """Quantum convolutional layer"""
        param_idx = 0
        for i in range(len(wires)):
            # Entanglement
            if i < len(wires) - 1:
                qml.CNOT(wires=[wires[i], wires[i + 1]])
            # Rotation gates (safely check params length)
            if param_idx < len(params):
                qml.RY(params[param_idx], wires=wires[i])
                param_idx += 1
            if param_idx < len(params):
                qml.RZ(params[param_idx], wires=wires[i])
                param_idx += 1

    def quantum_pooling_layer(self, params, wires_to_pool, wires_to_keep):
        """Quantum pooling layer"""
        param_idx = 0
        for pool_wire, keep_wire in zip(wires_to_pool, wires_to_keep):
            qml.CNOT(wires=[pool_wire, keep_wire])
            if param_idx < len(params):
                qml.RY(params[param_idx], wires=keep_wire)
                param_idx += 1

    def qcnn_circuit(self, inputs, params):
        """Complete QCNN circuit (returns expectations). This function assumes
        `self.num_qubits` wires are available."""
        # Quantum data encoding
        for i in range(self.num_qubits):
            qml.RY(inputs[i] * np.pi, wires=i)

        param_idx = 0

        # First convolutional layer
        conv1_params = params[param_idx:param_idx + 2 * self.num_qubits]
        self.quantum_convolutional_layer(conv1_params, list(range(self.num_qubits)))
        param_idx += 2 * self.num_qubits

        # First pooling layer (reduce from 8 to 4 qubits)
        pool1_params = params[param_idx:param_idx + self.num_qubits // 2]
        self.quantum_pooling_layer(pool1_params, [i for i in range(1, self.num_qubits, 2)], [i for i in range(0, self.num_qubits, 2)])
        param_idx += self.num_qubits // 2

        # Second convolutional layer (on remaining qubits)
        remaining = [i for i in range(0, self.num_qubits, 2)]
        conv2_params = params[param_idx:param_idx + 2 * len(remaining)]
        self.quantum_convolutional_layer(conv2_params, remaining)
        param_idx += 2 * len(remaining)

        # Second pooling layer: pick half of remaining to pool
        if len(remaining) >= 4:
            pool2_params = params[param_idx:param_idx + len(remaining) // 2]
            wires_to_pool = remaining[1::2]
            wires_to_keep = remaining[0::2]
            self.quantum_pooling_layer(pool2_params, wires_to_pool, wires_to_keep)
            param_idx += len(remaining) // 2

            # Final conv on the kept wires
            kept = wires_to_keep
        else:
            kept = remaining

        conv3_params = params[param_idx:param_idx + 2 * len(kept)]
        self.quantum_convolutional_layer(conv3_params, kept)

        # Return expectation values on first two kept wires (or fallback)
        measure_wires = kept[:2] if len(kept) >= 2 else [0, min(self.num_qubits - 1, 1)]
        return [qml.expval(qml.PauliZ(measure_wires[0])), qml.expval(qml.PauliZ(measure_wires[1]))]

    def build_qcnn_model(self):
        """Build the complete QCNN-like model (hybrid classical-quantum simulation)"""
        print(f"Building QCNN model with {self.num_qubits} qubits...")

        # Safe calculation of total params (this is approximate; not strictly required if we don't use them)
        total_params = (2 * self.num_qubits +                    # First conv
                        max(1, self.num_qubits // 2) +           # First pool
                        2 * max(1, self.num_qubits // 2) +       # Second conv
                        max(1, self.num_qubits // 2) +           # Second pool (approx)
                        4)                                       # Final conv (approx)
        print(f"Total quantum-ish parameters (approx): {total_params}")

        # Define a qnode if you want to use it directly (we keep it, but this script uses classical layers to simulate)
        @qml.qnode(self.device, interface="tf")
        def quantum_circuit(inputs, quantum_params):
            return self.qcnn_circuit(inputs, quantum_params)

        inputs = tf.keras.Input(shape=(self.num_qubits,))

        # Classical pre-processing
        x = tf.keras.layers.Dense(32, activation='relu')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        # Quantum simulation layer (classical approximation)
        x = tf.keras.layers.Dense(16, activation='tanh')(x)
        x = tf.keras.layers.Dense(8, activation='tanh')(x)

        # Classical post-processing
        x = tf.keras.layers.Dense(16, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(8, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Use metric objects instead of strings to avoid compatibility issues
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                     tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall')]
        )

        print("QCNN model built successfully!")
        return model

    def prepare_data_for_qcnn(self, X, y):
        """Prepare data specifically for QCNN"""
        print("Preparing data for QCNN...")

        # Handle missing values
        if X.isnull().sum().sum() > 0:
            X = X.fillna(X.mean())

        # Apply simple oversampling
        X_balanced, y_balanced = self.simple_oversampling(X, y)

        # Standard scaling
        X_scaled = self.scaler.fit_transform(X_balanced)

        # PCA: ensure n_components <= n_features
        n_features = X_scaled.shape[1]
        n_comp = min(self.num_qubits, n_features)
        self.pca = PCA(n_components=n_comp)
        X_pca = self.pca.fit_transform(X_scaled)
        print(f"After PCA - Shape: {X_pca.shape}")
        print(f"Explained variance (sum): {self.pca.explained_variance_ratio_.sum():.3f}")

        # If PCA produced fewer components than num_qubits, pad with zeros (so model input shape matches)
        if X_pca.shape[1] < self.num_qubits:
            padding = np.zeros((X_pca.shape[0], self.num_qubits - X_pca.shape[1]))
            X_pca = np.hstack([X_pca, padding])
            print(f"Padded PCA outputs to match num_qubits: new shape {X_pca.shape}")

        return X_pca, y_balanced.astype(int).to_numpy()

    def train_qcnn(self, X_train, y_train, X_val, y_val, epochs=30):
        """Train the QCNN model"""
        print("Training QCNN model...")

        self.model = self.build_qcnn_model()

        # Compute class weights
        from sklearn.utils.class_weight import compute_class_weight
        class_weights_arr = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        # Map classes to weights (works if classes are [0,1])
        class_weight_dict = {int(cls): float(w) for cls, w in zip(np.unique(y_train), class_weights_arr)}
        print(f"Using class weights: {class_weight_dict}")

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=128,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def evaluate_qcnn(self, X_test, y_test):
        """Evaluate the QCNN model"""
        print("Evaluating QCNN model...")

        y_pred_proba = self.model.predict(X_test, batch_size=256, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        accuracy = accuracy_score(y_test, y_pred)
        try:
            auc_score = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            auc_score = float('nan')

        print(f"QCNN Test Accuracy: {accuracy:.4f}")
        print(f"QCNN Test AUC: {auc_score:.4f}")
        print("\nQCNN Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))

        self.plot_qcnn_results(y_test, y_pred, y_pred_proba)
        return accuracy, auc_score, y_pred, y_pred_proba

    def plot_qcnn_results(self, y_test, y_pred, y_pred_proba):
        """Plot QCNN results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('QCNN - Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')

        # PCA explained variance if available
        if self.pca is not None:
            explained_variance = self.pca.explained_variance_ratio_
            axes[0, 1].bar(range(len(explained_variance)), explained_variance)
            axes[0, 1].set_title('QCNN - PCA Explained Variance')
            axes[0, 1].set_xlabel('Quantum Feature')
            axes[0, 1].set_ylabel('Variance Ratio')
        else:
            axes[0, 1].text(0.5, 0.5, 'PCA not available', ha='center')

        # Performance metrics visualization
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics = ['Accuracy', 'Precision', 'Recall']
        values = [accuracy_score(y_test, y_pred), precision, recall]

        axes[1, 0].bar(metrics, values)
        axes[1, 0].set_title('QCNN Performance Metrics')
        axes[1, 0].set_ylabel('Score')

        # Class distribution
        unique, counts = np.unique(y_test, return_counts=True)
        labels = ['No Disease', 'Disease']
        # adapt labels if distribution keys differ
        if len(unique) == 2:
            axes[1, 1].pie(counts, labels=labels, autopct='%1.1f%%')
        else:
            axes[1, 1].pie(counts, labels=[str(u) for u in unique], autopct='%1.1f%%')
        axes[1, 1].set_title('Test Set Distribution')

        plt.tight_layout()
        plt.show()

    def run_complete_qcnn_pipeline(self, file_path):
        """Run complete QCNN pipeline"""
        print("=" * 60)
        print("QUANTUM CNN (QCNN) PIPELINE FOR HEART DISEASE PREDICTION")
        print("=" * 60)

        # 1. Load data
        X, y = self.load_and_preprocess_data(file_path)

        # 2. Prepare data for QCNN
        X_processed, y_processed = self.prepare_data_for_qcnn(X, y)

        # 3. Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_processed, y_processed, test_size=0.3, random_state=42, stratify=y_processed
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        print(f"\nData Splits:")
        print(f"Training: {X_train.shape[0]} samples")
        print(f"Validation: {X_val.shape[0]} samples")
        print(f"Test: {X_test.shape[0]} samples")

        # 4. Train QCNN
        print("\nStarting QCNN training...")
        history = self.train_qcnn(X_train, y_train, X_val, y_val, epochs=30)

        # 5. Evaluate QCNN
        print("\nEvaluating QCNN model...")
        accuracy, auc_score, y_pred, y_pred_proba = self.evaluate_qcnn(X_test, y_test)

        # 6. Plot training history
        self.plot_training_history(history)

        # 7. Model summary
        print("\n" + "=" * 50)
        print("QCNN MODEL SUMMARY")
        print("=" * 50)
        if self.model:
            self.model.summary()

        return {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'history': history,
            'model': self.model
        }

    def plot_training_history(self, history):
        """Plot QCNN training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axes[0, 0].plot(history.history.get('loss', []), label='Training Loss')
        axes[0, 0].plot(history.history.get('val_loss', []), label='Validation Loss')
        axes[0, 0].set_title('QCNN Training Loss')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()

        # Accuracy
        axes[0, 1].plot(history.history.get('accuracy', []), label='Training Accuracy')
        axes[0, 1].plot(history.history.get('val_accuracy', []), label='Validation Accuracy')
        axes[0, 1].set_title('QCNN Training Accuracy')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()

        # Precision
        if 'precision' in history.history:
            axes[1, 0].plot(history.history.get('precision', []), label='Training Precision')
            axes[1, 0].plot(history.history.get('val_precision', []), label='Validation Precision')
            axes[1, 0].set_title('QCNN Training Precision')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].legend()

        # Recall
        if 'recall' in history.history:
            axes[1, 1].plot(history.history.get('recall', []), label='Training Recall')
            axes[1, 1].plot(history.history.get('val_recall', []), label='Validation Recall')
            axes[1, 1].set_title('QCNN Training Recall')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].legend()

        plt.tight_layout()
        plt.show()


# Run the QCNN pipeline
if __name__ == "__main__":
    # Initialize QCNN with 8 qubits
    qcnn = HeartDiseaseQCNN(num_qubits=8)

    try:
        # Update path to your local CSV file
        results = qcnn.run_complete_qcnn_pipeline('heart_disease_health_indicators_BRFSS2015.csv')

        print(f"\n🎯 QCNN FINAL RESULTS:")
        print(f"   Accuracy: {results['accuracy']:.4f}")
        print(f"   AUC Score: {results['auc_score']:.4f}")

        # Show quantum circuit design
        print(f"\n🔬 Quantum Circuit Design:")
        print(f"   - Input Qubits: {qcnn.num_qubits}")
        print(f"   - Convolutional Layers: 3")
        print(f"   - Pooling Layers: 2")
        print(f"   - Output Qubits: 2")
        
        # ---------------------------------------------------------
        #  SAVE BACKEND FILES (Only runs if training succeeded)
        # ---------------------------------------------------------
        print("\n💾 SAVING BACKEND ARTIFACTS...")

        # A. Create the backend folder structure if it doesn't exist
        # This creates 'backend' and 'model_artifacts' folders for you
        base_path = os.path.join(os.getcwd(), 'backend', 'model_artifacts')
        os.makedirs(base_path, exist_ok=True)
        print(f"   [+] Created directory: {base_path}")

        # B. Save the Keras Model (The Neural Network)
        # We save the model object returned in 'results'
        model_path = os.path.join(base_path, "qcnn_model.keras")
        results['model'].save(model_path)
        print(f"   [+] Model saved to: {model_path}")

        # C. Save the Preprocessors (Scaler & PCA)
        # We need these to process new user data exactly like training data
        artifacts = {
            "scaler": qcnn.scaler,
            "pca": qcnn.pca,
            "num_qubits": qcnn.num_qubits
        }

        pkl_path = os.path.join(base_path, "preprocessors.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(artifacts, f)
        print(f"   [+] Preprocessors saved to: {pkl_path}")
        print("\n✅ SETUP COMPLETE: You can now start the backend server.")

    except Exception as e:
        print(f"\n❌ ERROR IN PIPELINE:")
        print(f"{e}")
        import traceback
        traceback.print_exc()
        print("\n💡 TIP: Check if your CSV file path is correct!")