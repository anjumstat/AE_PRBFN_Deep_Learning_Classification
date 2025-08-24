

# -*- coding: utf-8 -*-
"""
Enhanced AE-RBFN with Loss Tracking, Averaged Attention Heatmap, and Full History Saving
@author: H.A.R
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, matthews_corrcoef, confusion_matrix)
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
import os
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)

    # I/O Configuration
    output_dir = 'E:/AE_RBFN/AE_RBFN_new/'
    os.makedirs(output_dir, exist_ok=True)
    data_path = 'E:/AE_RBFN/4_Species1.csv'

    # Data Loading
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1].values
    y = data['Species'].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    num_classes = len(np.unique(y))
    class_names = le.classes_

    # Train-Test Split (90-10)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42, stratify=y)

    # Normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to categorical
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

    # ================== MODEL COMPONENTS ================== #
    def phylogenetic_kmeans(X, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        kmeans.fit(X)
        return kmeans.cluster_centers_.astype(np.float32)

    class CodonGammaRBFLayer(layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        
        def build(self, input_shape):
            self.gamma = self.add_weight(
                shape=(input_shape[-1],),
                initializer=tf.keras.initializers.Constant(0.1),
                constraint=tf.keras.constraints.MinMaxNorm(0.01, 10.0),
                name='codon_gamma'
            )
            super().build(input_shape)
        
        def call(self, inputs, centroids):
            diff = tf.expand_dims(inputs, 1) - tf.expand_dims(centroids, 0)
            weighted_dist = tf.reduce_sum(diff**2 * tf.nn.softplus(self.gamma), axis=-1)
            return tf.exp(-0.5 * weighted_dist)

    class SpeciesAttention(layers.Layer):
        def __init__(self, num_species=4, **kwargs):
            super().__init__(**kwargs)
            self.num_species = num_species
        
        def build(self, input_shape):
            self.embedding = layers.Embedding(
                input_dim=self.num_species,
                output_dim=input_shape[-1],
                embeddings_initializer='glorot_uniform'
            )
            self.norm = layers.LayerNormalization()
            super().build(input_shape)
        
        def call(self, inputs, species_ids):
            attn_weights = tf.sigmoid(self.norm(self.embedding(species_ids)))
            return attn_weights * inputs

    class AERBFN(tf.keras.Model):
        def __init__(self, num_centroids=50, num_classes=4, **kwargs):
            super().__init__(**kwargs)
            self.num_centroids = min(num_centroids, X_train.shape[0]//10)
            self.centroids = tf.Variable(
                initial_value=phylogenetic_kmeans(X_train, self.num_centroids),
                trainable=True,
                name='centroids'
            )
            self.codon_gamma = CodonGammaRBFLayer()
            self.species_attention = SpeciesAttention(num_species=num_classes)
            self.output_layer = layers.Dense(num_classes, activation='softmax')
        
        def call(self, inputs):
            x, species_ids = inputs
            x = self.species_attention(x, species_ids)
            x = self.codon_gamma(x, self.centroids)
            return self.output_layer(x)

    # ================== TRAINING EXECUTION ================== #
    def calculate_class_metrics(y_true, y_pred, num_classes):
        """Calculate class-wise metrics (precision, recall, f1, mcc)"""
        class_metrics = {}
        
        for class_idx in range(num_classes):
            # Create binary labels for this class
            y_true_binary = (y_true == class_idx).astype(int)
            y_pred_binary = (y_pred == class_idx).astype(int)
            
            # Calculate metrics
            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            mcc = matthews_corrcoef(y_true_binary, y_pred_binary)
            
            class_metrics[f'precision_class_{class_idx}'] = precision
            class_metrics[f'recall_class_{class_idx}'] = recall
            class_metrics[f'f1_class_{class_idx}'] = f1
            class_metrics[f'mcc_class_{class_idx}'] = mcc
        
        return class_metrics

    def calculate_metrics(y_true, y_pred):
        """Calculate overall metrics and class-wise metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro'),
            'recall': recall_score(y_true, y_pred, average='macro'),
            'f1': f1_score(y_true, y_pred, average='macro'),
            'mcc': matthews_corrcoef(y_true, y_pred)
        }
        
        # Add class-wise metrics
        class_metrics = calculate_class_metrics(y_true, y_pred, num_classes)
        metrics.update(class_metrics)
        
        return metrics

    def train_ae_rbfn():
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        
        # Initialize metrics storage
        base_metrics = ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'training_time']
        metrics = {m: [] for m in base_metrics}
        test_metrics = {m: [] for m in base_metrics if m != 'training_time'}
        
        # Initialize class-wise metrics storage
        for class_idx in range(num_classes):
            for metric in ['precision', 'recall', 'f1', 'mcc']:
                metrics[f'{metric}_class_{class_idx}'] = []
                test_metrics[f'{metric}_class_{class_idx}'] = []
        
        conf_matrices = []
        all_histories = []
        all_attention_weights = []  # Store attention weights from all folds
        
        # Create a DataFrame to store fold-wise validation metrics
        fold_validation_df = pd.DataFrame(columns=[
            'Fold', 'Average_Validation_Accuracy', 'Max_Validation_Accuracy',
            'Final_Validation_Accuracy', 'Epochs'
        ])
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            print(f"\n=== Fold {fold+1}/10 ===")
            
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model = AERBFN(num_centroids=32, num_classes=num_classes)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Updated EarlyStopping callback with min_delta=0.005
            early_stop = callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                min_delta=0.005,  # Minimum improvement threshold
                mode='max',
                restore_best_weights=True
            )
            
            start_time = time.time()
            history = model.fit(
                x=(X_tr, y_tr),
                y=tf.keras.utils.to_categorical(y_tr, num_classes),
                validation_data=((X_val, y_val), tf.keras.utils.to_categorical(y_val, num_classes)),
                epochs=200,
                batch_size=1024,
                verbose=1,
                callbacks=[early_stop]
            )
            training_time = time.time() - start_time
            metrics['training_time'].append(training_time)
            
            # Calculate fold validation metrics
            val_acc_history = history.history['val_accuracy']
            avg_val_acc = np.mean(val_acc_history)
            max_val_acc = np.max(val_acc_history)
            final_val_acc = val_acc_history[-1]
            epochs_trained = len(val_acc_history)
            
            # Add to fold validation DataFrame
            fold_validation_df.loc[fold] = [
                fold+1,
                avg_val_acc,
                max_val_acc,
                final_val_acc,
                epochs_trained
            ]
            
            # Save full history (loss + accuracy)
            all_histories.append({
                'accuracy': history.history['accuracy'],
                'val_accuracy': history.history['val_accuracy'],
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss']
            })
            
            # Save attention weights
            attn_weights = model.species_attention.embedding.weights[0].numpy()
            all_attention_weights.append(attn_weights)
            
            # Validation metrics
            y_val_pred = model.predict((X_val, y_val), verbose=0)
            y_val_pred_classes = np.argmax(y_val_pred, axis=1)
            fold_metrics = calculate_metrics(y_val, y_val_pred_classes)
            for metric in fold_metrics:
                if metric in metrics:
                    metrics[metric].append(fold_metrics[metric])
            
            # Test metrics
            y_test_pred = model.predict((X_test, y_test), verbose=0)
            y_test_pred_classes = np.argmax(y_test_pred, axis=1)
            test_fold_metrics = calculate_metrics(y_test, y_test_pred_classes)
            for metric in test_fold_metrics:
                if metric in test_metrics:
                    test_metrics[metric].append(test_fold_metrics[metric])
            
            # Save model weights and confusion matrix
            model.save_weights(os.path.join(output_dir, f'AE-RBFN_fold{fold+1}.weights.h5'))
            conf_matrices.append(confusion_matrix(y_val, y_val_pred_classes))
        
        # Save fold validation metrics to CSV
        fold_validation_df.to_csv(os.path.join(output_dir, 'Fold_Validation_Metrics.csv'), index=False)
        
        # ================== Save All History Files ================== #
        # Pad histories to equal length (for averaging)
        max_epochs = max(len(h['accuracy']) for h in all_histories)
        padded_histories = []
        for h in all_histories:
            padded = {}
            for key in ['accuracy', 'val_accuracy', 'loss', 'val_loss']:
                padded[key] = np.pad(h[key], (0, max_epochs - len(h[key])), mode='edge')
            padded_histories.append(padded)
        
        # Compute averages
        avg_train_acc = np.mean([h['accuracy'] for h in padded_histories], axis=0)
        avg_val_acc = np.mean([h['val_accuracy'] for h in padded_histories], axis=0)
        avg_train_loss = np.mean([h['loss'] for h in padded_histories], axis=0)
        avg_val_loss = np.mean([h['val_loss'] for h in padded_histories], axis=0)
        
        # Save averaged metrics
        np.save(os.path.join(output_dir, 'avg_train_acc.npy'), avg_train_acc)
        np.save(os.path.join(output_dir, 'avg_val_acc.npy'), avg_val_acc)
        np.save(os.path.join(output_dir, 'avg_train_loss.npy'), avg_train_loss)
        np.save(os.path.join(output_dir, 'avg_val_loss.npy'), avg_val_loss)
        
        # Save raw histories (for flexibility)
        np.save(os.path.join(output_dir, 'all_histories.npy'), all_histories)
        
        # ================== Plot Accuracy/Loss Curves ================== #
        plt.figure(figsize=(12, 5))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(avg_train_acc, label='Training Accuracy', linewidth=2, color='blue')
        plt.plot(avg_val_acc, label='Validation Accuracy', linewidth=2, color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('AE-RBFN Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(avg_train_loss, label='Training Loss', linewidth=2, color='red')
        plt.plot(avg_val_loss, label='Validation Loss', linewidth=2, color='purple')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('AE-RBFN Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Training_History.png'))
        plt.close()
        
        # ================== Plot Averaged Attention Heatmap ================== #
        avg_attention_weights = np.mean(all_attention_weights, axis=0)
        np.save(os.path.join(output_dir, 'avg_attention_weights.npy'), avg_attention_weights)
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            avg_attention_weights, 
            cmap="YlOrRd", 
            xticklabels=[f"Codon {i+1}" for i in range(64)],
            yticklabels=class_names
        )
        plt.title("Average Species-Specific Codon Attention Weights (10 Folds)")
        plt.savefig(os.path.join(output_dir, 'Species_Attention_Avg.png'))
        plt.close()
        
        # ================== Save Results ================== #
        avg_conf_matrix = np.mean(conf_matrices, axis=0).astype(int)
        np.save(os.path.join(output_dir, 'Confusion_Matrix_Avg.npy'), avg_conf_matrix)
        
        # Create comprehensive results DataFrame
        results_data = []
        
        # Overall metrics
        overall_metrics = ['accuracy', 'precision', 'recall', 'f1', 'mcc']
        for metric in overall_metrics:
            results_data.append({
                'Metric': metric.capitalize(),
                'Type': 'Overall',
                'Class': 'All',
                'Validation_Mean': np.mean(metrics[metric]),
                'Validation_Std': np.std(metrics[metric]),
                'Test_Mean': np.mean(test_metrics[metric]),
                'Test_Std': np.std(test_metrics[metric])
            })
        
        # Class-wise metrics
        for class_idx in range(num_classes):
            for metric_type in ['precision', 'recall', 'f1', 'mcc']:
                metric_key = f'{metric_type}_class_{class_idx}'
                results_data.append({
                    'Metric': metric_type.capitalize(),
                    'Type': 'Class-wise',
                    'Class': f'Class_{class_idx} ({class_names[class_idx]})',
                    'Validation_Mean': np.mean(metrics[metric_key]),
                    'Validation_Std': np.std(metrics[metric_key]),
                    'Test_Mean': np.mean(test_metrics[metric_key]),
                    'Test_Std': np.std(test_metrics[metric_key])
                })
        
        results = pd.DataFrame(results_data)
        results.to_csv(os.path.join(output_dir, 'AE-RBFN_Metrics.csv'), index=False)
        
        # Save detailed metrics per fold
        detailed_metrics = []
        for fold in range(10):
            fold_data = {'Fold': fold + 1}
            for metric in metrics:
                if metric != 'training_time':
                    fold_data[metric] = metrics[metric][fold]
            detailed_metrics.append(fold_data)
        
        detailed_df = pd.DataFrame(detailed_metrics)
        detailed_df.to_csv(os.path.join(output_dir, 'AE-RBFN_Detailed_Metrics.csv'), index=False)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            avg_conf_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names, 
            yticklabels=class_names
        )
        plt.title('AE-RBFN Average Confusion Matrix')
        plt.savefig(os.path.join(output_dir, 'Confusion_Matrix.png'))
        plt.close()
        
        # Print summary
        print("\n=== Training Summary ===")
        print(f"Average Validation Accuracy: {np.mean(metrics['accuracy']):.4f} ± {np.std(metrics['accuracy']):.4f}")
        print(f"Average Test Accuracy: {np.mean(test_metrics['accuracy']):.4f} ± {np.std(test_metrics['accuracy']):.4f}")
        
        print("\n=== Class-wise Test Metrics ===")
        for class_idx in range(num_classes):
            print(f"Class {class_idx} ({class_names[class_idx]}):")
            print(f"  Precision: {np.mean(test_metrics[f'precision_class_{class_idx}']):.4f} ± {np.std(test_metrics[f'precision_class_{class_idx}']):.4f}")
            print(f"  Recall: {np.mean(test_metrics[f'recall_class_{class_idx}']):.4f} ± {np.std(test_metrics[f'recall_class_{class_idx}']):.4f}")
            print(f"  F1 Score: {np.mean(test_metrics[f'f1_class_{class_idx}']):.4f} ± {np.std(test_metrics[f'f1_class_{class_idx}']):.4f}")
            print(f"  MCC: {np.mean(test_metrics[f'mcc_class_{class_idx}']):.4f} ± {np.std(test_metrics[f'mcc_class_{class_idx}']):.4f}")

    # ================== MAIN EXECUTION ================== #
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    train_ae_rbfn()
    print("Training completed. Results saved to:", output_dir)

if __name__ == "__main__":
    main()