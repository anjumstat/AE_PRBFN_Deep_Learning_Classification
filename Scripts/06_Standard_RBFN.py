# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 20:08:26 2025

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
Enhanced Standard Radial Basis Function Network (RBFN) - Fixed Centroids
@author: H.A.R
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, matthews_corrcoef, confusion_matrix)
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import os
import time

# Configuration
CONFIG = {
    "data_path": 'E:/AE_RBFN/4_Species1.csv',
    "output_dir": 'E:/AE_RBFN/RBFN_Standard_new1/',
    "num_centroids": 50,          # Number of RBF centers
    "gamma": 0.1,                 # Fixed RBF width parameter
    "test_size": 0.10,
    "random_state": 42,
    "epochs": 1000,
    "batch_size": 1024
}

# Setup
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# Fixed GPU check
try:
    gpus = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available:", len(gpus))
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print("GPU check failed:", e)
    print("Running on CPU")

# Load data
data = pd.read_csv(CONFIG['data_path'])
X = data.iloc[:, :-1].values
y = data['Species'].values
le = LabelEncoder()
y = le.fit_transform(y)
num_classes = len(np.unique(y))
class_names = le.classes_

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=CONFIG['test_size'],
    random_state=CONFIG['random_state'],
    stratify=y
)

# Normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

# Convert labels to categorical
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

# Standard RBF Layer (Fixed Centroids)
class StandardRBFLayer(layers.Layer):
    def __init__(self, num_centroids, gamma, **kwargs):
        super(StandardRBFLayer, self).__init__(**kwargs)
        self.num_centroids = num_centroids
        self.gamma = gamma

    def build(self, input_shape):
        # Initialize centroids using K-means (fixed during training)
        kmeans = KMeans(
            n_clusters=self.num_centroids,
            random_state=CONFIG['random_state']
        )
        kmeans.fit(X_train)
        self.centroids = tf.Variable(
            initial_value=kmeans.cluster_centers_.astype(np.float32),
            trainable=False,  # Centroids remain fixed
            name='centroids',
            dtype=tf.float32
        )
        super(StandardRBFLayer, self).build(input_shape)

    def call(self, inputs):
        # Fixed Gaussian RBF activations
        diff = tf.expand_dims(inputs, 1) - self.centroids
        l2 = tf.reduce_sum(diff**2, axis=2)
        return tf.exp(-self.gamma * l2)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_centroids)

# Build Standard RBFN Model
def build_standard_rbfn(input_shape, num_classes):
    model = models.Sequential([
        StandardRBFLayer(
            num_centroids=CONFIG['num_centroids'],
            gamma=CONFIG['gamma'],
            input_shape=(input_shape,)
        ),
        layers.Dense(
            num_classes, 
            activation='softmax',
            kernel_initializer='glorot_uniform'
        )
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Function to calculate class-wise metrics
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

# Training with K-Fold CV
kf = KFold(n_splits=10, shuffle=True, random_state=CONFIG['random_state'])

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

# Create a DataFrame to store fold-wise validation metrics
fold_validation_df = pd.DataFrame(columns=[
    'Fold', 'Average_Validation_Accuracy', 'Max_Validation_Accuracy',
    'Final_Validation_Accuracy', 'Epochs'
])

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"\n=== Fold {fold+1}/10 ===")
    
    # Data split
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    y_tr_cat = tf.keras.utils.to_categorical(y_tr, num_classes=num_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)
    
    # Build and train model
    model = build_standard_rbfn(X_train.shape[1], num_classes)
    
    start_time = time.time()
    history = model.fit(
        X_tr, y_tr_cat,
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        validation_data=(X_val, y_val_cat),
        verbose=1,
        callbacks=[callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            min_delta=0.005,
            mode='max',
            restore_best_weights=True
        )]
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
    
    # Save full history
    all_histories.append({
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    })
    
    # Validation metrics
    y_val_pred = model.predict(X_val, verbose=0)
    y_val_pred_classes = np.argmax(y_val_pred, axis=1)
    fold_metrics = calculate_metrics(y_val, y_val_pred_classes)
    for metric in fold_metrics:
        if metric in metrics:
            metrics[metric].append(fold_metrics[metric])
    
    # Test metrics
    y_test_pred = model.predict(X_test, verbose=0)
    y_test_pred_classes = np.argmax(y_test_pred, axis=1)
    test_fold_metrics = calculate_metrics(y_test, y_test_pred_classes)
    for metric in test_fold_metrics:
        if metric in test_metrics:
            test_metrics[metric].append(test_fold_metrics[metric])
    
    # Confusion matrix
    conf_matrices.append(confusion_matrix(y_val, y_val_pred_classes))
    
    # Save model
    model.save(os.path.join(CONFIG['output_dir'], f"RBFN_std_fold{fold+1}.h5"))

# Save fold validation metrics to CSV
fold_validation_df.to_csv(os.path.join(CONFIG['output_dir'], 'Fold_Validation_Metrics.csv'), index=False)

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
np.save(os.path.join(CONFIG['output_dir'], 'avg_train_acc.npy'), avg_train_acc)
np.save(os.path.join(CONFIG['output_dir'], 'avg_val_acc.npy'), avg_val_acc)
np.save(os.path.join(CONFIG['output_dir'], 'avg_train_loss.npy'), avg_train_loss)
np.save(os.path.join(CONFIG['output_dir'], 'avg_val_loss.npy'), avg_val_loss)

# Save raw histories
np.save(os.path.join(CONFIG['output_dir'], 'all_histories.npy'), all_histories)

# ================== Plot Accuracy/Loss Curves ================== #
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(avg_train_acc, label='Training Accuracy', linewidth=2, color='blue')
plt.plot(avg_val_acc, label='Validation Accuracy', linewidth=2, color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('RBFN Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(avg_train_loss, label='Training Loss', linewidth=2, color='red')
plt.plot(avg_val_loss, label='Validation Loss', linewidth=2, color='purple')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('RBFN Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(CONFIG['output_dir'], 'Training_History.png'))
plt.close()

# ================== Save Results ================== #
avg_conf_matrix = np.mean(conf_matrices, axis=0).astype(int)
np.save(os.path.join(CONFIG['output_dir'], 'Confusion_Matrix_Avg.npy'), avg_conf_matrix)

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
results.to_csv(os.path.join(CONFIG['output_dir'], 'RBFN_Std_Metrics.csv'), index=False)

# Save detailed metrics per fold
detailed_metrics = []
for fold in range(10):
    fold_data = {'Fold': fold + 1}
    for metric in metrics:
        if metric != 'training_time':
            fold_data[metric] = metrics[metric][fold]
    detailed_metrics.append(fold_data)

detailed_df = pd.DataFrame(detailed_metrics)
detailed_df.to_csv(os.path.join(CONFIG['output_dir'], 'RBFN_Std_Detailed_Metrics.csv'), index=False)

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
plt.title('RBFN Average Confusion Matrix')
plt.savefig(os.path.join(CONFIG['output_dir'], 'Confusion_Matrix.png'))
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

print("\nStandard RBFN Training Completed")
print("Results saved to:", CONFIG['output_dir'])