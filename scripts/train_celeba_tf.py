import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
import threading
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime



warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# Add after other imports
import contextlib
import sys

@contextlib.contextmanager
def suppress_tf_warnings():
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stderr.close()
        sys.stderr = old_stderr

# ============================================================================
# GPU MONITORING THREAD
# ============================================================================
class GPUMonitor:
    def __init__(self):
        self.monitoring = False
        self.thread = None

    def start(self):
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self.monitoring = False
        if self.thread:
            self.thread.join()

    def _monitor(self):
        try:
            import GPUtil
            has_gputil = True
        except ImportError:
            has_gputil = False

        while self.monitoring:
            if has_gputil:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    print(
                        f"\r🖥️  GPU Usage: {gpu.load * 100:.1f}% | Memory: {gpu.memoryUsed:.0f}/{gpu.memoryTotal:.0f}MB ({gpu.memoryUtil * 100:.1f}%)",
                        end='', flush=True)
            else:
                # Fallback: Check if tensors are on GPU
                if tf.config.list_physical_devices('GPU'):
                    print(f"\r🖥️  GPU Active - Training in progress...", end='', flush=True)
            time.sleep(2)


# ============================================================================
# GPU VERIFICATION
# ============================================================================
print("=" * 80)
print("GPU CONFIGURATION CHECK")
print("=" * 80)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU DETECTED: {gpus[0].name}")
        print(f"   TensorFlow version: {tf.__version__}")
        print(f"   Number of GPUs: {len(gpus)}")

        # Test GPU with actual computation
        with tf.device('/GPU:0'):
            test_tensor = tf.random.normal([1000, 1000])
            result = tf.matmul(test_tensor, test_tensor)
        print(f"   ✅ GPU computation test: PASSED")
        print(f"   GPU will be used for training: YES")
    except RuntimeError as e:
        print(f"❌ GPU configuration error: {e}")
else:
    print("❌ NO GPU DETECTED - Training will use CPU (SLOW)")

print("=" * 80)
print("\n💡 TIP: Install GPUtil for detailed GPU monitoring: pip install gputil")
print("=" * 80)

# ============================================================================
# CONFIGURATION - MAXIMUM SPEED OPTIMIZATION
# ============================================================================
IMG_DIR = Path("C:/Users/user/datasets/celeba/img_align_celeba/img_align_celeba")
ATTR_FILE = Path("C:/Users/user/datasets/celeba/list_attr_celeba.csv")

# Aggressive speed optimizations for RTX A500
IMG_SIZE = 96  # Reduced from 128 (30% faster)
BATCH_SIZE = 128  # Increased from 64 (better GPU saturation)
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
PREFETCH_SIZE = tf.data.AUTOTUNE
NUM_PARALLEL_CALLS = tf.data.AUTOTUNE

# Use subset for faster initial training (set to 1.0 for full dataset)
USE_SUBSET = True
SUBSET_FRACTION = 0.3  # Use 30% of data (60k images instead of 200k)

# Use lightweight model
USE_MOBILENET = True  # MobileNetV2 is 3x faster than ResNet50

SELECTED_ATTRS = ['Smiling', 'Eyeglasses', 'Blond_Hair', 'Young', 'Male']
ATTR_NAMES_FR = ['Souriant', 'Lunettes', 'Cheveux_blonds', 'Jeune', 'Homme']

print(f"\n⚡ SPEED OPTIMIZATIONS ENABLED:")
print(f"   - Image size: {IMG_SIZE}x{IMG_SIZE} (smaller = faster)")
print(f"   - Batch size: {BATCH_SIZE} (larger = better GPU use)")
print(f"   - Model: {'MobileNetV2 (lightweight)' if USE_MOBILENET else 'ResNet50'}")
print(f"   - Dataset: {SUBSET_FRACTION * 100:.0f}% of full data" if USE_SUBSET else "   - Dataset: 100% (full)")
print()


# ============================================================================
# EFFICIENT TF.DATA PIPELINE (PYTORCH-LIKE SPEED)
# ============================================================================

def parse_image(img_path, labels):
    """Parse and preprocess a single image - OPTIMIZED"""
    # Read image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)

    # Resize with faster method
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method='bilinear')
    img = tf.cast(img, tf.float32) / 255.0

    # ImageNet normalization (vectorized)
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    return img, labels


def augment_image(img, labels):
    """Lightweight data augmentation"""
    # Only horizontal flip for speed
    img = tf.image.random_flip_left_right(img)
    return img, labels


def create_tf_dataset(df, img_dir, batch_size, shuffle=True, augment=False, cache=False):
    """Create MAXIMUM SPEED optimized tf.data.Dataset"""

    # Prepare file paths and labels
    file_paths = [str(img_dir / fname) for fname in df['image_id'].values]
    labels = df[SELECTED_ATTRS].values.astype(np.float32)

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    # Optimized options
    options = tf.data.Options()
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_fusion = True
    options.threading.private_threadpool_size = 8
    options.threading.max_intra_op_parallelism = 1
    dataset = dataset.with_options(options)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(10000, len(df)), reshuffle_each_iteration=True)

    # Parse images in parallel
    dataset = dataset.map(parse_image, num_parallel_calls=NUM_PARALLEL_CALLS)

    # Cache if requested (validation/test sets)
    if cache:
        dataset = dataset.cache()

    # Apply augmentation AFTER caching
    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=NUM_PARALLEL_CALLS)

    # Batch with drop_remainder for consistent sizes
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Prefetch to GPU
    dataset = dataset.prefetch(buffer_size=PREFETCH_SIZE)

    return dataset


# ============================================================================
# LIGHTWEIGHT MODEL (MOBILENETV2 OPTION)
# ============================================================================
def create_multitask_model(num_tasks=5, img_size=96, dropout_shared=0.3, dropout_heads=0.2):
    """
    Multi-task model with MobileNetV2 or ResNet50 backbone
    MobileNetV2 is 3x faster with minimal accuracy loss
    """
    inputs = layers.Input(shape=(img_size, img_size, 3), name='image_input')

    # Choose backbone based on configuration
    if USE_MOBILENET:
        from tensorflow.keras.applications import MobileNetV2
        backbone = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs,
            pooling='avg',
            alpha=1.0  # Width multiplier
        )
        feature_size = 1280
    else:
        backbone = ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs,
            pooling='avg'
        )
        feature_size = 2048

    # Freeze backbone for transfer learning
    backbone.trainable = False
    x = backbone.output  # Shape: (None, feature_size)

    # Lighter shared layers for speed
    x = layers.Dense(128, name='shared_dense')(x)  # Reduced from 256
    x = layers.BatchNormalization(name='shared_bn')(x)
    x = layers.ReLU(name='shared_relu')(x)
    shared_features = layers.Dropout(dropout_shared, name='shared_dropout')(x)

    # Task-specific heads (lighter)
    task_outputs = []
    for i in range(num_tasks):
        x = layers.Dense(32, name=f'task_{i}_dense')(shared_features)  # Reduced from 64
        x = layers.BatchNormalization(name=f'task_{i}_bn')(x)
        x = layers.ReLU(name=f'task_{i}_relu')(x)
        x = layers.Dropout(dropout_heads, name=f'task_{i}_dropout')(x)
        output = layers.Dense(1, activation='sigmoid', name=f'output_task_{i}')(x)
        task_outputs.append(output)

    model_name = 'MultiTask_MobileNetV2' if USE_MOBILENET else 'MultiTask_ResNet50'
    model = models.Model(inputs=inputs, outputs=task_outputs, name=model_name)

    return model


# ============================================================================
# METRICS
# ============================================================================
def calculate_metrics(y_true, y_pred, threshold=0.5):
    num_tasks = y_true.shape[1]
    metrics = {}

    for task_idx in range(num_tasks):
        pred = (y_pred[:, task_idx] >= threshold).astype(int)
        true = y_true[:, task_idx].astype(int)

        acc = accuracy_score(true, pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true, pred, average='binary', zero_division=0
        )

        try:
            auc_score = roc_auc_score(true, y_pred[:, task_idx])
        except:
            auc_score = 0.0

        metrics[ATTR_NAMES_FR[task_idx]] = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_score
        }

    return metrics


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def create_visualizations(history, test_metrics, save_dir='visualizations'):
    """Create and save all visualizations"""

    # Create directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    viz_dir = Path(save_dir) / f'run_{timestamp}'
    viz_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📊 Creating visualizations in: {viz_dir}")

    # 1. Training Loss Curves
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training and Validation Loss per Task', fontsize=16, fontweight='bold')

    for i in range(5):
        ax = axes[i // 3, i % 3]
        ax.plot(history.history[f'output_task_{i}_loss'], label='Train', linewidth=2)
        ax.plot(history.history[f'val_output_task_{i}_loss'], label='Val', linewidth=2)
        ax.set_title(f'{ATTR_NAMES_FR[i]}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[1, 2].axis('off')
    plt.tight_layout()
    plt.savefig(viz_dir / '01_loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Training Accuracy Curves
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training and Validation Accuracy per Task', fontsize=16, fontweight='bold')

    for i in range(5):
        ax = axes[i // 3, i % 3]
        ax.plot(history.history[f'output_task_{i}_accuracy'], label='Train', linewidth=2)
        ax.plot(history.history[f'val_output_task_{i}_accuracy'], label='Val', linewidth=2)
        ax.set_title(f'{ATTR_NAMES_FR[i]}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    axes[1, 2].axis('off')
    plt.tight_layout()
    plt.savefig(viz_dir / '02_accuracy_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Test Metrics Bar Chart
    metrics_df = pd.DataFrame(test_metrics).T

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Test Set Performance Metrics', fontsize=16, fontweight='bold')

    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        ax = axes[idx // 2, idx % 2]
        bars = ax.bar(range(len(ATTR_NAMES_FR)), metrics_df[metric],
                      color=sns.color_palette("husl", 5), alpha=0.8, edgecolor='black')
        ax.set_title(f'{title} by Task', fontsize=12, fontweight='bold')
        ax.set_xlabel('Task')
        ax.set_ylabel(title)
        ax.set_xticks(range(len(ATTR_NAMES_FR)))
        ax.set_xticklabels(ATTR_NAMES_FR, rotation=45, ha='right')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(viz_dir / '03_test_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Metrics Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(metrics_df[['accuracy', 'precision', 'recall', 'f1', 'auc']],
                annot=True, fmt='.3f', cmap='YlGnBu', cbar_kws={'label': 'Score'},
                linewidths=1, linecolor='gray', ax=ax)
    ax.set_title('Test Metrics Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Task', fontsize=12)
    plt.tight_layout()
    plt.savefig(viz_dir / '04_metrics_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 5. Overall Loss Evolution
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate total loss
    train_loss = np.sum([history.history[f'output_task_{i}_loss'] for i in range(5)], axis=0)
    val_loss = np.sum([history.history[f'val_output_task_{i}_loss'] for i in range(5)], axis=0)

    ax.plot(train_loss, label='Train Total Loss', linewidth=2.5, marker='o')
    ax.plot(val_loss, label='Val Total Loss', linewidth=2.5, marker='s')
    ax.set_title('Total Loss Evolution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / '05_total_loss.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 6. Average Metrics Comparison
    avg_metrics = {
        'Accuracy': np.mean(metrics_df['accuracy']),
        'Precision': np.mean(metrics_df['precision']),
        'Recall': np.mean(metrics_df['recall']),
        'F1-Score': np.mean(metrics_df['f1']),
        'AUC': np.mean(metrics_df['auc'])
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(avg_metrics.keys(), avg_metrics.values(),
                  color=sns.color_palette("Set2", 5), alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_title('Average Performance Across All Tasks', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(viz_dir / '06_average_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 7. Save metrics to CSV
    metrics_df.to_csv(viz_dir / 'test_metrics.csv')

    # 8. Save training history to CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(viz_dir / 'training_history.csv', index_label='epoch')

    # 9. Create summary text file
    with open(viz_dir / 'summary.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Image Size: {IMG_SIZE}x{IMG_SIZE}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Epochs Completed: {len(history.history['loss'])}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n\n")

        f.write("=" * 80 + "\n")
        f.write("TEST METRICS BY TASK\n")
        f.write("=" * 80 + "\n\n")

        for name, metrics in test_metrics.items():
            f.write(f"{name}:\n")
            f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score:  {metrics['f1']:.4f}\n")
            f.write(f"  AUC:       {metrics['auc']:.4f}\n\n")

        f.write("=" * 80 + "\n")
        f.write("AVERAGE METRICS\n")
        f.write("=" * 80 + "\n\n")
        for metric_name, value in avg_metrics.items():
            f.write(f"{metric_name}: {value:.4f}\n")

    print(f"✅ All visualizations saved to: {viz_dir}")
    print(f"   - 6 plots (.png)")
    print(f"   - 2 CSV files (metrics + history)")
    print(f"   - 1 summary file (.txt)")

    return viz_dir


class MetricsCallback(callbacks.Callback):
    def __init__(self, val_dataset, val_steps):
        super().__init__()
        self.val_dataset = val_dataset
        self.val_steps = val_steps
        self.history_metrics = []

    def on_epoch_end(self, epoch, logs=None):
        y_true_list, y_pred_list = [], []

        for X_batch, y_batch in self.val_dataset.take(self.val_steps):
            pred_batch = self.model.predict(X_batch, verbose=0)
            pred_batch = np.hstack([p.numpy() for p in pred_batch])
            y_true_list.append(y_batch.numpy())
            y_pred_list.append(pred_batch)

        y_true = np.vstack(y_true_list)
        y_pred = np.vstack(y_pred_list)

        metrics = calculate_metrics(y_true, y_pred)
        self.history_metrics.append(metrics)

        avg_acc = np.mean([m['accuracy'] for m in metrics.values()])
        avg_f1 = np.mean([m['f1'] for m in metrics.values()])

        print(f"\n   📊 Val Metrics - Acc: {avg_acc:.4f} | F1: {avg_f1:.4f}")


class GPUMonitorCallback(callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.gpu_monitor = GPUMonitor()

    def on_train_begin(self, logs=None):
        print("\n🚀 Starting GPU monitoring during training...")
        self.gpu_monitor.start()

    def on_train_end(self, logs=None):
        self.gpu_monitor.stop()
        print("\n✅ GPU monitoring stopped")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 80)
    print("🚀 TRAINING PIPELINE WITH REAL-TIME GPU MONITORING")
    print("=" * 80)

    # Load data
    print("\n📂 Loading data...")
    attr_df = pd.read_csv(ATTR_FILE)

    for attr in SELECTED_ATTRS:
        attr_df[attr] = (attr_df[attr] + 1) // 2

    # Use subset for faster training
    if USE_SUBSET and SUBSET_FRACTION < 1.0:
        print(f"   ⚡ Using {SUBSET_FRACTION * 100:.0f}% of dataset for faster training")
        attr_df = attr_df.sample(frac=SUBSET_FRACTION, random_state=42).reset_index(drop=True)

    train_df, temp_df = train_test_split(attr_df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"   Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # Create optimized datasets
    print("\n📦 Creating optimized tf.data pipelines...")
    with suppress_tf_warnings():
        # Cache validation/test sets in RAM for speed
        train_dataset = create_tf_dataset(train_df, IMG_DIR, BATCH_SIZE, shuffle=True, augment=True, cache=False)
        val_dataset = create_tf_dataset(val_df, IMG_DIR, BATCH_SIZE, shuffle=False, augment=False, cache=True)
        test_dataset = create_tf_dataset(test_df, IMG_DIR, BATCH_SIZE, shuffle=False, augment=False, cache=True)

    # Calculate steps
    train_steps = len(train_df) // BATCH_SIZE
    val_steps = len(val_df) // BATCH_SIZE
    test_steps = len(test_df) // BATCH_SIZE

    print(f"   Train steps: {train_steps} | Val steps: {val_steps}")
    print(f"   🚀 Optimized pipeline with prefetching + caching enabled")

    # Estimate time
    estimated_time_per_epoch = (train_steps * 0.25) / 60  # Optimistic estimate
    print(f"   ⏱️  Estimated time per epoch: ~{estimated_time_per_epoch:.1f} minutes")

    # Create model
    print("\n🏗️  Creating model...")
    model = create_multitask_model(num_tasks=len(SELECTED_ATTRS), img_size=IMG_SIZE)

    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    total_params = sum([tf.size(w).numpy() for w in model.weights])
    print(f"   Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # Compile
    print("\n⚙️  Compiling model...")

    # Use SGD with momentum for faster convergence on small batches
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=1e-7)

    model.compile(
        optimizer=optimizer,
        loss=['binary_crossentropy'] * len(SELECTED_ATTRS),
        metrics=['accuracy'],
        run_eagerly=False  # Use graph mode for speed
    )

    # Callbacks with GPU monitoring
    callbacks_list = [
        GPUMonitorCallback(),
        callbacks.ModelCheckpoint(
            'best_model_a500_tf.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        MetricsCallback(val_dataset, val_steps)
    ]

    # Training
    print("\n🎓 Starting training with GPU monitoring...\n")

    with suppress_tf_warnings():
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=NUM_EPOCHS,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            callbacks=callbacks_list,
            verbose=1
        )

    # Evaluation
    print("\n\n📊 Evaluating on test set...")

    y_true_list, y_pred_list = [], []

    for X_batch, y_batch in tqdm(test_dataset.take(test_steps), total=test_steps, desc="Testing"):
        pred_batch = model.predict(X_batch, verbose=0)
        pred_batch = np.hstack([p.numpy() for p in pred_batch])
        y_true_list.append(y_batch.numpy())
        y_pred_list.append(pred_batch)

    y_true = np.vstack(y_true_list)
    y_pred = np.vstack(y_pred_list)

    test_metrics = calculate_metrics(y_true, y_pred)

    print(f"\n🎯 TEST RESULTS:")
    print("-" * 70)
    for name, metrics in test_metrics.items():
        print(f"{name:15} | Acc: {metrics['accuracy']:.3f} | F1: {metrics['f1']:.3f} | AUC: {metrics['auc']:.3f}")
    print("-" * 70)

    avg_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in test_metrics.values()]),
        'f1': np.mean([m['f1'] for m in test_metrics.values()]),
        'auc': np.mean([m['auc'] for m in test_metrics.values()])
    }

    print(
        f"{'AVERAGE':15} | Acc: {avg_metrics['accuracy']:.3f} | F1: {avg_metrics['f1']:.3f} | AUC: {avg_metrics['auc']:.3f}")
    print("\n✅ Training complete!")

    return model, history, test_metrics


if __name__ == "__main__":
    model, history, test_metrics, viz_dir = main()
    print("\n💾 Model saved: best_model_a500_tf.keras")
    print(f"📊 Visualizations saved: {viz_dir}")
    print("🎉 Script completed successfully!")

