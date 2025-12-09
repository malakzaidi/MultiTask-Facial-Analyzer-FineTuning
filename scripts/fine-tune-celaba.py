# ============================================================================
# FINE-TUNING MODÈLE MULTI-TÂCHES CELEBA - TENSORFLOW/KERAS
# ============================================================================

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

print("=" * 80)
print("🔧 FINE-TUNING DU MODÈLE MULTI-TÂCHES CELEBA")
print("=" * 80)

# Configuration GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🖥️  GPU détecté : {gpus[0].name}")
    except RuntimeError as e:
        print(f"⚠️  Erreur GPU : {e}")
else:
    print("⚠️  Aucun GPU détecté")

print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

IMG_DIR = Path("C:/Users/user/datasets/celeba/img_align_celeba/img_align_celeba")
ATTR_FILE = Path("C:/Users/user/datasets/celeba/list_attr_celeba.csv")
MODEL_PATH = "best_model_a500_tf.keras"  # Modèle pré-entraîné

# Hyperparamètres pour fine-tuning
IMG_SIZE = 128
BATCH_SIZE = 16  # Réduit pour fine-tuning
NUM_EPOCHS_FINETUNE = 5  # Moins d'époques
LEARNING_RATE_FINETUNE = 1e-5  # LR très faible pour fine-tuning

SELECTED_ATTRS = ['Smiling', 'Eyeglasses', 'Blond_Hair', 'Young', 'Male']
ATTR_NAMES_FR = ['Souriant', 'Lunettes', 'Cheveux_blonds', 'Jeune', 'Homme']

print(f"\n⚙️  Configuration Fine-tuning :")
print(f"   Batch size     : {BATCH_SIZE}")
print(f"   Époques        : {NUM_EPOCHS_FINETUNE}")
print(f"   Learning rate  : {LEARNING_RATE_FINETUNE}")


# ============================================================================
# GÉNÉRATEUR DE DONNÉES (RÉUTILISATION)
# ============================================================================

class CelebADataGenerator(keras.utils.Sequence):
    """Générateur de données optimisé pour CelebA"""

    def __init__(self, dataframe, img_dir, batch_size=32, img_size=128,
                 attrs=SELECTED_ATTRS, shuffle=True, augment=False):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.batch_size = batch_size
        self.img_size = img_size
        self.attrs = attrs
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

        if augment:
            self.aug_layer = keras.Sequential([
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.05),
                layers.RandomContrast(0.1),
            ])

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = []
        batch_labels = []

        for idx in batch_indexes:
            img_name = self.df.loc[idx, 'image_id']
            img_path = self.img_dir / img_name

            try:
                img = load_img(img_path, target_size=(self.img_size, self.img_size))
                img_array = img_to_array(img) / 255.0
                img_array = (img_array - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

                batch_images.append(img_array)
                labels = [self.df.loc[idx, attr] for attr in self.attrs]
                batch_labels.append(labels)
            except:
                continue

        X = np.array(batch_images, dtype=np.float32)
        y = np.array(batch_labels, dtype=np.float32)

        if self.augment:
            X = self.aug_layer(X, training=True)

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


# ============================================================================
# MÉTRIQUES
# ============================================================================

def calculate_metrics(y_true, y_pred, threshold=0.5):
    """Calcul des métriques pour chaque tâche"""
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


class MetricsCallback(callbacks.Callback):
    """Callback pour métriques détaillées"""

    def __init__(self, val_generator):
        super().__init__()
        self.val_generator = val_generator
        self.history_metrics = []

    def on_epoch_end(self, epoch, logs=None):
        y_true_list, y_pred_list = [], []

        for i in range(len(self.val_generator)):
            X_batch, y_batch = self.val_generator[i]
            pred_batch = self.model.predict(X_batch, verbose=0)
            pred_batch = np.hstack([p for p in pred_batch])
            y_true_list.append(y_batch)
            y_pred_list.append(pred_batch)

        y_true = np.vstack(y_true_list)
        y_pred = np.vstack(y_pred_list)

        metrics = calculate_metrics(y_true, y_pred)
        self.history_metrics.append(metrics)

        avg_acc = np.mean([m['accuracy'] for m in metrics.values()])
        avg_f1 = np.mean([m['f1'] for m in metrics.values()])

        print(f"\n   📊 Val Metrics - Acc: {avg_acc:.4f} | F1: {avg_f1:.4f}")


# ============================================================================
# FONCTION PRINCIPALE DE FINE-TUNING
# ============================================================================

def finetune_model():
    print("\n" + "=" * 80)
    print("🚀 DÉBUT DU FINE-TUNING")
    print("=" * 80)

    # ------------------------------------------------------------------ CHARGEMENT MODÈLE
    print(f"\n📥 Chargement du modèle pré-entraîné : {MODEL_PATH}")

    if not Path(MODEL_PATH).exists():
        print(f"❌ Erreur : Le modèle {MODEL_PATH} n'existe pas !")
        print("   Veuillez d'abord entraîner le modèle avec train_celeba_tf.py")
        return None

    model = keras.models.load_model(MODEL_PATH)
    print("✅ Modèle chargé avec succès")

    # ------------------------------------------------------------------ DÉGEL DES COUCHES
    print("\n🔓 Dégel des couches pour fine-tuning...")

    # Stratégie : dégeler progressivement les couches du backbone
    # On garde les premières couches gelées, on dégèle les dernières

    total_layers = len(model.layers)
    print(f"   Nombre total de couches : {total_layers}")

    # Trouver le backbone ResNet50
    backbone = None
    for layer in model.layers:
        if 'resnet50' in layer.name.lower():
            backbone = layer
            break

    if backbone:
        print(f"   Backbone trouvé : {backbone.name}")
        print(f"   Couches dans backbone : {len(backbone.layers)}")

        # Dégeler les 30 dernières couches du backbone (fine-tuning partiel)
        for layer in backbone.layers[:-30]:
            layer.trainable = False
        for layer in backbone.layers[-30:]:
            layer.trainable = True

        trainable_backbone = sum([1 for l in backbone.layers if l.trainable])
        print(f"   Couches dégelées dans backbone : {trainable_backbone}")
    else:
        print("   ⚠️  Backbone non trouvé, dégel de toutes les couches")
        model.trainable = True

    # Toujours garder les têtes de tâches entraînables
    for layer in model.layers:
        if 'task_' in layer.name or 'output_' in layer.name or 'shared_' in layer.name:
            layer.trainable = True

    # Compter paramètres entraînables
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    total_params = sum([tf.size(w).numpy() for w in model.weights])
    print(f"\n📊 Paramètres après dégel :")
    print(f"   Total             : {total_params:,}")
    print(f"   Entraînables      : {trainable_params:,}")
    print(f"   Pourcentage       : {100 * trainable_params / total_params:.1f}%")

    # ------------------------------------------------------------------ DONNÉES
    print("\n📂 Chargement des données...")
    attr_df = pd.read_csv(ATTR_FILE)

    for attr in SELECTED_ATTRS:
        attr_df[attr] = (attr_df[attr] + 1) // 2

    train_df, temp_df = train_test_split(attr_df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"   Train : {len(train_df):,} images")
    print(f"   Val   : {len(val_df):,} images")
    print(f"   Test  : {len(test_df):,} images")

    # ------------------------------------------------------------------ GÉNÉRATEURS
    print("\n📦 Création des générateurs...")
    train_gen = CelebADataGenerator(
        train_df, IMG_DIR, BATCH_SIZE, IMG_SIZE,
        shuffle=True, augment=True
    )
    val_gen = CelebADataGenerator(
        val_df, IMG_DIR, BATCH_SIZE, IMG_SIZE,
        shuffle=False, augment=False
    )
    test_gen = CelebADataGenerator(
        test_df, IMG_DIR, BATCH_SIZE, IMG_SIZE,
        shuffle=False, augment=False
    )

    # ------------------------------------------------------------------ RECOMPILATION
    print("\n⚙️  Recompilation avec LR réduit pour fine-tuning...")

    # Learning rate très faible pour fine-tuning
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE_FINETUNE)

    model.compile(
        optimizer=optimizer,
        loss=['binary_crossentropy'] * len(SELECTED_ATTRS),
        metrics=['accuracy']
    )

    # ------------------------------------------------------------------ CALLBACKS
    callbacks_list = [
        callbacks.ModelCheckpoint(
            'finetuned_model_a500_tf.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
        MetricsCallback(val_gen)
    ]

    # ------------------------------------------------------------------ FINE-TUNING
    print("\n🎓 Début du fine-tuning...")
    print(f"   Learning rate : {LEARNING_RATE_FINETUNE}")
    print(f"   Époques       : {NUM_EPOCHS_FINETUNE}")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=NUM_EPOCHS_FINETUNE,
        callbacks=callbacks_list,
        verbose=1
    )

    # ------------------------------------------------------------------ ÉVALUATION
    print("\n📊 Évaluation finale sur test...")

    y_true_list, y_pred_list = [], []

    for i in tqdm(range(len(test_gen)), desc="Évaluation"):
        X_batch, y_batch = test_gen[i]
        pred_batch = model.predict(X_batch, verbose=0)
        pred_batch = np.hstack([p for p in pred_batch])
        y_true_list.append(y_batch)
        y_pred_list.append(pred_batch)

    y_true = np.vstack(y_true_list)
    y_pred = np.vstack(y_pred_list)

    test_metrics = calculate_metrics(y_true, y_pred)

    print(f"\n🎯 RÉSULTATS APRÈS FINE-TUNING :")
    print("-" * 70)
    for name, metrics in test_metrics.items():
        print(f"{name:15} | Acc: {metrics['accuracy']:.3f} | "
              f"F1: {metrics['f1']:.3f} | AUC: {metrics['auc']:.3f}")
    print("-" * 70)

    avg_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in test_metrics.values()]),
        'f1': np.mean([m['f1'] for m in test_metrics.values()]),
        'auc': np.mean([m['auc'] for m in test_metrics.values()])
    }

    print(f"{'MOYENNE':15} | Acc: {avg_metrics['accuracy']:.3f} | "
          f"F1: {avg_metrics['f1']:.3f} | AUC: {avg_metrics['auc']:.3f}")

    # ------------------------------------------------------------------ VISUALISATION
    print("\n📈 Génération des graphiques...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_title('Évolution de la Loss (Fine-tuning)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Époque')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy moyenne
    train_accs = []
    val_accs = []
    for epoch in range(NUM_EPOCHS_FINETUNE):
        train_acc = np.mean([history.history[f'output_task_{i}_accuracy'][epoch]
                             for i in range(len(SELECTED_ATTRS))])
        val_acc = np.mean([history.history[f'val_output_task_{i}_accuracy'][epoch]
                           for i in range(len(SELECTED_ATTRS))])
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    axes[1].plot(train_accs, label='Train Acc', linewidth=2)
    axes[1].plot(val_accs, label='Val Acc', linewidth=2)
    axes[1].set_title('Accuracy Moyenne (Fine-tuning)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Époque')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('finetuning_history.png', dpi=150, bbox_inches='tight')
    print("   📊 Graphique sauvegardé : finetuning_history.png")

    print("\n✅ Fine-tuning terminé avec succès !")
    print(f"💾 Modèle sauvegardé : finetuned_model_a500_tf.keras")

    return model, history, test_metrics


if __name__ == "__main__":
    model, history, test_metrics = finetune_model()
    print("\n🎉 Script terminé !")