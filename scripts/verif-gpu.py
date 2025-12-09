# ============================================================================
# VÉRIFICATION GPU
# ============================================================================

print("=" * 80)
print("✅ CONFIGURATION TENSORFLOW POUR RTX A500 4GB")
print("=" * 80)

# Configuration GPU avec gestion mémoire optimale
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🖥️  GPU détecté : {gpus[0].name}")
        print(f"   TensorFlow version : {tf.__version__}")
        print(f"   Nombre de GPUs : {len(gpus)}")

        # Vérification additionnelle
        print(f"   Device utilisé : {tf.test.gpu_device_name()}")

        # Test rapide pour confirmer l'utilisation GPU
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print(f"   Test GPU réussi : {c.device}")

    except RuntimeError as e:
        print(f"⚠️  Erreur configuration GPU : {e}")
else:
    print("⚠️  Aucun GPU détecté, utilisation du CPU")