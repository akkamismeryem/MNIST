import os
import gc

# TensorFlow importundan ÖNCE bellek ayarı
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

# =========================
# 0) GPU AYARLARI
# =========================
print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")
print("Bulunan GPU:", gpus)

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth aktif.")
    except RuntimeError as e:
        print("GPU ayarı hatası:", e)
else:
    print("UYARI: GPU bulunamadı, CPU ile çalışacak.")

tf.keras.mixed_precision.set_global_policy("mixed_float16")
print("Precision policy:", tf.keras.mixed_precision.global_policy())

# =========================
# 1) DATA
# =========================
IMG_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 16
NUM_CLASSES = 10

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train[:10000].astype("float32") / 255.0
y_train = y_train[:10000]

x_test = x_test[:2000].astype("float32") / 255.0
y_test = y_test[:2000]

def preprocess(image, label):
    image = tf.expand_dims(image, axis=-1)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.image.grayscale_to_rgb(image)
    return image, label

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = (
    train_ds
    .shuffle(1000)
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = (
    test_ds
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

print("Train sample:", x_train.shape)
print("Test sample:", x_test.shape)

if os.path.exists("results.csv"):
    os.remove("results.csv")

# =========================
# 2) MODEL ÇALIŞTIRMA
# =========================
def run_model(model_name):
    tf.keras.backend.clear_session()
    gc.collect()

    if model_name == "ResNet50":
        base = tf.keras.applications.ResNet50(
            weights=None,
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )

    elif model_name == "ResNet101":
        base = tf.keras.applications.ResNet101(
            weights=None,
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )

    elif model_name == "EfficientNet":
        base = tf.keras.applications.EfficientNetB0(
            weights=None,
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )

    else:
        raise ValueError("Geçersiz model adı")

    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        dtype="float32"
    )(x)

    model = tf.keras.models.Model(inputs=base.input, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"\n{model_name} eğitim başlıyor...\n")

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=test_ds,
        verbose=1
    )

    loss, acc = model.evaluate(test_ds, verbose=1)
    print(f"{model_name} Accuracy: {acc:.4f}")

    y_pred = model.predict(test_ds, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_test, y_pred_classes)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.tight_layout()
    plt.savefig(f"{model_name}_cm.png", dpi=200)
    plt.close()

    df = pd.DataFrame([{
        "Model": model_name,
        "Accuracy": acc,
        "Loss": loss,
        "Epoch": EPOCHS,
        "BatchSize": BATCH_SIZE
    }])

    df.to_csv(
        "results.csv",
        mode="a",
        index=False,
        header=not os.path.exists("results.csv")
    )

    del model
    del base
    gc.collect()
    tf.keras.backend.clear_session()

    return acc, history

# =========================
# 3) MODELLERİ ÇALIŞTIR
# =========================
models = ["ResNet50", "ResNet101", "EfficientNet"]

for m in models:
    run_model(m)

# =========================
# 4) TEK GRAFİK
# =========================
df = pd.read_csv("results.csv")

plt.figure(figsize=(8, 5))
plt.bar(df["Model"], df["Accuracy"], color=["blue", "green", "orange"])
plt.title("Model Karşılaştırma")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("final_comparison.png", dpi=200)
plt.show()

# =========================
# 5) TÜM CONFUSION MATRIXLERİ
# =========================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, name in enumerate(models):
    img = plt.imread(f"{name}_cm.png")
    axes[i].imshow(img)
    axes[i].set_title(name)
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("all_confusion_matrices.png", dpi=200)
plt.show()

print("\nTÜM MODELLER TAMAMLANDI")