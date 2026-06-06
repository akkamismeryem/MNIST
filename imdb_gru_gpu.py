import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

tf.keras.mixed_precision.set_global_policy("mixed_float16")

# =========================
# AYARLAR
# =========================
MAX_TOKENS = 30000
SEQ_LEN = 300
BATCH_SIZE = 64
EPOCHS = 5
EMBED_DIM = 128
LSTM_UNITS = 128

OUTPUT_DIR = "imdb_lstm_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# DATASET
# =========================
dataset = load_dataset("stanfordnlp/imdb")

x_train = dataset["train"]["text"]
y_train = np.array(dataset["train"]["label"])

x_test = dataset["test"]["text"]
y_test = np.array(dataset["test"]["label"])

print("Train:", len(x_train))
print("Test:", len(x_test))

# =========================
# TEXT VECTORIZATION
# =========================
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode="int",
    output_sequence_length=SEQ_LEN
)

vectorizer.adapt(tf.data.Dataset.from_tensor_slices(x_train).batch(256))

def make_dataset(texts, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((texts, labels))

    if shuffle:
        ds = ds.shuffle(10000)

    ds = ds.batch(BATCH_SIZE)

    ds = ds.map(
        lambda text, label: (vectorizer(text), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(x_train, y_train, shuffle=True)
test_ds = make_dataset(x_test, y_test, shuffle=False)

# =========================
# MODEL
# =========================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(SEQ_LEN,)),
    tf.keras.layers.Embedding(MAX_TOKENS, EMBED_DIM),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_UNITS)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# TRAIN
# =========================
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    verbose=1
)

# =========================
# TEST
# =========================
loss, acc = model.evaluate(test_ds, verbose=1)
print(f"\nLSTM Accuracy: {acc:.4f}")

y_prob = model.predict(test_ds, verbose=1)
y_pred = (y_prob >= 0.5).astype(int).reshape(-1)

report = classification_report(
    y_test,
    y_pred,
    target_names=["negative", "positive"]
)

print("\nClassification Report:")
print(report)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# =========================
# SAVE RESULTS
# =========================
with open(os.path.join(OUTPUT_DIR, "lstm_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"LSTM Accuracy: {acc:.4f}\n\n")
    f.write(report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(cm))

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["negative", "positive"],
            yticklabels=["negative", "positive"])
plt.title("LSTM Confusion Matrix")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "lstm_confusion_matrix.png"), dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("LSTM Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "lstm_accuracy.png"), dpi=200)
plt.close()

print("\nLSTM eğitimi tamamlandı.")
print("Çıktı klasörü:", OUTPUT_DIR)