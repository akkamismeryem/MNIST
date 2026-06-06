import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# =========================================================
# 0) AYARLAR
# =========================================================

OUTPUT_DIR = "imdb_gru_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_WORDS = 20000
MAX_LEN = 250
EMBEDDING_DIM = 128
BATCH_SIZE = 64
EPOCHS = 5
VALIDATION_RATIO = 0.2
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# =========================================================
# 1) GPU KONTROL
# =========================================================

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


# =========================================================
# 2) VERİ SETİNİ YÜKLE
# =========================================================

print("\nIMDB veri seti yükleniyor...")

dataset = load_dataset("stanfordnlp/imdb")

train_data = dataset["train"]
test_data = dataset["test"]

train_texts = train_data["text"]
train_labels = np.array(train_data["label"])

test_texts = test_data["text"]
test_labels = np.array(test_data["label"])

print("Train örnek sayısı:", len(train_texts))
print("Test örnek sayısı:", len(test_texts))


# =========================================================
# 3) METİN TEMİZLEME
# =========================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


print("\nMetinler temizleniyor...")

train_texts_clean = [clean_text(text) for text in train_texts]
test_texts_clean = [clean_text(text) for text in test_texts]


# =========================================================
# 4) TOKENIZER VE PADDING
# =========================================================

print("\nTokenizer hazırlanıyor...")

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(train_texts_clean)

x_train_seq = tokenizer.texts_to_sequences(train_texts_clean)
x_test_seq = tokenizer.texts_to_sequences(test_texts_clean)

x_train = pad_sequences(
    x_train_seq,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

x_test = pad_sequences(
    x_test_seq,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

y_train = train_labels
y_test = test_labels

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)


# =========================================================
# 5) GRU MODELİ
# =========================================================

def build_gru_model():
    model = Sequential([
        Embedding(
            input_dim=MAX_WORDS,
            output_dim=EMBEDDING_DIM,
            input_length=MAX_LEN
        ),

        Bidirectional(
            GRU(
                units=64,
                return_sequences=False
            )
        ),

        Dropout(0.5),

        Dense(64, activation="relu"),
        Dropout(0.3),

        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


model = build_gru_model()
model.summary()


# =========================================================
# 6) MODEL EĞİTİMİ
# =========================================================

best_model_path = os.path.join(OUTPUT_DIR, "gru_best_model.keras")

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    ),

    ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_loss",
        save_best_only=True
    )
]

print("\nGRU modeli eğitiliyor...")

history = model.fit(
    x_train,
    y_train,
    validation_split=VALIDATION_RATIO,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)


# =========================================================
# 7) TEST DEĞERLENDİRME
# =========================================================

print("\nGRU modeli test ediliyor...")

y_prob = model.predict(x_test)
y_pred = (y_prob >= 0.5).astype(int).reshape(-1)

accuracy = accuracy_score(y_test, y_pred)

print("\nGRU Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["negative", "positive"]
))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# =========================================================
# 8) SONUÇLARI CSV OLARAK KAYDET
# =========================================================

report = classification_report(
    y_test,
    y_pred,
    target_names=["negative", "positive"],
    output_dict=True
)

results = {
    "model": "GRU",
    "accuracy": accuracy,
    "negative_precision": report["negative"]["precision"],
    "negative_recall": report["negative"]["recall"],
    "negative_f1": report["negative"]["f1-score"],
    "positive_precision": report["positive"]["precision"],
    "positive_recall": report["positive"]["recall"],
    "positive_f1": report["positive"]["f1-score"],
    "macro_f1": report["macro avg"]["f1-score"],
    "weighted_f1": report["weighted avg"]["f1-score"]
}

results_df = pd.DataFrame([results])
results_df.to_csv(
    os.path.join(OUTPUT_DIR, "gru_results.csv"),
    index=False
)


# =========================================================
# 9) ACCURACY GRAFİĞİ
# =========================================================

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("GRU Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gru_accuracy.png"), dpi=300)
plt.close()


# =========================================================
# 10) LOSS GRAFİĞİ
# =========================================================

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("GRU Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gru_loss.png"), dpi=300)
plt.close()


# =========================================================
# 11) CONFUSION MATRIX GRAFİĞİ
# =========================================================

plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("GRU Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks([0, 1], ["negative", "positive"])
plt.yticks([0, 1], ["negative", "positive"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gru_confusion_matrix.png"), dpi=300)
plt.close()


print("\nGRU eğitimi tamamlandı.")
print("Çıktı klasörü:", OUTPUT_DIR)