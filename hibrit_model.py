import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, gc

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =====================
# AYARLAR
# =====================
EPOCHS = 16
BATCH_SIZE = 8
FEATURE_DIM = 512
IMG_SIZE = 224

# =====================
# DATA
# =====================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train[:10000]
y_train = y_train[:10000]
x_test = x_test[:2000]
y_test = y_test[:2000]

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

def preprocess(images):
    images = tf.image.resize(images[..., np.newaxis], (IMG_SIZE, IMG_SIZE))
    images = tf.image.grayscale_to_rgb(images)
    return images.numpy()

x_train = preprocess(x_train)
x_test = preprocess(x_test)

# =====================
# GOOGLENET BLOĞU
# =====================
def inception_module(x, f1, f3_in, f3_out, f5_in, f5_out, pool_proj):
    p1 = tf.keras.layers.Conv2D(f1, 1, padding="same", activation="relu")(x)

    p2 = tf.keras.layers.Conv2D(f3_in, 1, padding="same", activation="relu")(x)
    p2 = tf.keras.layers.Conv2D(f3_out, 3, padding="same", activation="relu")(p2)

    p3 = tf.keras.layers.Conv2D(f5_in, 1, padding="same", activation="relu")(x)
    p3 = tf.keras.layers.Conv2D(f5_out, 5, padding="same", activation="relu")(p3)

    p4 = tf.keras.layers.MaxPooling2D(3, strides=1, padding="same")(x)
    p4 = tf.keras.layers.Conv2D(pool_proj, 1, padding="same", activation="relu")(p4)

    return tf.keras.layers.Concatenate()([p1, p2, p3, p4])

# =====================
# MODEL OLUŞTURMA
# =====================
def build_model(model_name):
    inp = tf.keras.layers.Input(shape=(224,224,3))

    if model_name == "AlexNet":
        x = tf.keras.layers.Conv2D(96, 11, strides=4, activation="relu")(inp)
        x = tf.keras.layers.MaxPooling2D(3, strides=2)(x)
        x = tf.keras.layers.Conv2D(256, 5, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(3, strides=2)(x)
        x = tf.keras.layers.Conv2D(384, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv2D(384, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(3, strides=2)(x)
        x = tf.keras.layers.Flatten()(x)

    elif model_name == "GoogLeNet":
        x = tf.keras.layers.Conv2D(64, 7, strides=2, padding="same", activation="relu")(inp)
        x = tf.keras.layers.MaxPooling2D(3, strides=2)(x)
        x = tf.keras.layers.Conv2D(64, 1, activation="relu")(x)
        x = tf.keras.layers.Conv2D(192, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(3, strides=2)(x)
        x = inception_module(x, 64, 96, 128, 16, 32, 32)
        x = inception_module(x, 128, 128, 192, 32, 96, 64)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

    elif model_name == "ResNet50":
        base = tf.keras.applications.ResNet50(weights=None, include_top=False, input_tensor=inp)
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)

    elif model_name == "ResNet101":
        base = tf.keras.applications.ResNet101(weights=None, include_top=False, input_tensor=inp)
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)

    elif model_name == "EfficientNet":
        base = tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_tensor=inp)
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)

    feature = tf.keras.layers.Dense(FEATURE_DIM, activation="relu", name="feature_layer")(x)
    out = tf.keras.layers.Dense(10, activation="softmax")(feature)

    model = tf.keras.Model(inp, out)
    feature_model = tf.keras.Model(inp, feature)

    return model, feature_model

# =====================
# FEATURE ÇIKARMA
# =====================
model_names = ["AlexNet", "GoogLeNet", "ResNet50", "ResNet101", "EfficientNet"]

train_features_list = []
test_features_list = []

for name in model_names:
    print(f"\n{name} eğitiliyor...\n")

    model, feature_model = build_model(name)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        verbose=1
    )

    train_feat = feature_model.predict(x_train, verbose=1)
    test_feat = feature_model.predict(x_test, verbose=1)

    train_features_list.append(train_feat)
    test_features_list.append(test_feat)

    del model
    del feature_model
    gc.collect()
    tf.keras.backend.clear_session()

# =====================
# ALT ALTA STACKING
# =====================
X_train_stack = np.vstack(train_features_list)
X_test_stack = np.vstack(test_features_list)

y_train_stack = np.hstack([y_train] * len(model_names))
y_test_stack = np.hstack([y_test] * len(model_names))

print("Stack train shape:", X_train_stack.shape)
print("Stack test shape:", X_test_stack.shape)

# =====================
# SVM EĞİTİM
# =====================
svm = SVC(kernel="rbf", C=1.0, gamma="scale")

print("\nSVM eğitiliyor...\n")
svm.fit(X_train_stack, y_train_stack)

y_pred = svm.predict(X_test_stack)

acc = accuracy_score(y_test_stack, y_pred)
print("Feature Stacking + SVM Accuracy:", acc)

# =====================
# CONFUSION MATRIX
# =====================
cm = confusion_matrix(y_test_stack, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("Feature Stacking + SVM Confusion Matrix")
plt.tight_layout()
plt.savefig("feature_stacking_svm_cm.png")
plt.show()

# =====================
# RAPOR KAYIT
# =====================
report = classification_report(y_test_stack, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv("feature_stacking_svm_report.csv")

summary = pd.DataFrame([{
    "Model": "AlexNet+GoogLeNet+ResNet50+ResNet101+EfficientNet + SVM",
    "Fusion Type": "Vertical Feature Stacking",
    "Feature Dim Per Model": FEATURE_DIM,
    "Epoch": EPOCHS,
    "Batch Size": BATCH_SIZE,
    "Accuracy": acc,
    "Precision": report["weighted avg"]["precision"],
    "Recall": report["weighted avg"]["recall"],
    "F1-score": report["weighted avg"]["f1-score"]
}])

summary.to_csv("feature_stacking_svm_summary.csv", index=False)

print("\nTüm çıktılar kaydedildi.")