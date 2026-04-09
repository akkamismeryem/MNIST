import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))

    return x_train, y_train, x_test, y_test

def build_cnn_feature_extractor():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_cnn(model, x_train, y_train, epochs=1, batch_size=64):
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )
    return history

def extract_features(model, x_train, x_test):
    feature_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.layers[-2].output
    )

    train_features = feature_model.predict(x_train, verbose=1)
    test_features = feature_model.predict(x_test, verbose=1)

    return train_features, test_features

def train_svm(train_features, y_train, kernel="rbf", C=1.0, gamma="scale"):
    print("SVM eğitiliyor...")
    svm_model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
    svm_model.fit(train_features, y_train)
    return svm_model

def evaluate_hybrid_model(svm_model, test_features, y_test):
    y_pred = svm_model.predict(test_features)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nHybrid CNN+SVM Accuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("CNN + SVM Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig('output/cnn_svm_confusion_matrix.png')
    plt.show()

def main():
    print("Veriler yükleniyor...")
    x_train, y_train, x_test, y_test = load_data()

    print("CNN modeli oluşturuluyor...")
    cnn_model = build_cnn_feature_extractor()

    print("CNN eğitiliyor...")
    train_cnn(cnn_model, x_train, y_train, epochs=1, batch_size=64)

    print("CNN özellikleri çıkarılıyor...")
    train_features, test_features = extract_features(cnn_model, x_train, x_test)

    print(f"Train feature shape: {train_features.shape}")
    print(f"Test feature shape: {test_features.shape}")

    svm_model = train_svm(train_features, y_train, kernel="rbf", C=1.0, gamma="scale")

    print("Hibrit model değerlendiriliyor...")
    evaluate_hybrid_model(svm_model, test_features, y_test)

    print("CNN + SVM hibrit model tamamlandı.")

if __name__ == "__main__":
    main()