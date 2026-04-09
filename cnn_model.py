import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))

    return x_train, y_train, x_test, y_test

def build_cnn():
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

def evaluate_model(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("CNN Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig('output/cnn_confusion_matrix.png')
    plt.show()

def main():
    print("Veriler yükleniyor...")
    x_train, y_train, x_test, y_test = load_data()

    print("CNN modeli oluşturuluyor...")
    model = build_cnn()

    print("Model eğitiliyor...")
    train_cnn(model, x_train, y_train, epochs=1, batch_size=64)

    print("Model değerlendiriliyor...")
    evaluate_model(model, x_test, y_test)

    print("CNN işlemi tamamlandı.")
    print("Confusion matrix 'output/cnn_confusion_matrix.png' dosyasına kaydedildi.")

if __name__ == "__main__":
    main()