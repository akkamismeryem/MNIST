import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import os

def load_data():
    """
    Kaydedilmiş numpy dosyalarından verileri yükler.
    """
    x_train = np.load('output/X_train.npy')
    y_train = np.load('output/y_train.npy')
    x_test = np.load('output/X_test.npy')
    y_test = np.load('output/y_test.npy')
    return x_train, y_train, x_test, y_test

def train_decision_tree(x_train, y_train, max_depth=None, min_samples_leaf=1, max_features=None, criterion='gini'):
    # Modeli tanımla
    dt_model = DecisionTreeClassifier(
            max_depth=max_depth, 
            min_samples_leaf=min_samples_leaf, 
            max_features=max_features, 
            criterion=criterion, 
            random_state=42
        )
    dt_model.fit(x_train, y_train)
    return dt_model
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    # Metrikler
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nDecision Tree Başarı Metrikleri:")
    print(f"Doğruluk (Accuracy): {accuracy:.4f}")
    print(f"Kesinlik (Precision): {precision:.4f}")
    print(f"Duyarlılık (Recall): {recall:.4f}")
    print(f"F1-Skor: {f1:.4f}")
    
    print("\nDetaylı Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix (Karmaşıklık Matrisi)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=range(10), yticklabels=range(10))
    plt.title('Decision Tree Confusion Matrix')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')

    plt.savefig('output/decisiontree_confusion_matrisi.png')

    return accuracy, precision, recall, f1
def main():

    x_train, y_train, x_test, y_test = load_data() 
    print (f'Eğitim Verisi Boyutu: {x_train.shape}')
    print(f'Test Verisi Boyutu: {x_test.shape}')

    dt_model = train_decision_tree(x_train, y_train, max_depth=12, min_samples_leaf = 5,max_features='sqrt', criterion='entropy')

    evaluate_model(dt_model, x_test, y_test)

if __name__ == "__main__":
    main()