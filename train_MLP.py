import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import joblib 

# Función para cargar los datos desde dos directorios
def load_data(real_dir, fake_dir):
    # Obtener la lista de archivos .npy en cada directorio
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.npy')]
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith('.npy')]

    # Cargar los vectores desde los archivos .npy
    real_data = [np.load(f) for f in real_files]
    fake_data = [np.load(f) for f in fake_files]

    # Convertir a matrices numpy
    real_data = np.array(real_data)
    fake_data = np.array(fake_data)

    # Crear etiquetas: 1 para real, 0 para fake
    real_labels = np.ones(len(real_data))
    fake_labels = np.zeros(len(fake_data))

    # Unir los datos y etiquetas
    X = np.vstack((real_data, fake_data))
    y = np.hstack((real_labels, fake_labels))

    # Mezclar los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    print("Datos de entrenamiento:", X_train.shape)
    print("Datos de prueba:", X_test.shape)

    return X_train, X_test, y_train, y_test

# Función para entrenar el modelo
def train_model(real_dir, fake_dir):
    X_train, X_test, y_train, y_test = load_data(real_dir, fake_dir)

    # Normalizar los datos
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Crear la red neuronal con parámetros mejorados
    clf = MLPClassifier(hidden_layer_sizes=(256, 128, 64), 
                        activation='relu', 
                        solver='adam', 
                        alpha=0.001, 
                        max_iter=100, 
                        random_state=1, 
                        verbose=True)

    # Entrenar el modelo
    clf.fit(X_train, y_train)
    # Evaluar el modelo
    y_pred = clf.predict(X_test)
    
    return clf, scaler, y_test, y_pred



def evaluation_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
      

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")

    
    # Calcular la curva ROC y el AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc:.4f}")

    '''
    # Graficar la curva ROC
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    
    # Guardar como archivo PNG
    roc_path = os.path.join('.../train/1000', "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()'''



if __name__ == "__main__":
    # Rutas de los directorios con los vectores
    real_dir = ".../features/1000/real_img"
    fake_dir = ".../features/1000/fake_img"

    # Entrenar el modelo
    model, scaler, y_test, y_pred = train_model(real_dir, fake_dir)
    report = evaluation_model(y_test, y_pred)
    print(report)

    # Guardar el modelo y el scaler
    joblib.dump(model, '.../train/1000/mlp_model.pkl')
    joblib.dump(scaler, '.../train/1000/scaler.pkl')


