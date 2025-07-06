import os
import numpy as np
import joblib


# Función para predecir nuevos vectores
def predict_from_directory(clf, scaler, directory_path, expected_features=695):
    vector_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.npy')]
    vectors = np.array([np.load(f) for f in vector_files])
    
    # Ajustar cada vector para que tenga expected_features (recortar o rellenar)
    adjusted_vectors = []
    for v in vectors:
        if v.shape[0] > expected_features:
            adjusted = v[:expected_features]
        elif v.shape[0] < expected_features:
            pad_width = expected_features - v.shape[0]
            adjusted = np.pad(v, (0, pad_width), mode='constant')
        else:
            adjusted = v
        adjusted_vectors.append(adjusted)

    c_vectors = np.array(adjusted_vectors)

    # Normalizar los datos con el mismo scaler usado en entrenamiento
    c_vectors = scaler.transform(c_vectors)
    
    predictions = clf.predict(c_vectors)
    probabilities = clf.predict_proba(c_vectors)[:, 1]  # Probabilidad de ser "real"

    with open('.../test/1000/R800.txt', 'w') as f:
        for i, file in enumerate(vector_files):
            file_name = os.path.basename(file)  # Obtiene solo el nombre del archivo
            label = "Real" if predictions[i] == 1 else "Falso"
            f.write(f"{file_name}; {label} ({probabilities[i]:.2f})\n")

    print("Resultados guardados en 'R.txt'")

if __name__ == "__main__":
    # Directorio con los vectores a predecir    
    directory_path = ".../features/800"

    # Cargar el modelo y el scalerclf = joblib.load(model_path)
    scaler = joblib.load(".../train/800/scaler.pkl")
    model = joblib.load(".../train/800/mlp_model.pkl")

    predict_from_directory(model, scaler, directory_path)
