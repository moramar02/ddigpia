import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay
)

# Ruta al archivo de resultados
result_file = ".../test/400/R400.txt"
output_dir = ".../test/400new"
os.makedirs(output_dir, exist_ok=True)

# Listas para etiquetas verdaderas y predichas
y_true = []
y_pred = []

# Leer el archivo línea a línea
with open(result_file, "r") as f:
    for line in f:
        if ";" not in line:
            continue  # Saltar líneas mal formateadas

        filename, pred_info = line.strip().split(";")
        prediction = pred_info.strip().split()[0]  # "Real" o "Falso"

        # Inferir etiqueta real desde el nombre
        true_label = 1 if "real_" in filename.lower() else 0
        pred_label = 1 if prediction.lower() == "real" else 0

        y_true.append(true_label)
        y_pred.append(pred_label)

# Convertir a array
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Métricas
cm = confusion_matrix(y_true, y_pred)
cm_norm = confusion_matrix(y_true, y_pred, normalize='true')

# Guardar matriz de confusión sin normalizar
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Falso", "Real"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz Sin Normalizar")
plt.savefig(os.path.join(output_dir, "matrix_raw_400.png"))
plt.close()

# Guardar matriz de confusión normalizada
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=["Falso", "Real"])
disp_norm.plot(cmap=plt.cm.Blues)
plt.title("Matriz Normalizada")
plt.savefig(os.path.join(output_dir, "matrix_norm_400.pdf"))
plt.close()

print("Imágenes y métricas guardadas en:", output_dir)
