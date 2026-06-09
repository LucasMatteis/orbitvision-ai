import os
import cv2
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier

DATASET_PATH = "dataset"

X = []
y = []

labels = {
    "forest": 0,
    "fire": 1,
    "deforestation": 2
}

for class_name, label in labels.items():

    folder = os.path.join(DATASET_PATH, class_name)

    for filename in os.listdir(folder):

        img_path = os.path.join(folder, filename)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (100, 100))

        mean_b = np.mean(img[:, :, 0])
        mean_g = np.mean(img[:, :, 1])
        mean_r = np.mean(img[:, :, 2])

        features = [mean_b, mean_g, mean_r]

        X.append(features)
        y.append(label)

model = RandomForestClassifier()

model.fit(X, y)

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/model.pkl")

print("Modelo treinado com sucesso!")