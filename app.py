from flask import Flask, render_template, request
import os
import cv2
import joblib
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = joblib.load("model/model.pkl")

classes = {
    0: "Floresta Preservada",
    1: "Área Queimada",
    2: "Área Desmatada"
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    img = cv2.imread(filepath)

    img = cv2.resize(img, (100, 100))

    mean_b = np.mean(img[:, :, 0])
    mean_g = np.mean(img[:, :, 1])
    mean_r = np.mean(img[:, :, 2])

    prediction = model.predict(
        [[mean_b, mean_g, mean_r]]
    )[0]

    result = classes[prediction]

    if prediction == 0:
        risk = "Baixo"
        risk_class = "risk-low"
        emoji = "🌳"
        recommendation = "Monitoramento contínuo."

    elif prediction == 1:
        risk = "Alto"
        risk_class = "risk-high"
        emoji = "🔥"
        recommendation = "Acionar equipes ambientais."

    else:
        risk = "Médio"
        risk_class = "risk-medium"
        emoji = "🪓"
        recommendation = "Realizar inspeção da área."

    return render_template(
        "result.html",
        result=result,
        risk=risk,
        risk_class=risk_class,
        emoji=emoji,
        recommendation=recommendation
    )


if __name__ == "__main__":
    app.run(debug=True)