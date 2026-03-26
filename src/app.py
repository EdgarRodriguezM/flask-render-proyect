import os
from flask import Flask, render_template, request
import joblib
import numpy as np

# Configuramos Flask para que encuentre la carpeta templates en la raíz (fuera de src)
app = Flask(__name__, template_folder='../templates')

# --- RUTA DINÁMICA PARA EL MODELO ---
# Obtenemos la ruta de la carpeta actual (src)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Subimos un nivel y entramos a 'models'
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'diabetes_model.pkl')

# Cargar el modelo
model = joblib.load(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # ... (el resto del código de predicción se mantiene igual)
    try:
        input_data = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['blood_pressure']),
            float(request.form['skin_thickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['dpf']),
            float(request.form['age'])
        ]
        final_features = np.array(input_data).reshape(1, -1)
        prediction = model.predict(final_features)
        
        result = "Positivo (Riesgo detectado)" if prediction[0] == 1 else "Negativo (Sin riesgo aparente)"
        color = "danger" if prediction[0] == 1 else "success"
        
        return render_template('index.html', prediction_text=result, alert_class=color)
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}", alert_class="warning")

if __name__ == "__main__":
    app.run(debug=True)
