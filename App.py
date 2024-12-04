from flask import Flask, request, jsonify
from flask_cors import CORS 
import joblib

#Inicializar Flask
app = Flask(__name__)
CORS(app) #Permite solicitudes desde otros origenes

model = joblib.load('dropout_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        required_keys = [
            'estadoCivil', 'modoApli', 'carrera', 'turno', 'tituloPre',
            'nacionalidad', 'tituloMadre', 'tituloPadre', 'ocupacionMadre',
            'ocupacionPadre', 'desplazado', 'especial', 'deudor', 'vigente',
            'genero', 'becario', 'edad', 'internacional', 'acreditadas1',
            'inscritas1', 'evaluadas1', 'aprobadas1', 'calificadas1',
            'noevaluadas1', 'acreditadas2', 'inscritas2', 'evaluadas2',
            'aprobadas2', 'calificadas2', 'noevaluadas2'
        ]
        if not all(key in data for key in required_keys):
            return jsonify({'error': 'Faltan datos'}), 400

        input_data = [float(data[key]) for key in required_keys]

        # Predicción
        prediction = model.predict([input_data])[0]
        probability = model.predict_proba([input_data])[0].max()

        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)