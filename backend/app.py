from flask import Flask, request, jsonify, render_template, Response, session
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import io

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a strong secret in production

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Mappings for categorical values
weather_map = {'clear': 0, 'rainy': 1, 'foggy': 2, 'snowy': 3}
road_map = {'highway': 0, 'city': 1, 'rural': 2}
class_map = {0: "Low Risk", 1: "High Risk", 2: "Very High Risk"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    weather = data.get('weather', 'clear').lower()
    road_type = data.get('road_type', 'highway').lower()

    try:
        speed = float(data.get('vehicle_speed', 0))
    except (TypeError, ValueError):
        speed = 0.0

    try:
        traffic_density = float(data.get('traffic_density', 0))
    except (TypeError, ValueError):
        traffic_density = 0.0

    input_data = np.array([[weather_map.get(weather, 0), road_map.get(road_type, 0), speed, traffic_density]])
    prediction = model.predict(input_data)
    predicted_class = prediction[0]
    result = class_map.get(predicted_class, "Unknown Risk")

    # Store prediction in session
    if 'predictions' not in session:
        session['predictions'] = []
    predictions_list = session['predictions']
    predictions_list.append(result)
    session['predictions'] = predictions_list
    session.modified = True

    return jsonify({'result': result})
@app.route('/plot')
def plot():
    predictions = session.get('predictions', [])
    
    idx = request.args.get('index', default=None, type=int)
    risk_categories = ["Low Risk", "High Risk", "Very High Risk"]

    risk_counts = {risk: 0 for risk in risk_categories}
    
    if idx is not None and 0 <= idx < len(predictions):
        risk = predictions[idx].strip().title()  # normalize case & strip spaces
        print(f"Prediction at index {idx}: '{risk}'")  # Debugging print
        
        if risk in risk_counts:
            risk_counts[risk] = 1
        else:
            print(f"Warning: Risk '{risk}' not in predefined categories.")
    else:
        # Optional: sum counts of all predictions if no valid idx
        for r in predictions:
            r_norm = r.strip().title()
            if r_norm in risk_counts:
                risk_counts[r_norm] += 1

    labels = list(risk_counts.keys())
    counts = list(risk_counts.values())

    fig, ax = plt.subplots()
    ax.bar(labels, counts, color=['green', 'orange', 'red'])
    ax.set_ylabel('Prediction Count')
    ax.set_title(f'Accident Risk Prediction #{idx if idx is not None else "N/A"}')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    return Response(buf.getvalue(), mimetype='image/png')


@app.route('/session_length')
def session_length():
    predictions = session.get('predictions', [])
    return jsonify({'length': len(predictions)})

if __name__ == '__main__':
    app.run(debug=True)