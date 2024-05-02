from model import classifierDT
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

# Define the number of features expected by the classifier
NUM_FEATURES = 132

def preprocess_input_data(newdata):
    # If the input data has less than 132 features, add dummy values to make it compatible
    if len(newdata) < NUM_FEATURES:
        # Add dummy values to make the input data have 132 features
        newdata = np.concatenate([newdata, np.zeros(NUM_FEATURES - len(newdata))])
    return newdata.reshape(1, -1)

def make_prediction(newdata):
    probaDT = classifierDT.predict_proba(newdata)
    predDT = classifierDT.predict(newdata)
    return probaDT.round(5), predDT

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        newdata = np.array(data['newdata'])
        # Preprocess the input data to ensure it has the correct number of features
        newdata = preprocess_input_data(newdata)
        probabilities, prediction = make_prediction(newdata)
        response = {
            'probabilities': probabilities.tolist(),
            'prediction': prediction.tolist()
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
