
from model import classifierDT
from flask import Flask, request, jsonify
app = Flask(__name__)

def make_prediction(newdata):
    probaDT = classifierDT.predict_proba(newdata)
    predDT = classifierDT.predict(newdata)
    return probaDT.round(5), predDT





@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        newdata = data['newdata']
        probabilities, prediction = make_prediction(newdata)
        response = {
            'probabilities':probabilities.tolist(),
            'prediction': prediction.tolist()
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
