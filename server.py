from flask import Flask, request, jsonify, render_template
from predict import Predict

app = Flask(__name__)
 
@app.route('/')
def index():
    # Render the index.html file
    return render_template('index.html')

@app.route('/main')
def main():
    # Render the index.html file
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()  # Get input data from the request
    predictions = Predict(data)  # Use the ML model to make predictions
    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True)
