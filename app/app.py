from utils import Predictor
from utils import DataLoader

from flask import Flask, request, jsonify, make_response

import pandas as pd
import json


app = Flask(__name__)


@app.route('/predict', methods=['GET'])
def predict():
    received_keys = sorted(list(request.form.keys()))
    if len(received_keys) > 1 or 'data' not in received_keys:
        err = 'Wrong request keys'
        return make_response(jsonify(error=err), 400)

    data = json.loads(request.form.get(received_keys[0]))
    df = pd.DataFrame.from_dict(data)

    loader = DataLoader()
    loader.fit(df)
    processed_df = loader.load_data()

    predictor = Predictor()
    response_dict = {'prediction': predictor.predict(processed_df).tolist()}

    return make_response(jsonify(response_dict), 200)


if __name__ == "__main__":
    # for containerized app
    app.run(debug=True, host='0.0.0.0', port=8000)

    # For local tests
    # app.run(debug=True, host='127.0.0.1', port=8000)

# TODO: split new dataset into corresponding csvs
# TODO: test new DataLoader class in notebook
# TODO: Define the best model for classification
# TODO: Create and test new model using notebook
# TODO: Test Flask API locally and using Docker
