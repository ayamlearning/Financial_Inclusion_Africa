import pickle

from flask import Flask
from flask import request
from flask import jsonify


model_file = 'model_financial_inclusion.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('bank_account')

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()

    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    bank_account = y_pred >= 0.5

    result = {
        'bank_account_probability': float(y_pred),
        'bank_account': bool(bank_account)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8000)
