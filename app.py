from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Memuat dataset dari file CSV
data = pd.read_csv('drug_cosumption_new.csv')

# Menghapus kolom 'Unnamed: 0' jika ada
if 'Unnamed: 0' in data.columns:
    data.drop('Unnamed: 0', axis=1, inplace=True)

# Memisahkan atribut dan label
X = data.drop(columns=['heroin'])  # Atribut
y = data['heroin']  # Label

# Mengonversi label menjadi integer secara manual
class_mapping = {label: idx for idx, label in enumerate(np.unique(y))}

f = open('final_model.pickle','rb')
estimators = pickle.load(f)

# Route utama untuk menampilkan form input
@app.route('/')
def index():
    return render_template('index.html', columns=X.columns)

# Route untuk menangani prediksi data baru
@app.route('/predict', methods=['POST'])
def predict():
    data_input = request.form.to_dict(flat=True)
    print('Data input:', data_input)  # Log input data
    data_values = [[float(data_input[col]) for col in X.columns]]

    X_new = pd.DataFrame(data_values, columns=X.columns)
    pred = []  # Initialize pred here to avoid accumulation from previous requests

    for estimator in estimators:
        prediksi = estimator.predict(X_new)
        pred.append(prediksi[0])  # Menyimpan prediksi dalam list pred

    # Majority vote untuk prediksi data baru
    prediksi_majority_vote = np.bincount(pred).argmax()
    class_counts = np.bincount(pred, minlength=len(np.unique(y)))
    pred_result = {label: int(count) for label, count in zip(class_mapping.keys(), class_counts)}

    # Konversi int64 menjadi int untuk prediksi mayoritas
    prediksi_majority_vote = int(prediksi_majority_vote)
    pred_result = {label: int(count) for label, count in pred_result.items()}

    response = {
        'prediksi_majority_vote': prediksi_majority_vote,
        'pred_result': pred_result
    }

    print('Prediction response:', response)  # Log response data
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
