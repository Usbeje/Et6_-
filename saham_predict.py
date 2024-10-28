import subprocess
import sys

# Fungsi untuk menginstal modul jika belum terpasang
def install_and_import(package):
    if package == "sklearn":
        package = "scikit-learn"
    try:
        __import__(package)
    except ImportError:
        print(f"Memasang {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Daftar modul yang diperlukan
required_packages = [
    "numpy", "tensorflow", "scikit-learn", "pandas", 
    "yfinance", "python-dotenv"
]

# Memasang semua modul yang diperlukan
for package in required_packages:
    install_and_import(package)

# Setelah modul dipasang, impor modulnya
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import yfinance as yf

# Fungsi untuk mendapatkan data historis saham dari yfinance
def dapatkan_data(ticker, tahun_awal, tahun_akhir):
    data = yf.download(ticker, start=tahun_awal, end=tahun_akhir)
    return data['Close']

# Fungsi untuk membangun model LSTM
def buat_model_lstm(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Fungsi untuk memprediksi harga saham di masa depan
def prediksi_masa_depan(data, model, scaler, days_to_predict):
    last_60_days = data[-60:]  # Ambil 60 hari terakhir
    scaled_data = scaler.transform(np.array(last_60_days).reshape(-1, 1))

    future_predictions = []
    input_data = scaled_data.reshape(1, 60, 1)

    for _ in range(days_to_predict):
        pred_price = model.predict(input_data)[0][0]
        future_predictions.append(pred_price)
        
        # Perbarui input data untuk iterasi berikutnya
        input_data = np.append(input_data[:, 1:, :], [[[pred_price]]], axis=1)
    
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions

# Main program
ticker = input("Ticker yang ingin diprediksi (contoh: BTC-USD): ")
tahun_input = input("Tahun? (contoh: 2023-12-30 - 2024-10-01): ")
days_to_predict = int(input("Prediksi berapa hari ke depan?: "))

# Pemrosesan input tahun
tahun_awal, tahun_akhir = tahun_input.split(" - ") if " - " in tahun_input else tahun_input.split("-")
data = dapatkan_data(ticker, tahun_awal.strip(), tahun_akhir.strip())

# Scaling data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(np.array(data).reshape(-1, 1))

# Membagi data untuk pelatihan
X_train, Y_train = [], []
for i in range(60, len(data_scaled)):
    X_train.append(data_scaled[i-60:i, 0])
    Y_train.append(data_scaled[i, 0])

X_train, Y_train = np.array(X_train), np.array(Y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Membuat dan melatih model
model = buat_model_lstm((X_train.shape[1], 1))
print("Mulai pelatihan model...")
model.fit(X_train, Y_train, epochs=10, batch_size=32)

# Prediksi harga saham di masa depan
print("Memprediksi harga di masa depan...")
future_predictions = prediksi_masa_depan(data, model, scaler, days_to_predict)

# Menampilkan hasil prediksi
current_date = datetime.strptime(tahun_akhir.strip(), "%Y-%m-%d")
print("\nPrediksi harga saham untuk {} hari ke depan:".format(days_to_predict))
for i, pred in enumerate(future_predictions):
    pred_date = current_date + timedelta(days=i+1)
    print(f"{pred_date.date()}: {pred[0]:.2f}")
