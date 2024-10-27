import yfinance as yf
import numpy as np
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Fungsi untuk mendapatkan data historis saham
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

# Fungsi untuk melatih model dan memprediksi harga
def prediksi_harga(data, model, scaler, days_to_predict):
    scaled_data = scaler.transform(np.array(data).reshape(-1, 1))
    X_test, Y_test = [], []

    for i in range(60, len(scaled_data)):
        X_test.append(scaled_data[i-60:i, 0])
        Y_test.append(scaled_data[i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Prediksi
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # Menghitung perubahan persentase harga
    price_change_percent = ((predictions[-1] - data[-1]) / data[-1]) * 100

    return predictions, price_change_percent[0]  # Pastikan mengakses nilai tunggal

# Main program
ticker = input("Ticker yang ingin diprediksi (contoh: BTC-USD): ")
tahun_input = input("Tahun? (contoh: 2024-2025 atau 2024-12-30 - 2025-12-30): ")

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

# Melakukan prediksi
print("Memprediksi...")
predictions, price_change_percent = prediksi_harga(data, model, scaler, days_to_predict=30)

print(f"Prediksi selesai. Perubahan harga: {price_change_percent:.2f}%")

if price_change_percent > 0:
    print("Prediksi: Harga akan meningkat.")
else:
    print("Prediksi: Harga akan menurun.")
