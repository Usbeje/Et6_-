import numpy as np
import pandas as pd
import yfinance as yf
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from termcolor import colored

# Meminta input ticker dan periode waktu
ticker_input = input("Ticker yang ingin diprediksi (contoh: BTC-USD): ")
tahun_input = input("Tahun? (contoh: 2024-2025 atau 2024-12-30 - 2025-12-30): ")

# Memisahkan periode waktu
tahun_awal, tahun_akhir = tahun_input.split("-")
start_date = f"{tahun_awal[:4]}-{tahun_awal[5:7]}-{tahun_awal[8:10]}" if "'" in tahun_awal else f"{tahun_awal}-01-01"
end_date = f"{tahun_akhir[:4]}-{tahun_akhir[5:7]}-{tahun_akhir[8:10]}" if "'" in tahun_akhir else f"{tahun_akhir}-12-31"

# Mengunduh data dari Yahoo Finance
data = yf.download(ticker_input, start=start_date, end=end_date)
data = data[['Close']]
dataset = data.values

# Menyiapkan data pelatihan
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[0:train_size, :]

x_train, y_train = [], []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Membangun model LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Proses pelatihan model dengan indikator titik-titik
print("Memprediksi", end="")
for _ in range(5):  # waktu pelatihan tergantung jumlah epoch, bisa sesuaikan untuk visualisasi
    print(".", end="")
    time.sleep(1)
print("\nMulai pelatihan model...")
model.fit(x_train, y_train, batch_size=1, epochs=1)
print("Prediksi selesai.\n")

# Data pengujian
test_data = scaled_data[train_size - 60:, :]
x_test, y_test = [], dataset[train_size:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Prediksi harga
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Menghitung perubahan harga dalam persentase
initial_price = data['Close'].iloc[train_size]  # Harga awal periode prediksi
final_price = predictions[-1][0]  # Harga terakhir dari prediksi
price_change_percent = ((final_price - initial_price) / initial_price) * 100

# Menampilkan hasil prediksi dengan warna
if price_change_percent > 0:
    result_text = colored(f"{ticker_input} Naik = {price_change_percent:.2f}%", "green")
elif price_change_percent < 0:
    result_text = colored(f"{ticker_input} Turun = {price_change_percent:.2f}%", "red")
else:
    result_text = colored(f"{ticker_input} Tidak Naik atau Turun = {price_change_percent:.2f}%", "black")

print(result_text)
