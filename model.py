import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("house_prices.csv")  # ← ganti nama file sesuai yang kamu punya

# One-Hot Encoding untuk Lokasi
# Pada bagian ini, kita menggunakan pd.get_dummies untuk melakukan One-Hot Encoding pada kolom 'Lokasi'. One-Hot Encoding adalah teknik untuk mengubah variabel kategorik menjadi variabel numerik yang dapat diolah oleh model machine learning. Dengan menggunakan drop_first=True, kita menghapus salah satu dari variabel kategorik yang dihasilkan untuk menghindari multicollinearity.
df = pd.get_dummies(df, columns=['Lokasi'], drop_first=True)

# Pilih fitur
# Kita memilih fitur yang akan digunakan untuk model regresi linear. Fitur 'LuasTanah' dan fitur-fitur yang dihasilkan dari One-Hot Encoding untuk 'Lokasi' akan digunakan.
X = df[['LuasTanah'] + list(df.columns[df.columns.str.startswith('Lokasi_')])]
y = df['Harga']  # 'Harga' akan digunakan sebagai label atau target.

# Split data
# Kita menggunakan train_test_split untuk membagi dataset menjadi dua bagian: data pelatihan (X_train, y_train) dan data pengujian (X_test, y_test). Parameter test_size=0.2 berarti 20% dari dataset akan digunakan sebagai data pengujian. Random_state=42 digunakan untuk memastikan hasil split data tetap sama setiap kali program dijalankan.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat model dan latih
# Kita menggunakan LinearRegression untuk membuat model regresi linear. Kemudian, kita menggunakan metode fit() untuk melatih model dengan data pelatihan.
model = LinearRegression()
model.fit(X_train, y_train)

def predict_price(luas_tanah, lokasi):
    # Fungsi ini digunakan untuk memprediksi harga rumah berdasarkan luas tanah dan lokasi.
    lokasi_features = {col: 0 for col in df.columns if col.startswith("Lokasi_")}
    # Kita membuat dictionary untuk menampung fitur-fitur lokasi. Nilai default untuk setiap fitur adalah 0.
    if f'Lokasi_{lokasi}' in lokasi_features:
        lokasi_features[f'Lokasi_{lokasi}'] = 1
    # Jika lokasi yang diberikan cocok dengan salah satu fitur lokasi, maka kita mengubah nilainya menjadi 1.
    input_data = np.array([[luas_tanah] + list(lokasi_features.values())])
    # Kita membuat array numpy untuk menampung data input yang akan digunakan untuk prediksi. Data input ini terdiri dari luas tanah dan nilai-nilai fitur lokasi.
    return model.predict(input_data)[0]
    # Kita menggunakan model untuk memprediksi harga berdasarkan data input dan mengembalikan hasil prediksi.
