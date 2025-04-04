import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load Dataset
# Pada bagian ini, kita menggunakan pandas untuk membaca dataset dari file CSV bernama "house_prices.csv" dan menyimpannya dalam DataFrame bernama df.
df = pd.read_csv("house_prices.csv")

# 🔍 Cek nilai kosong
# Sebelum kita mulai membersihkan dataset, kita perlu mengetahui berapa banyak nilai kosong yang ada dalam dataset. Kita menggunakan metode isnull().sum() untuk menghitung jumlah nilai kosong untuk setiap kolom.
print("Sebelum dibersihkan:\n", df.isnull().sum())

# 🚫 Hapus baris yang mengandung nilai kosong
# Kita menggunakan metode dropna() untuk menghapus baris yang mengandung nilai kosong. Dengan demikian, kita dapat menghilangkan baris yang tidak lengkap dan membuat dataset lebih konsisten.
df = df.dropna()

# 🔍 Cek ulang
# Setelah menghapus baris yang mengandung nilai kosong, kita perlu mengetahui apakah masih ada nilai kosong yang tersisa dalam dataset. Kita menggunakan metode isnull().sum() lagi untuk menghitung jumlah nilai kosong yang tersisa.
print("Setelah dibersihkan:\n", df.isnull().sum())

# 2. Encode lokasi jadi angka
# Pada bagian ini, kita menggunakan LabelEncoder untuk mengubah kolom "Lokasi" yang berisi nilai kategorik menjadi nilai numerik. Hal ini dilakukan karena model regresi linear tidak dapat langsung mengolah nilai kategorik.
le = LabelEncoder()
df["Lokasi"] = le.fit_transform(df["Lokasi"])

# 3. Fitur dan Label
# Kita menentukan fitur dan label untuk model regresi linear. Fitur kita adalah "LuasTanah" dan "Lokasi", sedangkan label kita adalah "Harga".
X = df[["LuasTanah", "Lokasi"]]
y = df["Harga"]

# 4. Split data
# Kita menggunakan train_test_split untuk membagi dataset menjadi dua bagian: data pelatihan (X_train, y_train) dan data pengujian (X_test, y_test). Parameter test_size=0.2 berarti 20% dari dataset akan digunakan sebagai data pengujian.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model Regresi Linear
# Kita menggunakan LinearRegression untuk membuat model regresi linear. Kemudian, kita menggunakan metode fit() untuk melatih model dengan data pelatihan.
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Simpan model dan encoder
# Setelah model dilatih, kita menggunakan joblib untuk menyimpan model dan encoder lokasi dalam file bernama "model.pkl". Dengan demikian, kita dapat menggunakan model ini untuk prediksi di masa depan.
joblib.dump((model, le), "model.pkl")

print("✅ Model berhasil dilatih dan disimpan sebagai model.pkl!")
