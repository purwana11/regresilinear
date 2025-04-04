import os  # Import modul os untuk mengakses fungsi-fungsi sistem operasi
import numpy as np  # Import modul numpy untuk penggunaan array dan operasi matematika
import pandas as pd  # Import modul pandas untuk penggunaan DataFrame dan operasi data
import joblib  # Import modul joblib untuk load model yang telah disimpan
from flask import Flask, request, render_template  # Import modul Flask untuk membuat aplikasi web
import matplotlib  # Import modul matplotlib untuk membuat grafik
matplotlib.use('Agg')  # Menggunakan backend non-GUI untuk matplotlib agar dapat berjalan tanpa GUI
import matplotlib.pyplot as plt  # Import modul pyplot untuk membuat grafik
import io  # Import modul io untuk menghandle input/output
import base64  # Import modul base64 untuk mengkonversi gambar ke string base64

# Inisialisasi Flask
app = Flask(__name__)

# Load model dan encoder lokasi
if os.path.exists("model.pkl"):  # Cek apakah file model.pkl ada
    model, le = joblib.load("model.pkl")  # Load model dan encoder lokasi dari file model.pkl
else:
    raise FileNotFoundError("File model.pkl tidak ditemukan. Jalankan train_model.py terlebih dahulu.")  # Jika file tidak ditemukan, maka raise error

# Fungsi untuk membuat grafik prediksi
def generate_graph(luas_tanah, harga_prediksi):
    luas_sample = np.linspace(50, 500, 100)  # Membuat contoh luas tanah dari 50 hingga 500 dengan 100 titik
    lokasi_default = 0  # Menggunakan lokasi default 0 untuk contoh
    input_data = pd.DataFrame({
        "LuasTanah": luas_sample,  # Membuat DataFrame dengan luas tanah sebagai contoh
        "Lokasi": [lokasi_default]*len(luas_sample)  # Menggunakan lokasi default untuk semua contoh
    })

    harga_sample = model.predict(input_data)  # Menggunakan model untuk memprediksi harga berdasarkan contoh luas tanah

    plt.figure(figsize=(6, 4))  # Membuat figure dengan ukuran 6x4 inch
    plt.plot(luas_sample, harga_sample, label="Regresi Linear", color="blue")  # Membuat plot untuk regresi linear
    plt.scatter(luas_tanah, harga_prediksi, color="red", marker="o", label="Prediksi Anda")  # Membuat scatter plot untuk prediksi
    plt.xlabel("Luas Tanah (m²)")  # Label untuk sumbu x
    plt.ylabel("Harga Rumah")  # Label untuk sumbu y
    plt.legend()  # Menampilkan legenda
    plt.grid()  # Menampilkan grid

    img = io.BytesIO()  # Membuat BytesIO untuk menyimpan gambar
    plt.savefig(img, format="png")  # Menyimpan gambar ke BytesIO dalam format PNG
    img.seek(0)  # Mengembalikan posisi BytesIO ke awal
    graph_url = base64.b64encode(img.getvalue()).decode()  # Mengkonversi BytesIO ke string base64 dan decode ke string
    plt.close()  # Menutup plot untuk menghemat memori

    return f"data:image/png;base64,{graph_url}"  # Mengembalikan URL gambar dalam format base64

# Route utama
@app.route("/", methods=["GET", "POST"])
def index():
    graph_url = None  # Inisialisasi URL grafik sebagai None
    prediction_text = None  # Inisialisasi teks prediksi sebagai None

    if request.method == "POST":  # Jika request adalah POST
        try:
            luas_tanah = float(request.form["luas_tanah"])  # Mengambil luas tanah dari form
            lokasi = request.form["lokasi"]  # Mengambil lokasi dari form

            # Ubah lokasi ke bentuk numerik
            lokasi_encoded = le.transform([lokasi])[0]  # Menggunakan encoder untuk mengubah lokasi ke bentuk numerik

            # Prediksi harga
            input_data = pd.DataFrame([[luas_tanah, lokasi_encoded]], columns=["LuasTanah", "Lokasi"])  # Membuat DataFrame untuk input prediksi
            harga_prediksi = model.predict(input_data)[0]  # Menggunakan model untuk memprediksi harga

            # Buat grafik
            graph_url = generate_graph(luas_tanah, harga_prediksi)  # Menggunakan fungsi generate_graph untuk membuat grafik
            prediction_text = f"Prediksi harga rumah: Rp {harga_prediksi:,.2f}"  # Mengformat teks prediksi dengan harga prediksi

        except Exception as e:
            prediction_text = f"Terjadi kesalahan: {str(e)}"  # Jika terjadi kesalahan, maka tampilkan pesan kesalahan

    return render_template("index.html", prediction_text=prediction_text, graph_url=graph_url)  # Mengembalikan template dengan teks prediksi dan URL grafik

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Mengambil port dari environment variable atau default ke 8080
    app.run(host="0.0.0.0", port=port)  # Menjalankan aplikasi Flask pada host dan port yang ditentukan
