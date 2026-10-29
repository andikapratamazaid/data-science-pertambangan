# Data Science Perusahaan Pertambangan: EDA sampai API & Docker

Satu project capstone data science dengan konteks perusahaan tambang
(batubara dan nikel), mencakup seluruh alur yang biasa diminta dari
seorang data scientist: **EDA, regresi, klasifikasi, forecasting, NLP,
dashboard eksekutif, ML API, dan Docker**, dalam satu repositori yang
bisa dijalankan ujung ke ujung.

## Apa yang bisa diambil (sumber data)

Data operasional tambang tidak dipublikasikan perusahaan mana pun, jadi
project ini menggabungkan dua jenis data secara terbuka:

| Data | Status | Sumber | Dipakai di |
|------|--------|--------|------------|
| Harga batubara, nikel, emas, tembaga bulanan 2015–2024 | **ASLI** | World Bank Commodity Markets "Pink Sheet" | EDA, Forecasting |
| Harga saham ANTM, INCO, PTBA, MDKA bulanan 2019–2026 | **ASLI** | Yahoo Finance (adjusted close) | EDA |
| 20 headline berita tambang, Okt 2026 | **ASLI** | Media Indonesia (judul, media, tanggal, URL tersimpan) | NLP |
| Telemetri armada 12 unit x 365 hari (2025) | **SIMULASI** | Dibangkitkan `buat_data_simulasi.py`, hubungan antar-variabel realistis dan terdokumentasi | EDA, Regresi, Klasifikasi, Dashboard, API |
| Produksi batubara harian 365 hari | **SIMULASI** | Sama seperti di atas | EDA |

Bila suatu saat data operasional asli tersedia (ekspor sistem fleet
management atau ERP tambang), cukup ganti dua file simulasi di
`data/olahan/`; semua modul membaca dari file yang sama.

## Peta modul ke kebutuhan

| Kebutuhan | Modul | Isi |
|-----------|-------|-----|
| Ambil data asli | `src/ambil_data.py` | Unduh Pink Sheet World Bank + harga saham Yahoo |
| Siapkan data | `src/siapkan_data.py` | Ubah data mentah menjadi CSV olahan |
| Data simulasi | `src/buat_data_simulasi.py` | Telemetri armada + produksi harian (berlabel) |
| EDA | `src/01_eda.py` | 4 grafik + ringkasan: komoditas, saham, korelasi, produksi |
| Regresi | `src/02_regresi.py` | Prediksi konsumsi BBM per unit-hari: Linear vs Random Forest |
| Klasifikasi | `src/03_klasifikasi.py` | Prediksi unit gagal dalam 7 hari (predictive maintenance): Logistik vs Random Forest |
| Forecasting | `src/04_forecasting.py` | Ramalan harga nikel & batubara: Naive, SES, Holt, Holt-Winters, backtest 2023–2024 |
| NLP | `src/05_nlp.py` | Sentimen leksikon Bahasa Indonesia pada headline asli + frekuensi kata + topik |
| Dashboard eksekutif | `src/06_dashboard.py` | Satu file HTML mandiri merangkum semua modul |
| ML API | `src/api.py` | FastAPI: prediksi BBM, skor risiko gagal, ramalan harga |
| Docker | `Dockerfile`, `docker-compose.yml` | Image API siap `docker compose up --build` |

## Cara menjalankan

```
pip install -r requirements.txt
cd src
python3 ambil_data.py          # opsional, data mentah sudah disertakan
python3 siapkan_data.py
python3 buat_data_simulasi.py
python3 01_eda.py
python3 02_regresi.py
python3 03_klasifikasi.py
python3 04_forecasting.py
python3 05_nlp.py
python3 06_dashboard.py
```

API lokal (dari folder project):

```
uvicorn src.api:app --port 8000
# buka http://localhost:8000/docs
```

API lewat Docker:

```
docker compose up --build
```

Catatan: Dockerfile dan compose disertakan dan mengikuti pola standar;
API-nya sendiri sudah diverifikasi berjalan (uji TestClient: semua
endpoint 200, validasi input 422). Build image tidak diuji di
lingkungan pembuat project ini karena Docker tidak tersedia di sana.

## Hasil terverifikasi (dari output yang disertakan)

- **EDA**: nikel US$14.849/mt (Jan 2015) ke US$15.445/mt (Des 2024) dengan
  puncak masa lonjakan logam baterai; batubara memuncak pada krisis
  energi 2022. Pada data simulasi, produksi hari hujan 38% lebih rendah.
- **Regresi BBM**: regresi linear R2 0,954 (RMSE 20,3 liter) dan Random
  Forest R2 0,954 — nyaris sama, karena hubungan pada data memang
  dominan linear; jam operasi adalah fitur terpenting.
- **Klasifikasi risiko gagal**: regresi logistik AUC 0,834 mengalahkan
  Random Forest (0,792) — masuk akal karena fungsi risiko pada generator
  data berbentuk logistik. F1 pada ambang 0,5 rendah (0,30): sinyal
  getaran, suhu, dan hari sejak servis lebih cocok dipakai sebagai skor
  peringkat risiko daripada vonis ya/tidak.
- **Forecasting**: pada backtest 2023–2024, model Naive dan SES
  mengalahkan Holt dan Holt-Winters pada kedua komoditas (MAPE nikel
  55%, batubara 156%). Harga jatuh tajam setelah puncak krisis 2022 dan
  tidak ada model sederhana yang mengikutinya dengan baik. Ramalan 2025
  memakai model terbaik hasil backtest (nikel datar di sekitar
  US$15.445/mt dengan interval 80% US$13.514–17.376).
- **NLP**: 20 headline asli terbagi 7 positif, 7 negatif, 6 netral;
  kata tersering nikel (10x), produksi (8x), harga (6x). Tema dominan:
  kebijakan kuota RKAB dan gerak harga.

## Keterbatasan

1. Telemetri dan produksi adalah simulasi; metrik model operasional
   menunjukkan pola pipeline yang benar, bukan kinerja pada data asli.
2. Forecasting memakai deret harga nominal bulanan dan model
   exponential smoothing dasar; ARIMA, model dengan variabel eksogen
   (stok LME, permintaan Tiongkok), dan data harian adalah langkah
   lanjutan yang wajar.
3. NLP adalah baseline leksikon pada 20 judul: tidak menangkap konteks
   dan ironi. Langkah lanjutan: korpus lebih besar dan model terlatih
   (mis. IndoBERT).
4. Pembagian latih/uji regresi dan klasifikasi acak per baris; untuk
   data operasional asli, pembagian berbasis waktu lebih tepat karena
   baris unit yang sama muncul di banyak hari.
