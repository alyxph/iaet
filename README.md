# IAET-SIM Machine Learning API

Layanan inferensi Machine Learning berbasis **FastAPI** dan **XGBoost** untuk mengklasifikasi tingkat kemahiran bahasa Inggris mahasiswa (`beginner`, `intermediate`, `advanced`) pada platform IAET-SIM.

---

## 📋 Fitur & Arsitektur
- **Model:** XGBoost Classifier (`model_iaet_xgboost.joblib`)
- **Input:** Skor persentase (0–100%) untuk Listening, Structure (SWE), dan Reading
- **Output:** Tingkat kemahiran akhir (`grade`), status prediksi (`prediction`), dan level masing-masing subbab (`sub_levels`)
- **Integrasi Backend:** Siap dikonsumsi langsung oleh Express backend (`gradePredictor.js`)

---

## 🚀 Setup & Menjalankan Lokal

### 1. Buat Virtual Environment & Install Dependencies
Gunakan Python 3.10 (disarankan agar kompatibel dengan versi model training `scikit-learn==1.6.1`):

```bash
# Masuk ke direktori
cd iaet-ml-api

# Buat virtual environment
py -3.10 -m venv venv

# Aktivasi venv
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate

# Install dependensi
pip install -r requirements.txt
```

### 2. Jalankan Pengujian Otomatis
```bash
python test_api.py
```

### 3. Jalankan Server API
```bash
uvicorn app:app --port 8000 --reload
```

Dokumentasi Interaktif Swagger UI dapat diakses di:
👉 `http://localhost:8000/docs`

---

## 📡 Endpoint

### 1. Health Check
- **GET** `/health` atau `/`
- Respons:
  ```json
  {
    "status": "healthy",
    "service": "IAET-SIM ML API",
    "model_loaded": true
  }
  ```

### 2. Prediksi Kemahiran
- **POST** `/predict`
- **Request Body:**
  ```json
  {
    "listening_score": 60,
    "structure_score": 70,
    "reading_score": 86,
    "overall_score": 72.0
  }
  ```
- **Response Body:**
  ```json
  {
    "grade": "advanced",
    "prediction": "advanced",
    "sub_levels": {
      "listening": "Intermediate",
      "structure": "Intermediate",
      "reading": "Advanced"
    },
    "input_scores": {
      "listening": 60.0,
      "structure": 70.0,
      "reading": 86.0
    }
  }
  ```

---

## 🐳 Menjalankan dengan Docker

```bash
# Build image
docker build -t iaet-ml-api .

# Run container
docker run -d -p 8000:8000 --name iaet-ml-service iaet-ml-api
```

---

## ☁️ Deploy ke Cloud (Render.com)

1. Buat repository baru di GitHub bernama `iaet-ml-api` dan push isi folder ini.
2. Buka [dashboard.render.com](https://dashboard.render.com/) -> klik **New +** -> **Web Service**.
3. Hubungkan repository `iaet-ml-api`.
4. Konfigurasi:
   - **Environment:** `Python`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Setelah live, salin URL service (contoh: `https://iaet-ml-api.onrender.com/predict`) dan pasang di `backend/.env` pada project utama:
   ```env
   ML_API_URL=https://iaet-ml-api.onrender.com/predict
   ```
