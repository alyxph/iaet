from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 1. Inisialisasi FastAPI
app = FastAPI(
    title="IAET-SIM ML Classification API",
    description="API untuk memprediksi level kemahiran bahasa Inggris (Beginner, Intermediate, Advanced) menggunakan XGBoost.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Muat model dan komponen pendukung
MODEL_PATH = Path(__file__).resolve().parent / "model_iaet_xgboost.joblib"

try:
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    le = bundle["label_encoder"]
    LEVEL_COLS = bundle.get("fitur", ["reading_level", "SWE_level", "listening_level"])
except Exception as e:
    bundle = None
    model = None
    le = None
    LEVEL_COLS = ["reading_level", "SWE_level", "listening_level"]
    print(f"Warning: Gagal memuat model (.joblib): {e}")

# 3. Ambang batas sesuai dataset (50 x 4.51 = 225.5 poin maks per subbab)
# Batas persentase:
# - Beginner      : 0% - 30.0%  (<= 67.65 poin -> titik pisah 69.905)
# - Intermediate  : 32% - 80.0% (72.16 - 180.4 poin -> titik pisah 182.655)
# - Advanced      : 82% - 100%  (>= 184.91 poin)
BATAS_BEGINNER_MAKS = 69.905
BATAS_INTERMEDIATE_MAKS = 182.655
POIN_MAKS = 225.5

def skor_ke_level(persen: float) -> str:
    poin = (persen / 100.0) * POIN_MAKS
    if poin <= BATAS_BEGINNER_MAKS:
        return "Beginner"
    elif poin <= BATAS_INTERMEDIATE_MAKS:
        return "Intermediate"
    else:
        return "Advanced"

# 4. Skema request payload
class ScorePayload(BaseModel):
    listening_score: float = Field(..., ge=0, le=100, description="Skor listening dalam persen (0-100)")
    structure_score: float = Field(..., ge=0, le=100, description="Skor structure/SWE dalam persen (0-100)")
    reading_score: float = Field(..., ge=0, le=100, description="Skor reading dalam persen (0-100)")
    overall_score: float | None = Field(default=None, description="Skor total (opsional)")

@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "IAET-SIM ML API",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict_grade(payload: ScorePayload):
    if model is None or le is None:
        raise HTTPException(
            status_code=500,
            detail="Model ML belum dimuat dengan benar di server."
        )

    # 1. Klasifikasi level masing-masing subbab
    sub_levels = {
        "listening_level": skor_ke_level(payload.listening_score),
        "SWE_level": skor_ke_level(payload.structure_score),
        "reading_level": skor_ke_level(payload.reading_score),
    }

    # 2. Siapkan input DataFrame sesuai urutan fitur model
    df_input = pd.DataFrame([sub_levels])

    # 3. Prediksi XGBoost & decode label encoder
    pred_code = model.predict(df_input[LEVEL_COLS])
    pred_label = le.inverse_transform(pred_code.astype(int))[0]

    grade_normalized = str(pred_label).strip().lower()

    return {
        "grade": grade_normalized,
        "prediction": grade_normalized,
        "sub_levels": {
            "listening": sub_levels["listening_level"],
            "structure": sub_levels["SWE_level"],
            "reading": sub_levels["reading_level"],
        },
        "input_scores": {
            "listening": payload.listening_score,
            "structure": payload.structure_score,
            "reading": payload.reading_score,
        }
    }
