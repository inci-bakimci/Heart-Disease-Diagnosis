from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# API Tanımlama
app = FastAPI(title="Kalp Sağlığı Risk Analiz API", version="1.0")

# Model Yükleme
model = joblib.load("models/heart_model_xgboost.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# Gelen Verinin Formatı (Şema)
class PatientData(BaseModel):
    Age: int
    Sex: str
    ChestPainType: str
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: str
    MaxHR: int
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str

@app.post("/predict")
def predict_heart(patient: PatientData):
    # 1. Gelen veriyi işle (Preprocessing)
    df = pd.DataFrame([patient.dict()])
    
    # Özellik Mühendisliği (Basitleştirilmiş)
    df['RPP'] = (df['RestingBP'] * df['MaxHR']) / 100
    df['DTS_Simulated'] = 1 - (5 * df['Oldpeak']) - (4 * (1 if df['ExerciseAngina'].iloc[0] == 'Y' else 0))
    df['HighChol'] = (df['Cholesterol'] > 200).astype(int)
    
    # Kolon Hizalama
    df_final = pd.get_dummies(df).reindex(columns=feature_names, fill_value=0)
    
    # 2. Tahmin
    prediction = int(model.predict(df_final)[0])
    probability = float(model.predict_proba(df_final)[0][1])
    
    # 3. JSON Yanıtı
    return {
        "status": "success",
        "diagnosis": "Riskli" if prediction == 1 else "Sağlıklı",
        "risk_score": round(probability * 100, 2)
    }

# Çalıştırmak için: uvicorn api:app --reload