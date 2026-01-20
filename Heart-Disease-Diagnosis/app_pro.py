import streamlit as st
import requests  # API'ye bağlanmak için gerekli

st.set_page_config(page_title="Pro Kalp Analiz Paneli", layout="wide")

st.title("🏥 Merkezi Risk Analiz Sistemi (API Destekli)")

# API Adresini Tanımla (FastAPI'nin çalıştığı adres)
API_URL = "http://127.0.0.1:8000/predict"

# 1. Kullanıcıdan Verileri Al (Sol Panel)
with st.sidebar:
    st.header("Hasta Veri Girişi")
    age = st.number_input("Yaş", 18, 100, 50)
    sex = st.selectbox("Cinsiyet", ["M", "F"])
    cp = st.selectbox("Göğüs Ağrısı", ["ASY", "ATA", "NAP", "TA"])
    bp = st.number_input("Kan Basıncı", 80, 200, 120)
    chol = st.number_input("Kolesterol", 100, 500, 200)
    fbs = st.selectbox("Şeker > 120", [0, 1])
    ecg = st.selectbox("EKG", ["Normal", "ST", "LVH"])
    hr = st.slider("Maks. Kalp Hızı", 60, 220, 150)
    angina = st.selectbox("Egzersiz Anjinası", ["Y", "N"])
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# 2. Analiz Butonu
if st.button("API üzerinden Sorgula"):
    # API'nin beklediği JSON formatını hazırla
    payload = {
        "Age": age, "Sex": sex, "ChestPainType": cp, "RestingBP": bp,
        "Cholesterol": chol, "FastingBS": fbs, "RestingECG": ecg,
        "MaxHR": hr, "ExerciseAngina": angina, "Oldpeak": oldpeak, "ST_Slope": slope
    }
    
    with st.spinner('Merkezi sunucuya bağlanılıyor...'):
        try:
            # API'ye POST isteği gönder
            response = requests.post(API_URL, json=payload)
            result = response.json()
            
            if result["status"] == "success":
                st.subheader(f"Teşhis: {result['diagnosis']}")
                st.metric("Risk Skoru", f"%{result['risk_score']}")
                
                if result["risk_score"] > 50:
                    st.error("Kritik Seviye: Lütfen uzman doktora yönlendirin.")
                else:
                    st.success("Normal Seviye: Belirgin bir risk saptanmadı.")
        except Exception as e:
            st.error(f"API Sunucusuna bağlanılamadı! Lütfen uvicorn'un çalıştığından emin olun. Hata: {e}")

#İki Sistemi Aynı Anda Çalıştır
#Bu mimariyi test etmek için iki ayrı terminal açmalısın:

##Terminal 1 (Backend): uvicorn api:app --reload (Modelin burada bekliyor)

##Terminal 2 (Frontend): streamlit run app_pro.py (Arayüzün buradan API'ye bağlanıyor)