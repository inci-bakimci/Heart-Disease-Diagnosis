import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Sayfa Ayarları ve Başlık
st.set_page_config(page_title="Kalp Riski Tahmin Sistemi", layout="wide", page_icon="🩺")

# 2. Modeli ve Metadata'yı Yükle (Cache kullanarak hızı artırıyoruz)
@st.cache_resource
def load_assets():
    model = joblib.load("models/heart_model_xgboost.pkl")
    features = joblib.load("models/feature_names.pkl")
    return model, features

model, feature_names = load_assets()

# 3. Yan Panel (Sidebar) - Kullanıcı Bilgileri
st.sidebar.header("👤 Hasta Bilgileri")

def get_user_input():
    age = st.sidebar.slider("Yaş", 18, 95, 50)
    sex = st.sidebar.selectbox("Cinsiyet", ["M", "F"])
    cp = st.sidebar.selectbox("Göğüs Ağrısı Tipi", ["ASY", "ATA", "NAP", "TA"])
    bp = st.sidebar.number_input("Dinlenme Kan Basıncı (RestingBP)", 80, 200, 120)
    chol = st.sidebar.number_input("Kolesterol", 100, 600, 200)
    fbs = st.sidebar.selectbox("Açlık Kan Şekeri > 120 mg/dl", [0, 1])
    ecg = st.sidebar.selectbox("Resting EKG", ["Normal", "ST", "LVH"])
    max_hr = st.sidebar.slider("Maksimum Kalp Hızı", 60, 220, 150)
    angina = st.sidebar.selectbox("Egzersize Bağlı Anjina", ["Y", "N"])
    oldpeak = st.sidebar.number_input("Oldpeak (ST Depresyonu)", 0.0, 6.0, 1.0, step=0.1)
    slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])
    
    data = {
        'Age': age, 'Sex': sex, 'ChestPainType': cp, 'RestingBP': bp,
        'Cholesterol': chol, 'FastingBS': fbs, 'RestingECG': ecg,
        'MaxHR': max_hr, 'ExerciseAngina': angina, 'Oldpeak': oldpeak, 'ST_Slope': slope
    }
    return pd.DataFrame([data])

input_df = get_user_input()

# 4. Tahmin İçin Veriyi Hazırlama (Utils Mantığı)
def prepare_prediction(df):
    df_res = df.copy()
    # Özellik Mühendisliği (Train'deki ile birebir aynı)
    df_res['RPP'] = (df_res['RestingBP'] * df_res['MaxHR']) / 100
    angina_map = {'Y': 1, 'N': 0}
    df_res['DTS_Simulated'] = 1 - (5 * df_res['Oldpeak']) - (4 * df_res['ExerciseAngina'].map(angina_map))
    df_res['HR_Efficiency'] = df_res['MaxHR'] / (220 - df_res['Age'])
    df_res['Age_Oldpeak'] = df_res['Age'] * df_res['Oldpeak']
    df_res['HighChol'] = (df_res['Cholesterol'] > 200).astype(int)
    df_res['AgeGroup_Optimized'] = pd.cut(df_res['Age'], bins=[0, 45, 55, 120], labels=['Young', 'Middle', 'Senior+'])
    df_res['MetabolicRisk'] = ((df_res['FastingBS'] == 1) & (df_res['HighChol'] == 1)).astype(int)
    df_res['Cholesterol_Is_Missing'] = (df_res['Cholesterol'] == 0).astype(int)
    
    df_final = pd.get_dummies(df_res)
    df_final = df_final.reindex(columns=feature_names, fill_value=0)
    return df_final

# 5. Ana Ekran ve Tahmin Butonu
st.write("### 🏥 Yapay Zeka Destekli Kalp Hastalığı Risk Analizi")
st.info("Lütfen sol taraftaki panelden hasta verilerini giriniz ve 'Analizi Başlat' butonuna basınız.")

if st.button("🔍 Analizi Başlat"):
    processed_input = prepare_prediction(input_df)
    prediction = model.predict(processed_input)
    probability = model.predict_proba(processed_input)[0][1]
    
    # Görsel Sonuç Paneli
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Risk Olasılığı", f"%{probability*100:.2f}")
        
    with col2:
        if prediction[0] == 1:
            st.error("⚠️ Yüksek Risk Grubu")
        else:
            st.success("✅ Düşük Risk Grubu")

    # Risk Barı
    st.progress(float(probability))
    
    if probability > 0.80:
        st.warning("Not: Model bu vakada oldukça emin gözüküyor. Acil klinik muayene önerilir.")
        
# çalıştımak için: streamlit run .\app.