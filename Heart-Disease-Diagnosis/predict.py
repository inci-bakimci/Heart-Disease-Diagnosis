import joblib
import pandas as pd
import numpy as np

# 1. Modeli ve Özellik İsimlerini Yükle
model = joblib.load("models/heart_model_xgboost.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# 2. Test İçin Rastgele Bir "Yeni Hasta" Verisi Oluştur (Ham Veri)
# Bu veriler henüz özellik mühendisliğinden geçmedi!
new_patient_raw = pd.DataFrame([{
    'Age': 65,                      # İleri yaş
    'Sex': 'M',
    'ChestPainType': 'ASY',         # Asemptomatik (En riskli tip)
    'RestingBP': 150,               # Yüksek tansiyon
    'Cholesterol': 310,             # Yüksek kolesterol
    'FastingBS': 1,                 # Şeker var
    'RestingECG': 'ST',             # Anormal EKG
    'MaxHR': 110,                   # Düşük maksimum kalp hızı
    'ExerciseAngina': 'Y',          # Egzersizle gelen ağrı (Kritik)
    'Oldpeak': 2.5,                 # ST depresyonu (Yüksek risk)
    'ST_Slope': 'Flat'
}])

print("📊 Yeni hasta verisi alındı. İşleniyor...")

# 3. ÖN İŞLEME FONKSİYONU (Tahmin anında kullanılacak mini-pipeline)
def prepare_single_prediction(df):
    df_res = df.copy()
    
    # Klinik Skorlar (Train'deki Modül 2 mantığı)
    df_res['RPP'] = (df_res['RestingBP'] * df_res['MaxHR']) / 100
    angina_map = {'Y': 1, 'N': 0}
    df_res['DTS_Simulated'] = 1 - (5 * df_res['Oldpeak']) - (4 * df_res['ExerciseAngina'].map(angina_map))
    df_res['HR_Efficiency'] = df_res['MaxHR'] / (220 - df_res['Age'])
    df_res['Age_Oldpeak'] = df_res['Age'] * df_res['Oldpeak']
    df_res['HighChol'] = (df_res['Cholesterol'] > 200).astype(int)
    
    # Yaş Grubu (Train'deki Modül 2 mantığı)
    df_res['AgeGroup_Optimized'] = pd.cut(df_res['Age'], bins=[0, 45, 55, 120], labels=['Young', 'Middle', 'Senior+'])
    df_res['MetabolicRisk'] = ((df_res['FastingBS'] == 1) & (df_res['HighChol'] == 1)).astype(int)
    
    # Eksik Veri İşaretleme (Train'deki Modül 3 mantığı)
    df_res['Cholesterol_Is_Missing'] = (df_res['Cholesterol'] == 0).astype(int)
    
    # One-Hot Encoding ve Kolon Hizalama
    df_final = pd.get_dummies(df_res)
    df_final = df_final.reindex(columns=feature_names, fill_value=0)
    
    return df_final

# 4. Tahmini Gerçekleştir
processed_data = prepare_single_prediction(new_patient_raw)
prediction = model.predict(processed_data)
probability = model.predict_proba(processed_data)

# 5. Sonucu Yazdır
print("\n" + "="*30)
print("🩺 TAHMİN SONUCU")
print("="*30)
status = "KALP HASTALIĞI RİSKİ VAR" if prediction[0] == 1 else "RİSK DÜŞÜK / SAĞLIKLI"
print(f"Durum: {status}")
print(f"Risk Olasılığı: %{probability[0][1]*100:.2f}")
print("="*30)