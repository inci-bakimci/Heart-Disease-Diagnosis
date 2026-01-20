🏥 Kalp Hastalığı Teşhis Sistemi (End-to-End ML Pipeline)
Bu proje, kalp hastalığı riskini tahmin etmek için geliştirilmiş, yüksek doğruluklu bir makine öğrenmesi sistemidir. Proje; veri ön işlemeden model eğitimine, API geliştirmeden kullanıcı arayüzüne kadar uçtan uca bir mimariye sahiptir.

🚀 Proje Hakkında Genel Bakış
Proje kapsamında 7 farklı algoritma üzerinde çalışılmış ve tıbbi teşhislerde kritik olan Sınıf Dengesi (Hasta/Sağlıklı ayrımı) gözetilerek Özel Yapılandırılmış XGBoost modeli şampiyon seçilmiştir.

Doğruluk (Accuracy): %88

Duyarlılık (Recall - Class 1): %89

Duyarlılık (Recall - Class 0): %85

🛠️ Teknik Mimari ve Çalıştırma
Sistem iki ana modülden oluşmaktadır: Backend (API) ve Frontend (Streamlit Arayüzü).

Sistemi Başlatma Adımları:
Bu mimariyi test etmek için iki ayrı terminal açılmalıdır:

Terminal 1 (Backend - FastAPI): Modelin istekleri beklediği sunucu.

Bash

uvicorn api:app --reload
Terminal 2 (Frontend - Streamlit): Kullanıcının veri girişi yaptığı arayüz.

Bash

streamlit run app_pro.py
📂 Proje Yapısı (Directory Structure)
Plaintext

Heart-Disease-Diagnosis/
├── dataset/           # Analizde kullanılan heart.csv veri seti
├── models/            # Eğitilmiş ve kaydedilmiş (.joblib/.pkl) modeller
├── research/          # Veri ön işleme adımları ve model geliştirme (Jupyter Notebooks)
├── api.py             # FastAPI backend kodları
├── app_pro.py         # Streamlit frontend (Arayüz) kodları
├── train.py           # Modelin uçtan uca eğitim ve pipeline kodları
├── predict.py         # Model tahmini için kullanılan test scripti
├── requirements.txt   # Gerekli kütüphaneler listesi
└── README.md          # Proje dökümantasyonu
🧪 Model Geliştirme Süreci
Veri Ön İşleme: Eksik verilerin (Cholesterol 0 değerleri) median yöntemiyle sızıntısız doldurulması.

Özellik Mühendisliği: Kalp sağlığına yönelik klinik skorların (DTS, RPP) modele entegre edilmesi.

Optimizasyon: GridSearchCV ile hiperparametrelerin en iyi dengeyi verecek şekilde ayarlanması.
