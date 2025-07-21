Bu proje, bir telekomünikasyon firmasının müşteri verileri üzerinden müşteri kaybını (churn) tahmin etmek amacıyla XGBoost sınıflandırma algoritması kullanılarak geliştirilmiştir. Modelin başarımı çeşitli metriklerle (F1-Score, ROC AUC) değerlendirilmiş ve SHAP analizleriyle yorumlanabilir hale getirilmiştir. Ayrıca proje, kullanıcı dostu bir arayüzle Streamlit kullanılarak sunulmuştur.

     Proje Amacı
Müşteri kaybını önceden tahmin ederek, şirketlerin zamanında aksiyon alabilmesini sağlamak. Bu amaçla:

    Veri ön işleme

Dengesiz sınıf sorununa SMOTE ile çözüm

XGBoost ile model eğitimi

SHAP ile açıklanabilirlik

KMeans ile müşteri segmentasyonu

Streamlit arayüzü ile canlı tahmin

    Kullanılan Teknolojiler
Python

pandas, numpy, matplotlib, seaborn

scikit-learn

XGBoost

imbalanced-learn (SMOTE)

SHAP

KMeans

Streamlit

       Değerlendirme Sonuçları
F1-Score: 0.85

ROC AUC: 0.92

Önemli Özellikler: Tenure, MonthlyCharges, Contract
