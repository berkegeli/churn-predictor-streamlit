import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
with open("model/xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)


st.set_page_config(page_title="Churn Prediction App", layout="centered")

st.title("📊 Müşteri Kaybı Tahmin Uygulaması")
st.markdown("Bu uygulama, bir müşterinin hizmeti bırakma (churn) olasılığını tahmin eder.")

st.header("1️⃣ Kullanıcı Bilgilerini Girin")

gender = st.selectbox("Cinsiyet", ["Male", "Female"])
senior_citizen = st.selectbox("Yaşlı Müşteri (60+)?", [0, 1])
partner = st.selectbox("Eşi Var mı?", [0, 1])
dependents = st.selectbox("Bağımlı Kişisi Var mı?", [0, 1])
tenure = st.slider("Kullanım Süresi (ay)", 0, 72, 24)
monthly_charges = st.number_input("Aylık Ücret (₺)", min_value=0.0, value=75.0)
total_charges = st.number_input("Toplam Harcama (₺)", min_value=0.0, value=2000.0)

st.subheader("İnternet ve Sözleşme Bilgileri")
online_security = st.selectbox("Online Güvenlik", [0, 1])
online_backup = st.selectbox("Online Yedekleme", [0, 1])
device_protection = st.selectbox("Cihaz Koruma", [0, 1])
tech_support = st.selectbox("Teknik Destek", [0, 1])
streaming_tv = st.selectbox("TV Yayını", [0, 1])
streaming_movies = st.selectbox("Film Yayını", [0, 1])
paperless_billing = st.selectbox("Faturasız Ödeme", [0, 1])
contract_one_year = st.selectbox("1 Yıllık Sözleşme?", [0, 1])
contract_two_year = st.selectbox("2 Yıllık Sözleşme?", [0, 1])
payment_electronic = st.selectbox("Elektronik Çek Ödemesi?", [0, 1])
payment_credit_card = st.selectbox("Otomatik Kredi Kartı?", [0, 1])
payment_mailed = st.selectbox("Posta ile Ödeme?", [0, 1])
internet_fiber = st.selectbox("Fiber İnternet?", [0, 1])
multiple_lines = st.selectbox("Çoklu Hat?", [0, 1])

# Ek Özellikler
avg_monthly = total_charges / tenure if tenure > 0 else 0
long_tenure = 1 if tenure > 24 else 0
high_spender = 1 if monthly_charges > 80 else 0

# Veriyi hazırlama
data = {
    'gender': [1 if gender == 'Male' else 0],
    'SeniorCitizen': [senior_citizen],
    'Partner': [partner],
    'Dependents': [dependents],
    'tenure': [tenure],
    'PhoneService': [1],  # Sabit değer
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection],
    'TechSupport': [tech_support],
    'StreamingTV': [streaming_tv],
    'StreamingMovies': [streaming_movies],
    'PaperlessBilling': [paperless_billing],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'Contract_One year': [contract_one_year],
    'Contract_Two year': [contract_two_year],
    'PaymentMethod_Credit card (automatic)': [payment_credit_card],
    'PaymentMethod_Electronic check': [payment_electronic],
    'PaymentMethod_Mailed check': [payment_mailed],
    'MultipleLines_Yes': [multiple_lines],
    'InternetService_Fiber optic': [internet_fiber],
    'AvgMonthlyCharges': [avg_monthly],
    'LongTenure': [long_tenure],
    'HighSpender': [high_spender]
}

df = pd.DataFrame(data)

st.header("2️⃣ Tahmin Sonucu")

if st.button("Tahmin Et"):
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    if prediction == 1:
        st.error("⚠️ Müşteri **CHURN** olasılığı yüksek!")
    else:
        st.success("✅ Müşteri **kalma** olasılığı yüksek.")

    st.metric("Tahmin Skoru (Churn Olasılığı)", f"{probability:.2%}")

    # Risk seviyesi
    st.subheader("🔍 Risk Seviyesi")
    if probability > 0.75:
        st.warning("🚨 Yüksek Risk")
    elif probability > 0.5:
        st.info("⚠️ Orta Seviye Risk")
    else:
        st.success("🟢 Düşük Risk")

    # Alarm sistemi (örnek mantık)
    if probability > 0.8:
        st.error("🔔 ALARM: Bu müşteri özel ilgi gerektiriyor!")

    # Grafikler
    st.subheader("📈 Örnek Churn Oranı Grafiği")
    fig1, ax1 = plt.subplots()
    sns.barplot(x=["Churn", "Stay"], y=[probability, 1 - probability], ax=ax1, palette="Set2")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Olasılık")
    st.pyplot(fig1)

    st.subheader("📊 Model Başarı Grafiği (Örnek)")
    # Örnek ROC eğrisi simülasyonu (gerçek ROC değil)
    fig2, ax2 = plt.subplots()
    fpr = [0, 0.1, 0.2, 0.3, 1]
    tpr = [0, 0.4, 0.7, 0.9, 1]
    ax2.plot(fpr, tpr, marker='o')
    ax2.set_title("ROC Curve")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    st.pyplot(fig2)
import streamlit as st
import requests

st.title("Customer Churn Tahmin")

feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")

if st.button("Tahmin Et"):
    url = "http://localhost:8000/predict"
    payload = {"feature1": feature1, "feature2": feature2}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        st.success(f"Tahmin sonucu: {result['prediction']}")
    else:
        st.error("Tahmin yapılırken hata oluştu.")



