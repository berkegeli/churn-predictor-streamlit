
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE  # pip install imbalanced-learn
from sklearn.neural_network import MLPClassifier


df = pd.read_csv('C:/Users/BERK/PycharmProjects/churn_project/dataset/churn.csv')


df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())


binary_mapping = {
    'gender': {'Male': 1, 'Female': 0},
    'Partner': {'Yes': 1, 'No': 0},
    'Dependents': {'Yes': 1, 'No': 0},
    'PhoneService': {'Yes': 1, 'No': 0},
    'PaperlessBilling': {'Yes': 1, 'No': 0}
}

for col, mapping in binary_mapping.items():
    df[col] = df[col].map(mapping)


no_internet_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in no_internet_cols:
    df[col] = df[col].replace('No internet service', 'No')
    df[col] = df[col].map({'Yes': 1, 'No': 0})


df = pd.get_dummies(df, columns=[
    'Contract', 'PaymentMethod', 'MultipleLines', 'InternetService'
], drop_first=True)


df['AvgMonthlyCharges'] = df['TotalCharges'] / (df['tenure'].replace(0, 1))  # tenure=0 için bölme hatası önleme
df['LongTenure'] = (df['tenure'] > 24).astype(int)
df['HighSpender'] = (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75)).astype(int)


df.drop(['customerID', 'MultipleLines_No phone service', 'InternetService_No'], axis=1, inplace=True)


# Sınıf dengesizliği
sns.countplot(x='Churn', data=df)
plt.title('Churn Dağılımı (0=Kalmış, 1=Ayrılmış)')
plt.show()

# Korelasyon analizi
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title('Korelasyon Matrisi')
plt.show()


# MODELLEME

X = df.drop('Churn', axis=1)
y = df['Churn']


smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    scale_pos_weight=len(y_res[y_res==0])/len(y_res[y_res==1])  # Class imbalance düzeltme
)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 0.9]
}

grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_xgb = grid_search.best_estimator_
y_pred = best_xgb.predict(X_test)

print("XGBoost Sonuçları:")
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, best_xgb.predict_proba(X_test)[:,1]))





plt.figure(figsize=(10,6))
feat_importances = pd.Series(best_xgb.feature_importances_, index=X.columns)
feat_importances.nlargest(15).plot(kind='barh')
plt.title('XGBoost Özellik Önemleri')
plt.show()


import shap
from sklearn.cluster import KMeans


plt.figure(figsize=(10,6))
feat_importances = pd.Series(best_xgb.feature_importances_, index=X.columns)
feat_importances.nlargest(15).plot(kind='barh')
plt.title('XGBoost Özellik Önemleri')
plt.show()


import shap

# SHAP başlat
shap.initjs()

# SHAP değerlerini hesapla
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test)


if isinstance(explainer.expected_value, (list, np.ndarray)):
    expected_val = explainer.expected_value[0]  # binary için çoğunlukla [0]
else:
    expected_val = explainer.expected_value


shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test)


shap.decision_plot(expected_val, shap_values[:3], X_test.iloc[:3])



X_segmentation = df[['tenure', 'MonthlyCharges', 'AvgMonthlyCharges', 'LongTenure', 'HighSpender']]

# KMeans ile 4 segment belirliyoruz
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_segmentation)

# Segmentlerin dağılımını görselleştirme
plt.figure(figsize=(8, 6))
sns.countplot(x='Cluster', data=df)
plt.title('Müşteri Segmentleri')
plt.show()

# Segment merkezlerini görselleştirelim
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=X_segmentation.columns)
plt.figure(figsize=(10, 6))
cluster_centers.T.plot(kind='bar')
plt.title('KMeans Segment Merkezleri')
plt.show()

# Segmentler arasındaki özellik farklarını görmek
plt.figure(figsize=(10,6))
sns.boxplot(x='Cluster', y='tenure', data=df)
plt.title('Müşteri Segmentlerine Göre Tenure Dağılımı')
plt.show()


import pickle

import pickle

with open("C:/Users/BERK/PycharmProjects/churn_project/model/xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)


