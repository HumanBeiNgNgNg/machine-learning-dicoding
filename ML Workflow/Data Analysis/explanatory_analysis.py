# Explanatory Data Analysis (ExDA)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# 1. Load the CLEANED dataset directly
df = pd.read_csv("ML Workflow/cleaned_house_prices.csv")

# 2. Encode categorical columns for correlation calculation
df_lencoder = df.copy()
category_features = df_lencoder.select_dtypes(include=['object']).columns

label_encoder = LabelEncoder()
for col in category_features:
    df_lencoder[col] = label_encoder.fit_transform(df_lencoder[col])


# ==========================================
# 1. General Correlation Matrix (Heatmap)
# ==========================================

plt.figure(figsize=(12, 10))
correlation_matrix = df_lencoder.corr()

sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()


# ==========================================
# 2. Targeted Correlation with Target (SalePrice)
# ==========================================

# Menghitung korelasi antara variabel target (SalePrice) dan semua variabel lainnya
target_corr = correlation_matrix['SalePrice']

# Mengurutkan hasil korelasi berdasarkan nilai mutlak
target_corr_sorted = target_corr.abs().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
target_corr_sorted.plot(kind='bar')
plt.title('Correlation with SalePrice')
plt.xlabel('Variables')
plt.ylabel('Correlation Coefficient')
plt.tight_layout()
plt.show()