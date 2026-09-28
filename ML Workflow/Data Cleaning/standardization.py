import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("ML Workflow/house_prices.csv")

# Select numerical features
numeric_features = df.select_dtypes(include=['number']).columns

# Save original data for comparison
original_df = df.copy()

# Standardisasi fitur numerik
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Histogram Sebelum dan Setelah Standardisasi
plt.figure(figsize=(12, 5))

# Histogram Sebelum Standardisasi
plt.subplot(1, 2, 1)
sns.histplot(original_df[numeric_features[3]], kde=True)
plt.title("Histogram Sebelum Standardisasi")

# Histogram Setelah Standardisasi
plt.subplot(1, 2, 2)
sns.histplot(df[numeric_features[3]], kde=True)
plt.title("Histogram Setelah Standardisasi")

plt.show()