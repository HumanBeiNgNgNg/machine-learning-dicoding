import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("ML Workflow/house_prices.csv")

# Identify categorical features
category_features = df.select_dtypes(include=['object']).columns

label_encoder = LabelEncoder()
df_lencoder = pd.DataFrame(df)

for col in category_features:
    df_lencoder[col] = label_encoder.fit_transform(df[col])


# =========================
# Missing Value Analysis
# =========================

# Menghitung jumlah dan persentase missing values di setiap kolom
missing_values = df_lencoder.isnull().sum()
missing_percentage = (missing_values / len(df_lencoder)) * 100
 
missing_data = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percentage
}).sort_values(by='Missing Values', ascending=False)
 
print(missing_data[missing_data['Missing Values'] > 0])  # Menampilkan kolom dengan missing values


# =========================
# Distribution Analysis
# =========================

# Menghitung jumlah variabel
num_vars = df_lencoder.shape[1]
 
# Menentukan jumlah baris dan kolom untuk grid subplot
n_cols = 4  # Jumlah kolom yang diinginkan
n_rows = -(-num_vars // n_cols)  # Ceiling division untuk menentukan jumlah baris
 
# Membuat subplot
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
 
# Flatten axes array untuk memudahkan iterasi jika diperlukan
axes = axes.flatten()
 
# Plot setiap variabel
for i, column in enumerate(df_lencoder.columns):
    df_lencoder[column].hist(ax=axes[i], bins=20, edgecolor='black')
    axes[i].set_title(column)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
 
# Menghapus subplot yang tidak terpakai (jika ada)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
 
# Menyesuaikan layout agar lebih rapi
plt.tight_layout()
plt.show()


# ==============================
# Selected Column Distribution
# ==============================

# Visualisasi distribusi data untuk beberapa kolom
columns_to_plot = ['OverallQual', 'YearBuilt', 'LotArea', 'SaleType', 'SaleCondition']
 
plt.figure(figsize=(15, 10))
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df_lencoder[column], kde=True, bins=30)
    plt.title(f'Distribution of {column}')
 
plt.tight_layout()
plt.show()

"""
OverallQual: sebagian besar data terkonsentrasi di sekitar nilai tengah dengan pola multimodal (beberapa puncak), 
             dan sedikit miring ke kanan karena adanya ekor data hingga ke nilai tinggi.

YearBuilt: distribusi condong ke kiri (left-skewed), menunjukkan proporsi rumah yang dibangun lebih baru jauh lebih dominan 
           dibandingkan rumah-rumah lama.

LotArea: distribusi mendekati normal (unimodal, simetris), namun terdapat beberapa outlier yang signifikan pada sisi kanan 
         (lot area yang sangat besar).

SaleType: kategori tertentu mendominasi distribusi, menunjukkan preferensi atau pola tertentu dalam tipe penjualan.

SaleCondition: mirip dengan SaleType, beberapa kondisi penjualan lebih umum dibandingkan lainnya.

"""