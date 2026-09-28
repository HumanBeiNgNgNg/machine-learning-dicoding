import pandas as pd

train = pd.read_csv("ML Workflow/Iris.csv")
train.info()

# Menampilkan ringkasan informasi dari dataset
train.describe(include="all")

# Memeriksa jumlah nilai yang hilang di setiap kolom
missing_values = train.isnull().sum()
missing_values[missing_values > 0]

# Cara atasi missing value
"""
Pertama-tama, mari kita pisahkan kolom yang memiliki missing value lebih dari 75% dan kurang dari 75%.

"""

less = missing_values[missing_values < 1000].index
over = missing_values[missing_values >= 1000].index

"""
Berdasarkan hasil dari langkah sebelumnya, kita akan memutuskan bagaimana menangani nilai yang hilang dengan dua cara.

"""

# Cara pertama (Mengisi Nilai yang Hilang)
# Contoh mengisi nilai yang hilang dengan median untuk kolom numerik
numeric_features = train[less].select_dtypes(include=['number']).columns
train[numeric_features] = train[numeric_features].fillna(train[numeric_features].median())

# Contoh mengisi nilai yang hilang dengan mode untuk kolom kategori
kategorical_features = train[less].select_dtypes(include=['object']).columns
 
for column in kategorical_features:
    train[column] = train[column].fillna(train[column].mode()[0])

# Cara kedua (Menghapus Kolom dengan Banyak Nilai yang Hilang)
df = train.drop(columns=over)

"""
Terakhir, lakukan pemeriksaan terhadap data yang sudah melewati tahapan verifikasi missing value

"""

missing_values = df.isnull().sum()
missing_values[missing_values > 0]