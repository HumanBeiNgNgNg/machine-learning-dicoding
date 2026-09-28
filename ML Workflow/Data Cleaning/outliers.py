import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("ML Workflow/house_prices.csv")

numeric_features = df.select_dtypes(include=['number']).columns

for feature in numeric_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[feature])
    plt.title(f'Box Plot of {feature}')
    plt.show()

"""
Nilai yang berada di bawah batas bawah atau di atas batas atas dianggap sebagai outlier

Ada dua pilihan yang biasa dilakukan untuk mengatasi:
1. Anda dapat memilih untuk menghapus outlier.
2. Menggantinya dengan nilai yang lebih moderat (seperti batas terdekat), atau menerapkan transformasi.


Langkah-langkah umum untuk mendeteksi dan menangani outlier menggunakan metode IQR.

1. Menghitung IQR, Q1, dan Q3
    a. Q1 (Quartile 1): Nilai di persentil ke-25 data.
    b. Q3 (Quartile 3): Nilai di persentil ke-75 data.
    c. IQR: Rentang antara Q3 dan Q1 (IQR = Q3 - Q1).

2. Menentukan Batas Bawah dan Batas Atas
    a. Batas Bawah: Q1 - 1.5 * IQR
    b. Batas Atas: Q3 + 1.5 * IQR

"""
# Contoh sederhana untuk mengidentifikasi outliers menggunakan IQR
Q1 = df[numeric_features].quantile(0.25)
Q3 = df[numeric_features].quantile(0.75)
IQR = Q3 - Q1


# Selanjutnya kita filter data untuk menghapus baris yang mengandung outlier.

# Filter dataframe untuk hanya menyimpan baris yang tidak mengandung outliers pada kolom numerik
non_outlier_condition = ~(
    (df[numeric_features] < (Q1 - 1.5 * IQR)) | 
    (df[numeric_features] > (Q3 + 1.5 * IQR))
).any(axis=1)

df_filtered_numeric = df.loc[non_outlier_condition, numeric_features]
 
# Menggabungkan kembali dengan kolom kategorikal
categorical_features = df.select_dtypes(include=['object']).columns
df = pd.concat(
    [df_filtered_numeric, df.loc[non_outlier_condition, categorical_features]],
    axis=1
)


"""
    PRO TIPS     

Jika Anda tidak ingin menghapus outliers seperti contoh di atas, silakan gunakan metode agregasi seperti berikut.


```
median = df['column_name'].median()
df['column_name'] = df['column_name'].apply(lambda x: median if x < (Q1 - 1.5  IQR) or x > (Q3 + 1.5  IQR) else x)

```
atau

```
# Mengganti outlier dengan nilai batas terdekat
df['column_name'] = df['column_name'].apply(lambda x: (Q1 - 1.5  IQR) if x < lower_bound else (Q3 + 1.5  IQR) if x > (Q3 + 1.5 * IQR) else x)
 
```

"""
