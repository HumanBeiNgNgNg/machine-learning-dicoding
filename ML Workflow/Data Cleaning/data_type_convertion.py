import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv("ML Workflow/house_prices.csv")

# Identify categorical features
category_features = df.select_dtypes(include=['object']).columns

print(df[category_features])


# One-Hot Encoding
df_one_hot = pd.get_dummies(df, columns=category_features)

print("Data setelah One-Hot Encoding:")
print(df_one_hot)


# Label Encoding
label_encoder = LabelEncoder()
df_lencoder = pd.DataFrame(df)

for col in category_features:
    df_lencoder[col] = label_encoder.fit_transform(df[col])

print("Data setelah Label Encoding:")
print(df_lencoder)


# Ordinal Encoding
ordinal_encoder = OrdinalEncoder()
df_ordinal = pd.DataFrame(df)

df_ordinal[category_features] = ordinal_encoder.fit_transform(
    df[category_features]
)

print("Data setelah Ordinal Encoding:")
print(df_ordinal)