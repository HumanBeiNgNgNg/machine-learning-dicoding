import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Load raw dataset
df = pd.read_csv("ML Workflow/house_prices.csv")

# --- STEP 1: Duplicated Data ---
df = df.drop_duplicates()

# --- STEP 2: Missing Values ---
missing_values = df.isnull().sum()

# Separate columns with >= 1000 missing values and < 1000 missing values
less = missing_values[missing_values < 1000].index
over = missing_values[missing_values >= 1000].index

# Drop columns with too many missing values (>= 1000)
df = df.drop(columns=over)

# Fill numeric missing values with median
numeric_features = df[df.columns.intersection(less)].select_dtypes(include=['number']).columns
df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())

# Fill categorical missing values with mode
categorical_features = df[df.columns.intersection(less)].select_dtypes(include=['object']).columns
for column in categorical_features:
    df[column] = df[column].fillna(df[column].mode()[0])


# --- STEP 3: Save Cleaned Dataset into ML Workflow/ ---
df.to_csv("ML Workflow/cleaned_house_prices.csv", index=False)
print("Success! Created 'ML Workflow/cleaned_house_prices.csv'.")