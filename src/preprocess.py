## 2. src/preprocess.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# 1. Load dataset
data_path = "/content/Crop_recommendation.csv"
df = pd.read_csv(data_path)

print("Original Data Shape:", df.shape)
print("Columns:", df.columns)

# 2. Handle missing values (if any)
df = df.dropna()  # drop rows with missing values
print("After handling missing values:", df.shape)

# 3. Encode categorical target (Crop names)
if 'label' in df.columns:
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    print("Label encoding done. Example:", df['label'].head())

# 4. Normalize numerical features
features = df.drop(columns=['label'])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
scaled_df['label'] = df['label']

# 5. Save preprocessed dataset
os.makedirs("data/processed", exist_ok=True)
scaled_df.to_csv("data/processed/crop_recommendation_clean.csv", index=False)

print("✅ Preprocessing Completed. Clean dataset saved at data/processed/")
