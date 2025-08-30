# AI for Sustainable Agriculture — Preprocessing Progress (30%)

## Dataset
- Name: Crop Recommendation Dataset
- Source: [Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- Features: N, P, K, Temperature, Humidity, pH, Rainfall
- Target: Crop (label)

## Preprocessing Steps Completed
1. Loaded dataset using pandas
2. Checked data info:
   - `.info()`
   - `.describe()`
   - `.isnull().sum()`
3. Handled missing values
4. Removed duplicate rows
5. Encoded categorical target column
6. Normalized numerical features using StandardScaler
7. Saved cleaned dataset as `data/processed/crop_recommendation_clean.csv`

## Next Steps
- Model training (ML algorithms)
- Model evaluation
- Prototype development
