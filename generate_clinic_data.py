
import geopandas as gpd
import pandas as pd
import numpy as np

# Load district codes from GeoJSON
uttarakhand_map = gpd.read_file('UTTARAKHAND_DISTRICTS.geojson')
district_codes = uttarakhand_map['dtcode11'].astype(int).tolist()
NUM_DISTRICTS = len(district_codes)
NUM_PATIENTS = 50000

print("Generating synthetic clinical data for Uttarakhand...")

np.random.seed(101)
clinical_data = {
    'Patient_ID': range(1001, 1001 + NUM_PATIENTS),
    'dtcode11': np.random.choice(district_codes, NUM_PATIENTS),
    'Age': np.random.randint(18, 85, NUM_PATIENTS),
    'Sex': np.random.choice(['Male', 'Female'], NUM_PATIENTS, p=[0.52, 0.48]),
    'BMI': np.clip(np.random.normal(26, 6, NUM_PATIENTS), 15, 50),
    'Has_Hypertension': np.random.choice([0, 1], NUM_PATIENTS, p=[0.7, 0.3]),
    'Has_High_Cholesterol': np.random.choice([0, 1], NUM_PATIENTS, p=[0.6, 0.4]),
    'Family_History_Diabetes': np.random.choice([0, 1], NUM_PATIENTS, p=[0.85, 0.15])
}
clinical_df = pd.DataFrame(clinical_data)

risk_score = (
    0.04 * (clinical_df['Age'] - 50) +
    0.15 * (clinical_df['BMI'] - 25) +
    0.8 * clinical_df['Has_Hypertension'] +
    0.6 * clinical_df['Has_High_Cholesterol'] +
    1.0 * clinical_df['Family_History_Diabetes'] +
    np.random.normal(0, 0.5, NUM_PATIENTS) - 2.5
)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

disease_probability = sigmoid(risk_score)
clinical_df['Has_Disease'] = (np.random.rand(NUM_PATIENTS) < disease_probability).astype(int)

clinical_df.to_csv('synthetic_clinical_dataset.csv', index=False)

print("\n--- Synthetic Clinical Data Generation Complete ---")
print(clinical_df.head())