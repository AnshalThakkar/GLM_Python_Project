

import pandas as pd
import numpy as np
import statsmodels.api as sm

print("--- 1. Libraries Imported ---")


data = {
    'PolicyID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Age': [25, 45, 30, 50, 22, 60, 21, 35, 28, 42],
    'Region': ['North', 'South', 'North', 'South', 'North', 'South', 'North', 'South', 'North', 'South'],
    'Claim_Count': [0, 0, 1, 0, 2, 0, 1, 0, 0, 1],
    'Exposure': [1.0, 1.0, 0.5, 1.0, 0.8, 1.0, 0.3, 1.0, 1.0, 0.7]
}


df = pd.DataFrame(data)

print("--- 2. Data Loaded into Pandas DataFrame ---")
print(df)
print("\n")



age_bins = [18, 29, 39, 49, 61]
age_labels = ['18-29', '30-39', '40-49', '50-61']

df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=True)

print("--- 3. Feature Engineering Complete (New 'Age_Group' column) ---")
print(df)
print("\n")



y = df['Claim_Count']


X = pd.get_dummies(df['Age_Group'], drop_first=True, dtype=int) 


offset = np.log(df['Exposure'])


X = sm.add_constant(X)


poisson_model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)

results = poisson_model.fit()

print("--- 4. Model Fitted! ---")
print("\n")


# --- Step 5: Print The Results (Your "Analysis") ---
print("--- 5. MODEL RESULTS ---")
print(results.summary())