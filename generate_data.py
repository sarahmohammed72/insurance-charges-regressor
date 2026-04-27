"""
Generates insurance.csv with the same structure as the well-known Kaggle dataset.
Original data: Machine Learning with R - Brett Lantz
Columns: age, sex, bmi, children, smoker, region, charges
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 1338

# Generate features with distributions close to original data
ages = np.random.randint(18, 65, n)
sexes = np.random.choice(['male', 'female'], n)
bmis = np.clip(np.random.normal(30.66, 6.1, n), 15.96, 53.13).round(3)
children = np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.43, 0.24, 0.18, 0.11, 0.03, 0.01])
smokers = np.random.choice(['yes', 'no'], n, p=[0.205, 0.795])
regions = np.random.choice(['southwest', 'southeast', 'northwest', 'northeast'], n)

# Generate charges with formula similar to original data:
# Smokers pay much more, high BMI with smoking compounds, age increases cost
charges = []
for i in range(n):
    base = 2500 + ages[i] * 250 + children[i] * 500
    if smokers[i] == 'yes':
        base += 15000
        if bmis[i] > 30:
            base += 18000  # Strong smoking + obesity interaction
    base += (bmis[i] - 25) * 100
    noise = np.random.normal(0, 2500)
    charges.append(max(1121.87, base + noise))

df = pd.DataFrame({
    'age': ages,
    'sex': sexes,
    'bmi': bmis,
    'children': children,
    'smoker': smokers,
    'region': regions,
    'charges': np.round(charges, 5)
})

df.to_csv('insurance.csv', index=False)
print(f"Created insurance.csv with {len(df)} rows")
print(df.head())
print(f"\nMean charges: ${df['charges'].mean():.2f}")
print(f"Median charges: ${df['charges'].median():.2f}")
