"""
Insurance Cost Prediction - AutoGluon Version
==============================================
Run this script to train the model automatically with AutoGluon.

Installation:
    pip install autogluon.tabular
"""
import pandas as pd
from autogluon.tabular import TabularPredictor

# 1) Load data
df = pd.read_csv('insurance.csv')
print(f"Data shape: {df.shape}")

# 2) Train/test split
train_data = df.sample(frac=0.8, random_state=42)
test_data = df.drop(train_data.index).reset_index(drop=True)

# 3) Train AutoGluon - tries dozens of models and creates an ensemble
predictor = TabularPredictor(
    label='charges',
    problem_type='regression',
    eval_metric='r2',
    path='AutogluonModels/insurance'
).fit(
    train_data,
    time_limit=300,
    presets='best_quality'
)

# 4) Evaluation
print("\n" + "="*60)
print("Leaderboard:")
print("="*60)
print(predictor.leaderboard(test_data, silent=True))

# 5) Sample predictions
print("\n" + "="*60)
print("Example Predictions:")
print("="*60)

samples = pd.DataFrame([
    {'age': 35, 'sex': 'male',   'bmi': 28.5, 'children': 2, 'smoker': 'no',  'region': 'southeast'},
    {'age': 35, 'sex': 'male',   'bmi': 28.5, 'children': 2, 'smoker': 'yes', 'region': 'southeast'},
    {'age': 60, 'sex': 'female', 'bmi': 35.0, 'children': 0, 'smoker': 'yes', 'region': 'northeast'},
])
preds = predictor.predict(samples)
samples['predicted_charges'] = preds.values
print(samples)

print("\n[SAVED] Model saved to: AutogluonModels/insurance/")
print("To load the model later:")
print("   from autogluon.tabular import TabularPredictor")
print("   predictor = TabularPredictor.load('AutogluonModels/insurance/')")
