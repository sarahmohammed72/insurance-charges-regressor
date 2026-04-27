"""
Insurance Cost Prediction - Model Training Pipeline
Trains a regression model to predict medical insurance charges
based on demographic and health features.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 1) Load Data
df = pd.read_csv('insurance.csv')
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDescriptive Statistics:")
print(df.describe())

# 2) Exploratory Data Analysis
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

axes[0,0].hist(df['charges'], bins=40, color='steelblue', edgecolor='black')
axes[0,0].set_title('Distribution of Charges (Target)', fontsize=12, fontweight='bold')
axes[0,0].set_xlabel('Charges ($)')

axes[0,1].scatter(df['age'], df['charges'], alpha=0.4, c='coral')
axes[0,1].set_title('Age vs Charges')
axes[0,1].set_xlabel('Age')
axes[0,1].set_ylabel('Charges ($)')

colors = df['smoker'].map({'yes': 'red', 'no': 'green'})
axes[0,2].scatter(df['bmi'], df['charges'], c=colors, alpha=0.5)
axes[0,2].set_title('BMI vs Charges (Red=Smoker, Green=Non-Smoker)')
axes[0,2].set_xlabel('BMI')

sns.boxplot(data=df, x='smoker', y='charges', ax=axes[1,0])
axes[1,0].set_title('Smoker Effect on Charges')

sns.boxplot(data=df, x='children', y='charges', ax=axes[1,1], color='skyblue')
axes[1,1].set_title('Number of Children vs Charges')

sns.boxplot(data=df, x='region', y='charges', ax=axes[1,2])
axes[1,2].set_title('Region vs Charges')
axes[1,2].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=120, bbox_inches='tight')
print("\nEDA plots saved to eda_plots.png")

# 3) Prepare Data for Training
X = df.drop('charges', axis=1)
y = df['charges']

categorical = ['sex', 'smoker', 'region']
numeric = ['age', 'bmi', 'children']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), categorical)
], remainder='passthrough')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")

# 4) Train and Compare Models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42)
}

results = {}
print("\nTraining Models and Comparison")
print("-" * 40)

for name, model in models.items():
    pipe = Pipeline([
        ('prep', preprocessor),
        ('model', model)
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    results[name] = {'pipeline': pipe, 'mae': mae, 'rmse': rmse, 'r2': r2}
    print(f"\n{name}:")
    print(f"  MAE  = ${mae:,.2f}")
    print(f"  RMSE = ${rmse:,.2f}")
    print(f"  R2   = {r2:.4f}")

best_name = max(results, key=lambda k: results[k]['r2'])
best_pipe = results[best_name]['pipeline']
print(f"\nBest Model: {best_name} (R2 = {results[best_name]['r2']:.4f})")

# 5) Feature Importance
if hasattr(best_pipe.named_steps['model'], 'feature_importances_'):
    feature_names = (list(best_pipe.named_steps['prep']
                         .named_transformers_['cat']
                         .get_feature_names_out(categorical))
                    + numeric)
    importances = best_pipe.named_steps['model'].feature_importances_
    
    fig, ax = plt.subplots(figsize=(10, 6))
    idx = np.argsort(importances)
    ax.barh([feature_names[i] for i in idx], importances[idx], color='teal')
    ax.set_title(f'Feature Importance - {best_name}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=120, bbox_inches='tight')
    print("\nFeature importance plot saved to feature_importance.png")
    
    print("\nFeature Ranking:")
    for i in idx[::-1]:
        print(f"  {feature_names[i]:25s}: {importances[i]:.4f}")

# 6) Predictions vs Actual Plot
preds_best = best_pipe.predict(X_test)
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, preds_best, alpha=0.5, color='purple')
lims = [min(y_test.min(), preds_best.min()), max(y_test.max(), preds_best.max())]
ax.plot(lims, lims, 'r--', label='Perfect Prediction')
ax.set_xlabel('Actual Charges ($)')
ax.set_ylabel('Predicted Charges ($)')
ax.set_title(f'{best_name}: Predicted vs Actual')
ax.legend()
plt.tight_layout()
plt.savefig('predictions.png', dpi=120, bbox_inches='tight')

# 7) Save Model
joblib.dump(best_pipe, 'insurance_model.pkl')
print(f"\nModel saved to insurance_model.pkl")

# 8) Sample Predictions
print("\nSample Predictions:")
print("-" * 40)
sample = pd.DataFrame([{
    'age': 35, 'sex': 'male', 'bmi': 28.5,
    'children': 2, 'smoker': 'no', 'region': 'southeast'
}])
pred = best_pipe.predict(sample)[0]
print(f"Person: 35yo male, BMI=28.5, 2 children, non-smoker")
print(f"Predicted charges: ${pred:,.2f}")

sample2 = pd.DataFrame([{
    'age': 35, 'sex': 'male', 'bmi': 28.5,
    'children': 2, 'smoker': 'yes', 'region': 'southeast'
}])
pred2 = best_pipe.predict(sample2)[0]
print(f"\nSame person but smoker:")
print(f"Predicted charges: ${pred2:,.2f}")
print(f"\nSmoking premium: ${pred2-pred:,.2f}")
