"""
استكشاف الداتا + تدريب موديل ريقريشن لتوقع تكلفة التأمين الطبي
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ==========================================
# 1) تحميل الداتا
# ==========================================
df = pd.read_csv('insurance.csv')
print("="*60)
print("📊 شكل الداتا:", df.shape)
print("="*60)
print("\nأول 5 صفوف:")
print(df.head())
print("\nمعلومات الأعمدة:")
print(df.info())
print("\nالقيم المفقودة:")
print(df.isnull().sum())
print("\nإحصاءات وصفية:")
print(df.describe())

# ==========================================
# 2) EDA - رسومات استكشافية
# ==========================================
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# توزيع التكلفة (المتغير الهدف)
axes[0,0].hist(df['charges'], bins=40, color='steelblue', edgecolor='black')
axes[0,0].set_title('Distribution of Charges (Target)', fontsize=12, fontweight='bold')
axes[0,0].set_xlabel('Charges ($)')

# العمر vs التكلفة
axes[0,1].scatter(df['age'], df['charges'], alpha=0.4, c='coral')
axes[0,1].set_title('Age vs Charges')
axes[0,1].set_xlabel('Age')
axes[0,1].set_ylabel('Charges ($)')

# BMI vs التكلفة ملوّن بالمدخن
colors = df['smoker'].map({'yes': 'red', 'no': 'green'})
axes[0,2].scatter(df['bmi'], df['charges'], c=colors, alpha=0.5)
axes[0,2].set_title('BMI vs Charges (Red=Smoker, Green=Non)')
axes[0,2].set_xlabel('BMI')

# مدخن vs غير مدخن
sns.boxplot(data=df, x='smoker', y='charges', ax=axes[1,0], palette=['green','red'])
axes[1,0].set_title('Smoker Effect on Charges')

# عدد الأطفال vs التكلفة
sns.boxplot(data=df, x='children', y='charges', ax=axes[1,1], color='skyblue')
axes[1,1].set_title('Children vs Charges')

# المنطقة vs التكلفة
sns.boxplot(data=df, x='region', y='charges', ax=axes[1,2])
axes[1,2].set_title('Region vs Charges')
axes[1,2].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=120, bbox_inches='tight')
print("\n✅ حفظ الرسومات في eda_plots.png")

# ==========================================
# 3) تجهيز الداتا للتدريب
# ==========================================
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
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# ==========================================
# 4) تدريب 3 موديلات والمقارنة
# ==========================================
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42)
}

results = {}
print("\n" + "="*60)
print("🤖 تدريب الموديلات والمقارنة")
print("="*60)

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
    print(f"  R²   = {r2:.4f}")

# اختيار الأفضل
best_name = max(results, key=lambda k: results[k]['r2'])
best_pipe = results[best_name]['pipeline']
print(f"\n🏆 الموديل الأفضل: {best_name} (R² = {results[best_name]['r2']:.4f})")

# ==========================================
# 5) أهمية الفيتشرز
# ==========================================
if hasattr(best_pipe.named_steps['model'], 'feature_importances_'):
    feature_names = (list(best_pipe.named_steps['prep']
                         .named_transformers_['cat']
                         .get_feature_names_out(categorical))
                    + numeric)
    importances = best_pipe.named_steps['model'].feature_importances_
    
    fig, ax = plt.subplots(figsize=(10, 6))
    idx = np.argsort(importances)
    ax.barh([feature_names[i] for i in idx], importances[idx], color='teal')
    ax.set_title(f'Feature Importance — {best_name}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=120, bbox_inches='tight')
    print("\n✅ حفظ feature_importance.png")
    
    print("\nترتيب الفيتشرز:")
    for i in idx[::-1]:
        print(f"  {feature_names[i]:25s}: {importances[i]:.4f}")

# ==========================================
# 6) رسم: التنبؤ vs الفعلي
# ==========================================
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

# ==========================================
# 7) حفظ الموديل
# ==========================================
joblib.dump(best_pipe, 'insurance_model.pkl')
print(f"\n💾 تم حفظ الموديل في insurance_model.pkl")

# ==========================================
# 8) مثال على التنبؤ
# ==========================================
print("\n" + "="*60)
print("📌 مثال تنبؤ:")
print("="*60)
sample = pd.DataFrame([{
    'age': 35, 'sex': 'male', 'bmi': 28.5,
    'children': 2, 'smoker': 'no', 'region': 'southeast'
}])
pred = best_pipe.predict(sample)[0]
print(f"شخص عمره 35، رجال، BMI=28.5، عنده طفلين، مو مدخن")
print(f"➡️  التكلفة المتوقعة: ${pred:,.2f}")

sample2 = pd.DataFrame([{
    'age': 35, 'sex': 'male', 'bmi': 28.5,
    'children': 2, 'smoker': 'yes', 'region': 'southeast'
}])
pred2 = best_pipe.predict(sample2)[0]
print(f"\nنفس الشخص لكن مدخن:")
print(f"➡️  التكلفة المتوقعة: ${pred2:,.2f}")
print(f"\n💡 الفرق بسبب التدخين: ${pred2-pred:,.2f}")
