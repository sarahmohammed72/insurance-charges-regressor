"""
Insurance Cost Prediction — AutoGluon Version (الطريقة الموصى بها في المشروع)
=============================================================================
شغّل هذا السكربت لتدريب الموديل تلقائياً بـ AutoGluon.

التثبيت:
    pip install autogluon.tabular
"""
import pandas as pd
from autogluon.tabular import TabularPredictor

# 1) تحميل الداتا
df = pd.read_csv('insurance.csv')
print(f"Data shape: {df.shape}")

# 2) تقسيم تدريب/اختبار
train_data = df.sample(frac=0.8, random_state=42)
test_data  = df.drop(train_data.index).reset_index(drop=True)

# 3) تدريب AutoGluon - يجرّب عشرات الموديلات ويعمل ensemble
predictor = TabularPredictor(
    label='charges',
    problem_type='regression',
    eval_metric='r2',
    path='AutogluonModels/insurance'
).fit(
    train_data,
    time_limit=300,           # 5 دقائق تدريب
    presets='best_quality'    # أعلى جودة (يأخذ وقت أكثر لكن يعطي نتايج أحسن)
)

# 4) التقييم
print("\n" + "="*60)
print("Leaderboard:")
print("="*60)
print(predictor.leaderboard(test_data, silent=True))

# 5) أمثلة تنبؤ
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

print("\n💾 الموديل محفوظ في: AutogluonModels/insurance/")
print("💡 لتحميل الموديل لاحقاً:")
print("   from autogluon.tabular import TabularPredictor")
print("   predictor = TabularPredictor.load('AutogluonModels/insurance/')")
