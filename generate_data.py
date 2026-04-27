"""
يولّد ملف insurance.csv بنفس بنية الداتا المشهورة في Kaggle.
الداتا الأصلية: Machine Learning with R - Brett Lantz
الأعمدة: age, sex, bmi, children, smoker, region, charges
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 1338

# توليد الفيتشرز بتوزيعات قريبة من الداتا الأصلية
ages = np.random.randint(18, 65, n)
sexes = np.random.choice(['male', 'female'], n)
bmis = np.clip(np.random.normal(30.66, 6.1, n), 15.96, 53.13).round(3)
children = np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.43, 0.24, 0.18, 0.11, 0.03, 0.01])
smokers = np.random.choice(['yes', 'no'], n, p=[0.205, 0.795])
regions = np.random.choice(['southwest', 'southeast', 'northwest', 'northeast'], n)

# توليد charges بمعادلة قريبة من اللي تطلع من الداتا الأصلية:
# المدخن يدفع أكثر بكثير، الـ BMI العالي مع التدخين يضاعف، العمر يزيد التكلفة
charges = []
for i in range(n):
    base = 2500 + ages[i] * 250 + children[i] * 500
    if smokers[i] == 'yes':
        base += 15000
        if bmis[i] > 30:
            base += 18000  # تأثير قوي للتدخين مع السمنة
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
