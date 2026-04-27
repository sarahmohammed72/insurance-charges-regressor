# 🚀 دليل التشغيل في VS Code

دليل خطوة بخطوة لتشغيل المشروع على جهازك باستخدام VS Code.

---

## 📋 المتطلبات قبل البدء

تأكد عندك:
- ✅ **Python 3.9+** مثبت ([نزّله من هنا](https://www.python.org/downloads/))
- ✅ **VS Code** مثبت
- ✅ **Git** مثبت

### Extensions مطلوبة في VS Code:
افتح VS Code واضغط `Ctrl+Shift+X` (أو `Cmd+Shift+X` للماك) وثبّت:
1. **Python** (من Microsoft)
2. **Jupyter** (من Microsoft)
3. **Pylance** (من Microsoft)

---

## 🔧 الخطوة 1: فتح المشروع

1. حمّل مجلد المشروع وفك ضغطه
2. افتح VS Code
3. `File` → `Open Folder` → اختر مجلد `insurance_project`
4. افتح Terminal داخل VS Code: ``Ctrl+` `` (أو من القائمة `Terminal` → `New Terminal`)

---

## 🐍 الخطوة 2: إنشاء بيئة افتراضية (Virtual Environment)

**ليش مهمة؟** عشان تعزل مكتبات المشروع عن بقية المشاريع.

### في الـ Terminal اكتب:

**على Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**على Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

> 💡 لما تشتغل تمام، بتشوف `(venv)` في بداية سطر الـ Terminal.

### ربط البيئة بـ VS Code:
1. اضغط `Ctrl+Shift+P` (أو `Cmd+Shift+P`)
2. اكتب: `Python: Select Interpreter`
3. اختر اللي فيه `./venv/bin/python` أو `.\venv\Scripts\python.exe`

---

## 📦 الخطوة 3: تثبيت المكتبات

في نفس الـ Terminal:

```bash
pip install -r requirements.txt
```

**انتظر شوي** (المكتبات حجمها ~200MB).

### (اختياري) لو تبي تستخدم AutoGluon:
```bash
pip install autogluon.tabular
```
> ⚠️ AutoGluon حجمه كبير (~2GB) ويأخذ وقت في التثبيت.

---

## 🏃 الخطوة 4: تشغيل التدريب

### الطريقة الأولى: من Terminal مباشرة

```bash
python train.py
```

بتشوف:
```
📊 شكل الداتا: (1338, 7)
...
🤖 تدريب الموديلات والمقارنة
...
🏆 الموديل الأفضل: Gradient Boosting (R² = 0.9553)
💾 تم حفظ الموديل في insurance_model.pkl
```

### الطريقة الثانية: من النوتبوك (مفضّلة للعرض)

1. افتح ملف `Insurance_Project.ipynb` في VS Code
2. أعلى يمين الملف، اضغط **Select Kernel** → اختر `venv`
3. اضغط على زر **Run All** أو شغّل كل خلية بـ `Shift+Enter`

---

## 🌐 الخطوة 5: تشغيل الـ API (Deployment)

في الـ Terminal:

```bash
uvicorn app:app --reload
```

بتشوف:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### جرّب الـ API:
1. افتح متصفح على: `http://127.0.0.1:8000/docs`
2. بتشوف واجهة Swagger UI تفاعلية
3. اضغط على `POST /predict` → `Try it out`
4. عدّل الـ JSON كذا:
```json
{
  "age": 35,
  "sex": "male",
  "bmi": 28.5,
  "children": 2,
  "smoker": "no",
  "region": "southeast"
}
```
5. اضغط **Execute** → بتشوف التنبؤ

> ⛔ لما تخلص: اضغط `Ctrl+C` في الـ Terminal لإيقاف الـ API.

---

## 📤 الخطوة 6: رفع المشروع على GitHub

### إنشاء Git repo:
```bash
git init
git add .
git commit -m "Initial commit: Insurance prediction ML project"
```

### إنشاء `.gitignore`:
أنشئ ملف `.gitignore` في المجلد وحط فيه:
```
venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
AutogluonModels/
```

### رفع على GitHub:
1. روح [github.com](https://github.com) وأنشئ repo جديد (مثلاً: `insurance-ml-project`)
2. **لا** تضيف README أو .gitignore (موجودين عندنا)
3. ارجع للـ Terminal:
```bash
git remote add origin https://github.com/USERNAME/insurance-ml-project.git
git branch -M main
git push -u origin main
```

---

## 🆘 مشاكل شائعة وحلولها

### ❌ "python command not found"
- **Windows:** استخدم `py` بدل `python`
- **Mac/Linux:** استخدم `python3`

### ❌ "pip install" يطلع خطأ
جرّب:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### ❌ النوتبوك ما يشتغل
1. تأكد إنك مختار الـ kernel الصحيح (venv)
2. لو ما طلع لك الخيار، شغّل في Terminal:
```bash
pip install ipykernel
python -m ipykernel install --user --name=venv
```

### ❌ "Module not found"
تأكد إنك في الـ venv ((venv) ظاهرة في Terminal)، ولو لا فعّلها مرة ثانية.

### ❌ Port 8000 already in use
شغّل بـ port ثاني:
```bash
uvicorn app:app --reload --port 8001
```

---

## 📁 ترتيب الملفات بعد التشغيل

```
insurance_project/
├── venv/                        ← البيئة الافتراضية (لا ترفعها لـ Git)
├── insurance.csv                ← الداتا
├── insurance_model.pkl          ← الموديل (يطلع بعد train.py)
├── eda_plots.png               ← رسومات EDA
├── feature_importance.png      ← رسم أهمية الفيتشرز
├── predictions.png             ← رسم التنبؤ
├── train.py                    ← سكربت التدريب
├── train_autogluon.py          ← نسخة AutoGluon
├── app.py                      ← الـ API
├── Insurance_Project.ipynb     ← النوتبوك
├── README.md                   ← التقرير الكامل (للأستاذ)
├── requirements.txt            ← المكتبات
└── VSCODE_SETUP.md            ← هذا الملف
```

---

## 🎯 نصيحة أخيرة قبل التسليم

1. **شغّل النوتبوك من البداية للنهاية** عشان كل الـ outputs تظهر محفوظة
2. **خذ سكرين شوت** للـ API وهي شغالة (Swagger UI) وأضفها للـ README
3. **اختبر الـ Repo** على جهاز ثاني أو حذف الـ venv وأعد التثبيت — تأكد إنه يشتغل من الصفر
4. **حدّث README** في القسم الأخير بـ اسمك ولينك الـ Repo

بالتوفيق! 💪
