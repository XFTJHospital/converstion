# train_model.py  —— 单文件合并版

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib

# ====================================
# 1. 读取数据
# ====================================
data = pd.read_csv("rHCC_training_v2.csv")

# ====================================
# 2. 编码类别变量
# ====================================
data_encoded = data.copy()
label_encoders = {}

categorical_cols = [
    "Age", 
    "Protrusion_from_surface", 
    "Child_Pugh_grade", 
    "Hemodynamic_instability", 
    "CSPH"
]

for col in categorical_cols:
    le = LabelEncoder()
    data_encoded[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# 编码目标变量（治疗方案）
target_le = LabelEncoder()
data_encoded["Treatment_strategy"] = target_le.fit_transform(data_encoded["Treatment_strategy"])

# ====================================
# 3. 拆分训练集和验证集
# ====================================
X = data_encoded.drop(columns=["Treatment_strategy"])
y = data_encoded["Treatment_strategy"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ====================================
# 4. 训练 XGBoost 模型
# ====================================
model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=len(np.unique(y)),
    learning_rate=0.05,
    n_estimators=300,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)

# ====================================
# 5. 模型评估
# ====================================
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
report = classification_report(y_val, y_pred, target_names=target_le.classes_)
conf_mat = confusion_matrix(y_val, y_pred)

print("\n✅ Model training complete!")
print(f"Accuracy: {acc:.3f}")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", conf_mat)

# ====================================
# 6. 保存模型（使用 XGBoost 原生格式，避免环境兼容问题）
# ====================================
model.save_model("rHCC_xgb_model.json")

# 其余组件仍然用 joblib 保存
joblib.dump(target_le, "rHCC_target_encoder.joblib")
joblib.dump(label_encoders, "rHCC_feature_encoders.joblib")
joblib.dump(X.columns.tolist(), "rHCC_feature_names.pkl")

print("✅ All model components have been saved successfully (JSON format for model).")
