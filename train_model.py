# ====================================
# train_model.py —— 全组件合并保存版（兼容 Streamlit）
# ====================================

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
data = pd.read_csv("rHCC.csv")  # ⚠️ 根据你最新版本调整

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
# 6. 合并保存所有组件到一个文件
# ====================================
model_package = {
    "model": model,
    "feature_encoders": label_encoders,
    "target_encoder": target_le,
    "feature_names": X.columns.tolist()
}

joblib.dump(model_package, "rHCC_model_merged.pkl")

print("\n💾 All components (model + encoders) saved in 'rHCC_model_merged.pkl'")
