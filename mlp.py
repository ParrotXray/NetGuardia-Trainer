"""
MLP 訓練階段：只用 Autoencoder 偵測到的異常樣本訓練分類器
生成 mlp_attack_classifier.h5, label_encoder.pkl
"""
import pandas as pd
import numpy as np
from keras import Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

print("=" * 50)
print("🧠 Step 3: MLP 分類器訓練")
print("=" * 50)

# === 1️⃣ 讀取 Autoencoder 輸出 ===
df = pd.read_csv("output_autoencoder.csv")
df.columns = df.columns.str.strip()

print(f"✅ 載入資料: {df.shape}")

# === 2️⃣ 檢查欄位 ===
if 'is_anomaly' not in df.columns:
    raise KeyError("❌ 找不到 is_anomaly 欄位，請先執行 Autoencoder 訓練。")

if 'Label' not in df.columns:
    raise KeyError("❌ 找不到 Label 欄位，請確認資料完整性。")

# === 3️⃣ 只用異常樣本訓練 MLP ===
df_anomaly = df[df['is_anomaly'] == 1].copy()
print(f"🚨 異常樣本數量: {len(df_anomaly)} / {len(df)}")
print(f"📋 異常樣本標籤分布:\n{df_anomaly['Label'].value_counts()}")

# === 4️⃣ 準備特徵與標籤 ===
# 移除所有非特徵欄位
exclude_cols = ['Label', 'anomaly_score', 'is_anomaly', 'anomaly_if']
X = df_anomaly.drop(columns=exclude_cols, errors='ignore')
y = df_anomaly['Label']

print(f"🔢 特徵維度: {X.shape}")

# === 5️⃣ 清理數值 ===
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
X = np.clip(X, -1e9, 1e9)
print("✅ 資料清理完成")

# === 6️⃣ 使用 Autoencoder 的 scaler（關鍵！）===
scaler = joblib.load("scaler_ae.pkl")
X_scaled = scaler.transform(X)
print("✅ 已使用 Autoencoder 的 scaler 標準化特徵")

# === 7️⃣ 標籤編碼 ===
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
print(f"📦 Label 編碼完成，共 {len(encoder.classes_)} 類別：")
for idx, label in enumerate(encoder.classes_):
    count = (y_encoded == idx).sum()
    print(f"  {idx}: {label} ({count} samples)")

# === 8️⃣ 分割資料集 ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)
print(f"\n📊 訓練集: {X_train.shape}, 測試集: {X_test.shape}")

# === 9️⃣ 建立 MLP 模型 ===
mlp = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(np.unique(y_encoded)), activation='softmax')
])

mlp.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

print("\n📝 MLP 模型結構:")
mlp.summary()

# === 🔟 訓練 MLP ===
print("\n🚀 開始訓練 MLP...")
history = mlp.fit(
    X_train, y_train, 
    epochs=30, 
    batch_size=256, 
    validation_data=(X_test, y_test), 
    verbose=1
)

# === 11️⃣ 評估模型 ===
loss, acc = mlp.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ 測試集準確率: {acc:.4f}")
print(f"📉 測試集損失: {loss:.4f}")

# 詳細評估報告
y_pred = np.argmax(mlp.predict(X_test, verbose=0), axis=1)
print("\n📊 分類報告:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# === 12️⃣ 混淆矩陣 ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=encoder.classes_, 
            yticklabels=encoder.classes_)
plt.title('MLP Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('mlp_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("📊 已保存混淆矩陣: mlp_confusion_matrix.png")

# === 13️⃣ 訓練曲線 ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
ax1.plot(history.history['accuracy'], label='Train Acc')
ax1.plot(history.history['val_accuracy'], label='Val Acc')
ax1.set_title('MLP Accuracy Curve')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss
ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Val Loss')
ax2.set_title('MLP Loss Curve')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mlp_training_curves.png', dpi=150, bbox_inches='tight')
print("📊 已保存訓練曲線: mlp_training_curves.png")

plt.show()

# === 14️⃣ 預測全部異常樣本並輸出 ===
pred_all = np.argmax(mlp.predict(X_scaled, verbose=0), axis=1)
df_anomaly["predicted_label"] = encoder.inverse_transform(pred_all)
df_anomaly.to_csv("output_mlp.csv", index=False)
print("\n💾 已輸出含預測結果的檔案: output_mlp.csv")

# === 15️⃣ 儲存模型與編碼器 ===
mlp.save("mlp_attack_classifier.h5")
joblib.dump(encoder, "label_encoder.pkl")
print("✅ 已保存模型: mlp_attack_classifier.h5")
print("✅ 已保存編碼器: label_encoder.pkl")

print("\n" + "=" * 50)
print("✅ MLP 訓練完成！")
print("=" * 50)