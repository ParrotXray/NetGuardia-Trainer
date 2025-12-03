"""
預處理階段：使用 IsolationForest 標記異常（可選）
這個步驟可以跳過，直接用原始 CSV 進入 Autoencoder
"""
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import IsolationForest

print("=" * 50)
print("📊 Step 1: 預處理 (IsolationForest)")
print("=" * 50)

# 讀取原始資料
df = pd.read_csv("Wednesday-workingHours.pcap_ISCX.csv")
df.columns = df.columns.str.strip()

print(f"✅ 載入資料: {df.shape}")
print(f"📋 標籤分布:\n{df['Label'].value_counts()}")

# 保留 Label，但從特徵中移除
labels = df['Label'].copy()
df_features = df.drop(columns=['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Label'], errors='ignore')

# 提取數值特徵
X = df_features.select_dtypes(include=[np.number])
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
X = np.clip(X, -1e9, 1e9)

print(f"🔢 特徵維度: {X.shape}")

# IsolationForest 標記異常（這只是參考，真正的異常偵測在 Autoencoder）
clf = IsolationForest(contamination=0.05, random_state=42)
clf.fit(X)
anomaly_if = np.where(clf.predict(X) == 1, 0, 1)

print(f"🚨 IsolationForest 偵測異常: {anomaly_if.sum()} / {len(df)}")

# 組合結果
output = X.copy()
output['anomaly_if'] = anomaly_if
output['Label'] = labels.values

# 儲存
output.to_csv("output_anomaly.csv", index=False)
print("💾 已輸出: output_anomaly.csv")
print("=" * 50)