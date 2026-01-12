import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import IsolationForest
from pathlib import Path
import os

print("=" * 60)
print("CIC-IDS2017 資料集預處理流程")
print("=" * 60)

# ============================================================
# Step 1: 載入所有資料集
# ============================================================
print("\n📂 Step 1: 載入資料集...")

file_paths = [
    './csv/Monday-WorkingHours.pcap_ISCX.csv',
    './csv/Tuesday-WorkingHours.pcap_ISCX.csv',
    './csv/Wednesday-workingHours.pcap_ISCX.csv',
    './csv/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    './csv/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    './csv/Friday-WorkingHours-Morning.pcap_ISCX.csv',
    './csv/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    './csv/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    './csv/FTP-BruteForce.csv'
]

datasets = []
for filename in os.listdir("../csv"):
    try:
        print(f"  載入: {filename}")
        df = pd.read_csv(f"./csv/{filename}", encoding='utf-8', encoding_errors='replace')
        df.columns = df.columns.str.strip()  # 清理欄位名稱
        datasets.append(df)
        print(f"       ✓ 形狀: {df.shape}, 標籤: {df['Label'].nunique()} 類")
    except FileNotFoundError:
        print(f"       ✗ 檔案不存在: {filename}")
    except Exception as e:
        print(f"       ✗ 錯誤: {e}")

if not datasets:
    raise ValueError("❌ 沒有成功載入任何資料集!")

# ============================================================
# Step 2: 合併資料集
# ============================================================
print("\n🔗 Step 2: 合併資料集...")
df_combined = pd.concat(datasets, ignore_index=True)

# 保留標籤
labels = df_combined['Label'].str.replace('�', '-', regex=False).copy()

print(f"✅ 合併後資料: {df_combined.shape}")
print(f"\n📊 標籤分布:")
print(labels.value_counts())

# ============================================================
# Step 3: 特徵準備
# ============================================================
print("\n🛠️  Step 3: 特徵準備...")

# 移除非特徵欄位
non_feature_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Label']
df_features = df_combined.drop(columns=non_feature_cols, errors='ignore')

# 提取數值特徵
X = df_features.select_dtypes(include=[np.number])
print(f"  原始特徵維度: {X.shape}")

# 處理異常值
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)
X = np.clip(X, -1e9, 1e9)

print(f"  清理後特徵維度: {X.shape}")

# ============================================================
# Step 4: IsolationForest 異常偵測 (可選)
# ============================================================
print("\n🔍 Step 4: IsolationForest 異常偵測...")

contamination_rate = 0.05  # 預期異常比例
clf = IsolationForest(
    contamination=contamination_rate,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print(f"  訓練 IsolationForest (contamination={contamination_rate})...")
clf.fit(X)

# 預測: -1 為異常, 1 為正常
predictions = clf.predict(X)
anomaly_if = np.where(predictions == 1, 0, 1)  # 轉換為 0=正常, 1=異常

anomaly_count = anomaly_if.sum()
anomaly_ratio = anomaly_count / len(df_combined) * 100

print(f"  ✅ 偵測完成!")
print(f"  🚨 異常數量: {anomaly_count:,} / {len(df_combined):,} ({anomaly_ratio:.2f}%)")

# ============================================================
# Step 5: 輸出結果
# ============================================================
print("\n💾 Step 5: 儲存處理後資料...")

# 組合結果
output = X.copy()
output['anomaly_if'] = anomaly_if
output['Label'] = labels.values

# 儲存主要輸出檔案
output_path = "../output_anomaly.csv"
output.to_csv(output_path, index=False)
print(f"  ✅ 已儲存: {output_path}")

# 額外儲存統計資訊
stats = {
    'total_samples': len(df_combined),
    'total_features': X.shape[1],
    'anomaly_if_count': int(anomaly_count),
    'anomaly_if_ratio': float(anomaly_ratio),
    'label_distribution': labels.value_counts().to_dict()
}

import json
with open('../preprocessing_stats.json', 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)
print(f"  ✅ 已儲存統計: preprocessing_stats.json")

# 選擇性儲存模型
model_path = "../isolation_forest_model.joblib"
joblib.dump(clf, model_path)
print(f"  ✅ 已儲存模型: {model_path}")

print("\n" + "=" * 60)
print("✨ 預處理完成!")
print("=" * 60)
print("\n📌 下一步:")
print("  1. 使用 'output_anomaly.csv' 訓練 Autoencoder")
print("  2. 'anomaly_if' 欄位為 IsolationForest 的參考標記")
print("  3. 'Label' 欄位為真實標籤,可用於評估")