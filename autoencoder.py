"""
Autoencoder 訓練階段（最終改良版 - 修正離群點過濾）：
- 修正離群點過濾邏輯，避免刪除所有樣本
- 逐欄位過濾，而非要求所有欄位都符合
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from scipy import stats

print("=" * 50)
print("🤖 Step 2: Autoencoder 訓練（最終改良版 v2）")
print("=" * 50)

print("TensorFlow version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))

# === 1️⃣ 讀資料 ===
df = pd.read_csv("output_anomaly.csv")
df.columns = df.columns.str.strip()

print(f"✅ 載入資料: {df.shape}")

# === 2️⃣ 保存標籤 ===
labels = df['Label'].copy()
print(f"📋 標籤分布:\n{labels.value_counts()}")

# === 3️⃣ 只用 BENIGN 訓練 Autoencoder ===
df_benign = df[df['Label'] == 'BENIGN'].copy()

# 移除所有非特徵欄位
exclude_cols = ['Label', 'anomaly_if']
X_train = df_benign.drop(columns=exclude_cols, errors='ignore')
X_train = X_train.select_dtypes(include=[np.number])

print(f"✅ BENIGN 樣本數（處理前）: {len(X_train)} / {len(df)}")
print(f"🔢 特徵維度: {X_train.shape}")

# === 4️⃣ 清理數值 ===
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_train = np.clip(X_train, -1e9, 1e9)

# === 🆕 5️⃣ 改良的離群點處理（平衡版）===
print("\n🔍 移除 BENIGN 離群點（平衡版 v4）...")

# 先備份
X_train_backup = X_train.copy()

# 策略：只使用整體 MSE + 極端值雙重過濾（不用 IQR）
# 計算每個樣本的標準化後平方誤差總和
X_train_normalized = (X_train - X_train.mean()) / (X_train.std() + 1e-8)
sample_mse = (X_train_normalized ** 2).sum(axis=1)

# 方法 1: 移除 MSE 最高的 3% 樣本（溫和）
mse_threshold = sample_mse.quantile(0.97)
mse_mask = sample_mse < mse_threshold

print(f"  📊 MSE 過濾: 移除前 3% 高 MSE 樣本")
print(f"     MSE 門檻: {mse_threshold:.2f}")
print(f"     保留: {mse_mask.sum()} / {len(X_train)}")

# 方法 2: 只移除有極端極端值的樣本（0.05% 和 99.95%）
extreme_mask = pd.Series([True] * len(X_train), index=X_train.index)
extreme_cols = []

for col in X_train.columns:
    # 只針對真正的極端值
    lower_extreme = X_train[col].quantile(0.0005)
    upper_extreme = X_train[col].quantile(0.9995)

    col_extreme = (X_train[col] < lower_extreme) | (X_train[col] > upper_extreme)

    if col_extreme.sum() > 0:
        extreme_mask = extreme_mask & ~col_extreme
        extreme_cols.append(col)

print(f"  📊 極端值過濾: {len(extreme_cols)} 個欄位有極端值")
print(f"     保留: {extreme_mask.sum()} / {len(X_train)}")

# 結合兩種方法（OR 邏輯：任一方法認為正常即保留）
# 只移除兩個方法都認為是離群點的樣本
final_mask = mse_mask | extreme_mask

X_train_clean = X_train[final_mask]

outliers_removed = len(X_train) - len(X_train_clean)
outlier_ratio = outliers_removed / len(X_train)

print(f"  ❌ 移除離群點: {outliers_removed} ({outlier_ratio:.2%})")
print(f"  ✅ 保留樣本數: {len(X_train_clean)}")

# 安全檢查：如果移除太多（>10%）或太少樣本，使用原始資料
if outlier_ratio > 0.10:
    print(f"  ⚠️ 離群點比例過高 ({outlier_ratio:.2%})，使用原始資料")
    X_train = X_train_backup
elif outlier_ratio < 0.005:  # 改成 0.5%
    print(f"  ℹ️ 離群點極少 ({outlier_ratio:.2%})，使用原始資料")
    X_train = X_train_backup
elif len(X_train_clean) < 10000:  # 至少保留 1 萬個樣本
    print(f"  ⚠️ 保留樣本數太少 ({len(X_train_clean)})，使用原始資料")
    X_train = X_train_backup
else:
    print(f"  ✅ 離群點處理成功")
    X_train = X_train_clean

print(f"\n📊 最終訓練樣本數: {len(X_train)}")

# === 6️⃣ 標準化 ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print(f"✅ 標準化完成，形狀: {X_train_scaled.shape}")

# === 7️⃣ 建立 Autoencoder ===
input_dim = X_train_scaled.shape[1]
print(f"\n🧩 Autoencoder 輸入維度: {input_dim}")

input_layer = layers.Input(shape=(input_dim,))

# Encoder
encoded = layers.Dense(256, activation='relu',
                      kernel_regularizer=regularizers.l2(0.001))(input_layer)
encoded = layers.BatchNormalization()(encoded)
encoded = layers.Dropout(0.2)(encoded)

encoded = layers.Dense(128, activation='relu',
                      kernel_regularizer=regularizers.l2(0.001))(encoded)
encoded = layers.BatchNormalization()(encoded)
encoded = layers.Dropout(0.2)(encoded)

encoded = layers.Dense(64, activation='relu',
                      kernel_regularizer=regularizers.l2(0.001))(encoded)
encoded = layers.BatchNormalization()(encoded)
encoded = layers.Dropout(0.1)(encoded)

# Bottleneck
bottleneck = layers.Dense(8, activation='relu',
                         kernel_regularizer=regularizers.l2(0.001))(encoded)

# Decoder
decoded = layers.Dense(64, activation='relu',
                      kernel_regularizer=regularizers.l2(0.001))(bottleneck)
decoded = layers.BatchNormalization()(decoded)
decoded = layers.Dropout(0.1)(decoded)

decoded = layers.Dense(128, activation='relu',
                      kernel_regularizer=regularizers.l2(0.001))(decoded)
decoded = layers.BatchNormalization()(decoded)
decoded = layers.Dropout(0.2)(decoded)

decoded = layers.Dense(256, activation='relu',
                      kernel_regularizer=regularizers.l2(0.001))(decoded)
decoded = layers.BatchNormalization()(decoded)
decoded = layers.Dropout(0.2)(decoded)

output_layer = layers.Dense(input_dim, activation='linear')(decoded)

autoencoder = models.Model(inputs=input_layer, outputs=output_layer)

optimizer = Adam(learning_rate=0.001)
autoencoder.compile(optimizer=optimizer, loss='mse')

print("\n📝 模型結構:")
autoencoder.summary()

# === 8️⃣ 設定 Callbacks ===
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# === 9️⃣ 訓練模型 ===
print("\n🚀 開始訓練...")
history = autoencoder.fit(
    X_train_scaled, X_train_scaled,
    epochs=100,
    batch_size=512,
    validation_split=0.15,
    callbacks=[early_stop, reduce_lr],
    shuffle=True,
    verbose=1
)

print(f"\n✅ 訓練完成，實際訓練 {len(history.history['loss'])} 個 epoch")

# === 🔟 用整份資料做重建誤差 ===
print("\n🔍 計算全部資料的重建誤差...")

df_all = df.drop(columns=exclude_cols, errors='ignore')
X_all = df_all.select_dtypes(include=[np.number])
X_all = X_all[X_train.columns]
X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0)
X_all = np.clip(X_all, -1e9, 1e9)

X_all_scaled = scaler.transform(X_all)

recon = autoencoder.predict(X_all_scaled, verbose=0)
mse = np.mean(np.square(X_all_scaled - recon), axis=1)

# === 11️⃣ 詳細診斷 ===
print("\n" + "=" * 50)
print("🔍 重建誤差診斷")
print("=" * 50)

mse_benign = mse[labels == 'BENIGN']
mse_attack = mse[labels != 'BENIGN']

print(f"\n📊 BENIGN 樣本 MSE:")
print(f"  - Mean: {mse_benign.mean():.6f}")
print(f"  - Std:  {mse_benign.std():.6f}")
print(f"  - Min:  {mse_benign.min():.6f}")
print(f"  - Max:  {mse_benign.max():.6f}")
print(f"  - Median: {np.median(mse_benign):.6f}")
print(f"  - 95th percentile: {np.percentile(mse_benign, 95):.6f}")
print(f"  - 99th percentile: {np.percentile(mse_benign, 99):.6f}")

print(f"\n🚨 Attack 樣本 MSE:")
print(f"  - Mean: {mse_attack.mean():.6f}")
print(f"  - Std:  {mse_attack.std():.6f}")
print(f"  - Min:  {mse_attack.min():.6f}")
print(f"  - Max:  {mse_attack.max():.6f}")
print(f"  - Median: {np.median(mse_attack):.6f}")

print(f"\n📈 MSE 比值 (Attack/BENIGN):")
if mse_benign.mean() > 0:
    print(f"  - Mean 比值: {mse_attack.mean() / mse_benign.mean():.2f}x")
if mse_benign.max() > 0:
    print(f"  - Max 比值: {mse_attack.max() / mse_benign.max():.2f}x")
print(f"  - Median 比值: {np.median(mse_attack) / np.median(mse_benign):.2f}x")

print(f"\n🎯 各攻擊類型 MSE:")
for attack_type in sorted(labels[labels != 'BENIGN'].unique()):
    mse_type = mse[labels == attack_type]
    count = len(mse_type)
    print(f"  {attack_type:20s}: Mean={mse_type.mean():.6f}, "
          f"Median={np.median(mse_type):.6f}, "
          f"Max={mse_type.max():.6f}, Count={count}")

# === 12️⃣ 多種門檻策略 ===
print("\n" + "=" * 50)
print("🎯 門檻策略比較")
print("=" * 50)

thresholds = {}

for p in [75, 80, 85, 90, 95, 99]:
    thresholds[f"All_P{p}"] = np.percentile(mse, p)

for p in [85, 90, 95, 97, 99, 99.5]:
    thresholds[f"BENIGN_P{p}"] = np.percentile(mse_benign, p)

for n in [2, 2.5, 3, 3.5]:
    thresholds[f"BENIGN_M+{n}S"] = mse_benign.mean() + n * mse_benign.std()

for n in [2, 3, 4, 5]:
    median = np.median(mse_benign)
    mad = np.median(np.abs(mse_benign - median))
    thresholds[f"BENIGN_Med+{n}MAD"] = median + n * mad

print(f"\n{'策略':<20} {'門檻值':<12} {'偵測攻擊':<15} {'誤報':<10} {'偵測率':<10} {'Precision':<10} {'F1':<10}")
print("-" * 100)

best_strategy = None
best_f1 = 0
all_results = []

for name, threshold in sorted(thresholds.items(), key=lambda x: x[1]):
    is_anomaly = (mse > threshold).astype(int)

    tp = ((labels != 'BENIGN') & (is_anomaly == 1)).sum()
    fp = ((labels == 'BENIGN') & (is_anomaly == 1)).sum()
    fn = ((labels != 'BENIGN') & (is_anomaly == 0)).sum()
    tn = ((labels == 'BENIGN') & (is_anomaly == 0)).sum()

    total_attacks = (labels != 'BENIGN').sum()
    detection_rate = tp / total_attacks if total_attacks > 0 else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = detection_rate
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"{name:<20} {threshold:<12.6f} {tp:>6}/{total_attacks:<7} {fp:<10} "
          f"{detection_rate:>8.2%}  {precision:>8.4f}  {f1:>8.4f}")

    all_results.append({
        'strategy': name,
        'threshold': threshold,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'detection_rate': detection_rate,
        'precision': precision,
        'f1': f1
    })

    if f1 > best_f1:
        best_f1 = f1
        best_strategy = name
        best_threshold = threshold
        best_results = all_results[-1]

print(f"\n🏆 推薦策略: {best_strategy}")
print(f"🎯 推薦門檻: {best_threshold:.6f}")
print(f"📊 F1-Score: {best_f1:.4f}")

# === 13️⃣ 使用推薦門檻 ===
threshold = best_threshold
is_anomaly = (mse > threshold).astype(int)

print(f"\n📊 使用推薦門檻的結果:")
print(f"  - 偵測到異常: {is_anomaly.sum()} / {len(df)}")

# === 14️⃣ 輸出結果 ===
output = X_all.copy()
output['anomaly_score'] = mse
output['is_anomaly'] = is_anomaly
output['Label'] = labels.values

output.to_csv("output_autoencoder.csv", index=False)
print("\n💾 已輸出: output_autoencoder.csv")

# === 15️⃣ 儲存模型 ===
autoencoder.save("autoencoder_cic_model.h5")
joblib.dump(scaler, "scaler_ae.pkl")
joblib.dump({
    'threshold': threshold,
    'strategy': best_strategy,
    'all_thresholds': thresholds,
    'best_results': best_results
}, "threshold_info.pkl")

print("✅ 已保存模型和門檻資訊")

# === 16️⃣ 視覺化（簡化版，9 張圖）===
fig = plt.figure(figsize=(20, 12))

# 1. 訓練損失
ax1 = plt.subplot(3, 3, 1)
ax1.plot(history.history['loss'], label='Train', linewidth=2)
ax1.plot(history.history['val_loss'], label='Val', linewidth=2)
ax1.set_title('Training Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. MSE 分佈
ax2 = plt.subplot(3, 3, 2)
ax2.hist(mse_benign, bins=100, alpha=0.7, label='BENIGN', color='green', density=True)
ax2.hist(mse_attack, bins=100, alpha=0.7, label='Attack', color='red', density=True)
ax2.axvline(threshold, color='black', linestyle='--', linewidth=2)
ax2.set_xlabel('MSE')
ax2.set_title('MSE Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. MSE 分佈（放大）
ax3 = plt.subplot(3, 3, 3)
max_display = np.percentile(mse, 98)
ax3.hist(mse_benign[mse_benign < max_display], bins=100, alpha=0.7, label='BENIGN', color='green', density=True)
ax3.hist(mse_attack[mse_attack < max_display], bins=100, alpha=0.7, label='Attack', color='red', density=True)
ax3.axvline(threshold, color='black', linestyle='--', linewidth=2)
ax3.set_xlabel('MSE')
ax3.set_title(f'MSE Distribution (Zoom < {max_display:.2f})')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 各攻擊類型 MSE
ax4 = plt.subplot(3, 3, 4)
attack_types = sorted(labels[labels != 'BENIGN'].unique())
mse_by_type = [mse_benign] + [mse[labels == at] for at in attack_types]
labels_plot = ['BENIGN'] + list(attack_types)
bp = ax4.boxplot(mse_by_type, labels=labels_plot, patch_artist=True)
ax4.axhline(threshold, color='red', linestyle='--', linewidth=2)
ax4.set_ylabel('MSE')
ax4.set_title('MSE by Type')
ax4.set_xticklabels(labels_plot, rotation=45, ha='right')
ax4.grid(True, alpha=0.3, axis='y')

# 5. F1-Score 比較
ax5 = plt.subplot(3, 3, 5)
top10 = sorted(all_results, key=lambda x: x['f1'], reverse=True)[:10]
names = [r['strategy'] for r in top10]
f1s = [r['f1'] for r in top10]
colors = ['gold' if r['strategy'] == best_strategy else 'steelblue' for r in top10]
ax5.barh(names, f1s, color=colors)
ax5.set_xlabel('F1-Score')
ax5.set_title('Top 10 Strategies')
ax5.grid(True, alpha=0.3, axis='x')

# 6. 混淆矩陣
ax6 = plt.subplot(3, 3, 6)
cm = np.array([[best_results['tn'], best_results['fp']],
               [best_results['fn'], best_results['tp']]])
im = ax6.imshow(cm, cmap='Blues')
ax6.set_xticks([0, 1])
ax6.set_yticks([0, 1])
ax6.set_xticklabels(['Pred Normal', 'Pred Attack'])
ax6.set_yticklabels(['True Normal', 'True Attack'])
for i in range(2):
    for j in range(2):
        ax6.text(j, i, f'{cm[i, j]:,}', ha="center", va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black", fontweight='bold')
ax6.set_title('Confusion Matrix')
plt.colorbar(im, ax=ax6)

# 7-9. 其他圖表（簡化）
ax7 = plt.subplot(3, 3, 7)
ax7.text(0.5, 0.5, f'Detection Rate\n{best_results["detection_rate"]:.2%}',
         ha='center', va='center', fontsize=20, fontweight='bold')
ax7.axis('off')

ax8 = plt.subplot(3, 3, 8)
ax8.text(0.5, 0.5, f'Precision\n{best_results["precision"]:.4f}',
         ha='center', va='center', fontsize=20, fontweight='bold')
ax8.axis('off')

ax9 = plt.subplot(3, 3, 9)
ax9.text(0.5, 0.5, f'F1-Score\n{best_f1:.4f}',
         ha='center', va='center', fontsize=20, fontweight='bold')
ax9.axis('off')

plt.tight_layout()
plt.savefig('autoencoder_final_analysis.png', dpi=150, bbox_inches='tight')
print("📊 已保存分析圖: autoencoder_final_analysis.png")
plt.show()

print("\n" + "=" * 50)
print("✅ 訓練完成！")
print("=" * 50)
print(f"  - 離群點移除: {outliers_removed} ({outlier_ratio:.2%})")
print(f"  - 最終訓練樣本: {len(X_train)}")
print(f"  - 最佳門檻: {best_threshold:.6f}")
print(f"  - F1-Score: {best_f1:.4f}")
print("=" * 50)