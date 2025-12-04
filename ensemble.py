"""
Deep Autoencoder + Ensemble for Network Intrusion Detection
策略:
1. Deep Autoencoder (6 層) - 更深的特徵學習
2. Random Forest - 基於統計特徵的分類
3. Ensemble - 結合兩者的優勢
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

print("=" * 60)
print("🤖 Deep Autoencoder + Ensemble")
print("=" * 60)

print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")

# === 1️⃣ 載入資料 ===
print("\n📂 載入資料...")
df = pd.read_csv("output_anomaly.csv")
df.columns = df.columns.str.strip()
labels = df['Label'].copy()

print(f"✅ 總樣本: {len(df):,}")
print(f"   BENIGN: {(labels == 'BENIGN').sum():,}")
print(f"   Attack: {(labels != 'BENIGN').sum():,}")

# === 2️⃣ 準備資料 ===
print("\n🎯 準備資料...")

exclude_cols = ['Label', 'anomaly_if']
X_all = df.drop(columns=exclude_cols, errors='ignore')
X_all = X_all.select_dtypes(include=[np.number])

# 標籤: 0=BENIGN, 1=Attack
y_all = (labels != 'BENIGN').astype(int)

# 分割訓練集 (只用 BENIGN) 和測試集
X_benign = X_all[y_all == 0].copy()
X_test_all = X_all.copy()
y_test = y_all.copy()

print(f"✅ BENIGN 訓練: {len(X_benign):,}")
print(f"✅ 全部測試: {len(X_test_all):,}")
print(f"🔢 特徵數: {X_all.shape[1]}")

# === 3️⃣ 預處理 ===
print("\n🧹 預處理...")

# 清理
X_benign = X_benign.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test_all = X_test_all.replace([np.inf, -np.inf], np.nan).fillna(0)

# Winsorization
clip_params = {}
for col in X_benign.columns:
    lower = X_benign[col].quantile(0.005)
    upper = X_benign[col].quantile(0.995)
    X_benign[col] = np.clip(X_benign[col], lower, upper)
    X_test_all[col] = np.clip(X_test_all[col], lower, upper)
    clip_params[col] = {'lower': lower, 'upper': upper}

# 標準化
scaler = StandardScaler()
X_benign_scaled = scaler.fit_transform(X_benign)
X_test_scaled = scaler.transform(X_test_all)

# 後處理裁剪
X_benign_scaled = np.clip(X_benign_scaled, -5, 5)
X_test_scaled = np.clip(X_test_scaled, -5, 5)

print(f"✅ 完成")

# === 4️⃣ 建立 Deep Autoencoder ===
print("\n" + "=" * 60)
print("🧩 建立 Deep Autoencoder (6 層)")
print("=" * 60)

input_dim = X_benign_scaled.shape[1]
encoding_dim = 16  # 更小的 bottleneck

print(f"架構: {input_dim} → 1024 → 512 → 256 → 128 → 64 → {encoding_dim}")

# Encoder
inputs = layers.Input(shape=(input_dim,))

# Layer 1
x = layers.Dense(1024, activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

# Layer 2
x = layers.Dense(512, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.25)(x)

# Layer 3
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)

# Layer 4
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.15)(x)

# Layer 5
x = layers.Dense(64, activation='relu')(x)
x = layers.BatchNormalization()(x)

# Bottleneck (Layer 6) - 加入 L2 正則化
encoded = layers.Dense(encoding_dim, activation='relu',
                       kernel_regularizer=regularizers.l2(0.0001),
                       name='bottleneck')(x)

# Decoder (對稱結構)
# Layer 1
x = layers.Dense(64, activation='relu')(encoded)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.15)(x)

# Layer 2
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)

# Layer 3
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.25)(x)

# Layer 4
x = layers.Dense(512, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

# Layer 5
x = layers.Dense(1024, activation='relu')(x)
x = layers.BatchNormalization()(x)

# Output
decoded = layers.Dense(input_dim, activation='linear')(x)

# 建立模型
deep_ae = models.Model(inputs, decoded, name='deep_autoencoder')

# 編譯
deep_ae.compile(
    optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
    loss='mse',
    metrics=['mae']
)

print(f"\n📊 模型架構:")
deep_ae.summary()

print(f"\n🎯 總參數: {deep_ae.count_params():,}")

# === 5️⃣ 訓練 Deep Autoencoder ===
print("\n" + "=" * 60)
print("🚀 訓練 Deep Autoencoder")
print("=" * 60)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    )
]

history = deep_ae.fit(
    X_benign_scaled, X_benign_scaled,
    epochs=100,
    batch_size=1024,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1
)

epochs = len(history.history['loss'])
print(f"\n✅ 完成: {epochs} epochs")
print(f"   Final Train Loss: {history.history['loss'][-1]:.6f}")
print(f"   Final Val Loss: {history.history['val_loss'][-1]:.6f}")

# === 6️⃣ Deep AE 預測 ===
print("\n🔍 Deep AE 異常分數...")

# 計算 MSE
predictions = deep_ae.predict(X_test_scaled, batch_size=2048, verbose=1)
ae_mse = np.mean(np.square(X_test_scaled - predictions), axis=1)

print(f"✅ Deep AE 異常分數計算完成")

# === 7️⃣ 訓練 Random Forest ===
print("\n" + "=" * 60)
print("🌲 訓練 Random Forest")
print("=" * 60)

# 為 RF 準備訓練資料 (需要有標籤)
# 使用部分資料來訓練 RF (平衡抽樣)
print("準備 RF 訓練資料...")

# 取 BENIGN 和 Attack 各 50,000 筆
benign_indices = np.where(y_all == 0)[0]
attack_indices = np.where(y_all == 1)[0]

n_samples = min(50000, len(attack_indices))
benign_sample = np.random.choice(benign_indices, n_samples, replace=False)
attack_sample = np.random.choice(attack_indices, n_samples, replace=False)

train_indices = np.concatenate([benign_sample, attack_sample])
np.random.shuffle(train_indices)

X_rf_train = X_test_scaled[train_indices]
y_rf_train = y_test[train_indices]

print(f"RF 訓練資料: {len(X_rf_train):,} (BENIGN: {n_samples:,}, Attack: {n_samples:,})")

# 訓練 RF
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("訓練 Random Forest...")
rf.fit(X_rf_train, y_rf_train)
print("✅ RF 訓練完成")

# RF 預測 (機率)
rf_proba = rf.predict_proba(X_test_scaled)[:, 1]  # Attack 的機率

print(f"✅ RF 預測完成")

# === 8️⃣ Ensemble 策略 ===
print("\n" + "=" * 60)
print("🔀 Ensemble 策略")
print("=" * 60)

# 正規化分數到 [0, 1]
ae_score_norm = (ae_mse - ae_mse.min()) / (ae_mse.max() - ae_mse.min() + 1e-10)
rf_score_norm = rf_proba

print("測試多種 Ensemble 策略...")

strategies = {}

# 策略 1: 不同權重組合
for ae_w in [0.3, 0.4, 0.5, 0.6, 0.7]:
    rf_w = 1 - ae_w
    name = f"W_{int(ae_w*10)}:{int(rf_w*10)}"
    strategies[name] = ae_w * ae_score_norm + rf_w * rf_score_norm

# 策略 2: Max
strategies['Max'] = np.maximum(ae_score_norm, rf_score_norm)

# 策略 3: Min
strategies['Min'] = np.minimum(ae_score_norm, rf_score_norm)

# 策略 4: Product
strategies['Product'] = ae_score_norm * rf_score_norm

# 策略 5: Average
strategies['Average'] = (ae_score_norm + rf_score_norm) / 2

# === 9️⃣ 評估 ===
print(f"\n{'Strategy':<12} {'Threshold':>10} {'TPR':>7} {'FPR':>7} {'Prec':>7} {'F1':>7}")
print("-" * 60)

results = []

for name, score in strategies.items():
    # 尋找最佳門檻 (基於 F1)
    thresholds = np.percentile(score[y_test == 0], [90, 92, 94, 95, 96, 97, 98, 99])

    best_f1 = 0
    best_threshold = None
    best_metrics = None

    for threshold in thresholds:
        pred = (score > threshold).astype(int)

        tp = ((y_test == 1) & (pred == 1)).sum()
        fp = ((y_test == 0) & (pred == 1)).sum()
        fn = ((y_test == 1) & (pred == 0)).sum()
        tn = ((y_test == 0) & (pred == 0)).sum()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * prec * tpr / (prec + tpr) if (prec + tpr) > 0 else 0

        if f1 > best_f1 and prec > 0.5:  # 確保 precision > 0.5
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                          'tpr': tpr, 'fpr': fpr, 'precision': prec, 'f1': f1}

    if best_metrics:
        print(f"{name:<12} {best_threshold:>10.4f} {best_metrics['tpr']:>6.1%} "
              f"{best_metrics['fpr']:>6.1%} {best_metrics['precision']:>6.2f} "
              f"{best_metrics['f1']:>6.3f}")

        results.append({
            'name': name,
            'score': score,
            'threshold': best_threshold,
            **best_metrics
        })

# 找最佳策略
best = max(results, key=lambda x: x['f1'])

print(f"\n🏆 最佳策略: {best['name']}")
print(f"   Threshold: {best['threshold']:.4f}")
print(f"   TPR: {best['tpr']:.2%}")
print(f"   FPR: {best['fpr']:.2%}")
print(f"   Precision: {best['precision']:.3f}")
print(f"   F1: {best['f1']:.3f}")

# === 🔟 與原始 AE 比較 ===
print("\n" + "=" * 60)
print("📊 vs 原始 Autoencoder")
print("=" * 60)

try:
    ae_output = pd.read_csv("output_v3.csv")
    ae_anomaly = ae_output['anomaly'].values

    ae_tp = ((y_test == 1) & (ae_anomaly == 1)).sum()
    ae_fp = ((y_test == 0) & (ae_anomaly == 1)).sum()
    ae_fn = ((y_test == 1) & (ae_anomaly == 0)).sum()
    ae_tpr = ae_tp / (ae_tp + ae_fn)
    ae_fpr = ae_fp / (ae_fp + (y_test == 0).sum())
    ae_prec = ae_tp / (ae_tp + ae_fp)
    ae_f1 = 2 * ae_prec * ae_tpr / (ae_prec + ae_tpr)

    print(f"\n{'Model':<25} {'TPR':>7} {'FPR':>7} {'Precision':>10} {'F1':>7}")
    print("-" * 60)
    print(f"{'Original AE (3 layers)':<25} {ae_tpr:>6.1%} {ae_fpr:>6.1%} {ae_prec:>10.3f} {ae_f1:>6.3f}")
    print(f"{'Deep AE (6 layers)':<25} {best['tpr']:>6.1%} {best['fpr']:>6.1%} {best['precision']:>10.3f} {best['f1']:>6.3f}")

    tpr_improve = (best['tpr'] - ae_tpr) / ae_tpr * 100
    f1_improve = (best['f1'] - ae_f1) / ae_f1 * 100

    print(f"\n📈 改善:")
    print(f"   TPR: {tpr_improve:+.1f}%")
    print(f"   F1: {f1_improve:+.1f}%")
except Exception as e:
    print(f"⚠️ 找不到原始 AE 結果: {e}")

# === 各攻擊類型 ===
print("\n🎯 各攻擊類型偵測率:")
for at in sorted(labels[labels != 'BENIGN'].unique()):
    mask = (labels == at)
    detected = (best['score'][mask] > best['threshold']).sum()
    total = mask.sum()
    rate = detected / total if total > 0 else 0
    status = '✅' if rate > 0.5 else '⚠️' if rate > 0.2 else '❌'
    print(f"{status} {at[:30]:<30} {detected:>6}/{total:<6} ({rate:>6.1%})")

# === 儲存 ===
print("\n💾 儲存...")

output = X_all.copy()
output['deep_ae_mse'] = ae_mse
output['rf_proba'] = rf_proba
output['ensemble_score'] = best['score']
output['ensemble_anomaly'] = (best['score'] > best['threshold']).astype(int)
output['Label'] = labels.values

output.to_csv("output_deep_ae_ensemble.csv", index=False)
print(f"✅ output_deep_ae_ensemble.csv")

deep_ae.save("deep_autoencoder.keras")
joblib.dump(rf, "random_forest.pkl")
joblib.dump({
    'scaler': scaler,
    'clip_params': clip_params,
    'best': best,
    'results': results,
    'encoding_dim': encoding_dim
}, "deep_ae_ensemble_config.pkl")
print(f"✅ deep_autoencoder.keras, random_forest.pkl, deep_ae_ensemble_config.pkl")

# === 視覺化 ===
print("\n📊 生成視覺化...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. 訓練曲線
ax = axes[0, 0]
ax.plot(history.history['loss'], label='Train', linewidth=2)
ax.plot(history.history['val_loss'], label='Val', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Deep AE Training History')
ax.legend()
ax.grid(alpha=0.3)

# 2. 分數分佈
ax = axes[0, 1]
bins = 50
ax.hist(best['score'][y_test == 0], bins=bins, alpha=0.7, label='BENIGN', color='green', density=True)
ax.hist(best['score'][y_test == 1], bins=bins, alpha=0.7, label='Attack', color='red', density=True)
ax.axvline(best['threshold'], color='black', linestyle='--', linewidth=2, label='Threshold')
ax.set_xlabel('Ensemble Score')
ax.set_title('Score Distribution')
ax.legend()
ax.grid(alpha=0.3)

# 3. 策略比較
ax = axes[0, 2]
top_strategies = sorted(results, key=lambda x: x['f1'], reverse=True)[:8]
names = [r['name'] for r in top_strategies]
f1s = [r['f1'] for r in top_strategies]
colors = ['gold' if r['name'] == best['name'] else 'steelblue' for r in top_strategies]
ax.barh(names, f1s, color=colors)
ax.set_xlabel('F1-Score')
ax.set_title('Ensemble Strategies')
ax.grid(alpha=0.3, axis='x')

# 4. 混淆矩陣
ax = axes[1, 0]
cm = np.array([[best['tn'], best['fp']], [best['fn'], best['tp']]])
im = ax.imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        text = f"{cm[i,j]:,}\n({cm[i,j]/cm.sum():.1%})"
        color = 'white' if cm[i,j] > cm.max()/2 else 'black'
        ax.text(j, i, text, ha='center', va='center', color=color, fontweight='bold', fontsize=10)
ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_xticklabels(['Normal', 'Attack'])
ax.set_yticklabels(['Normal', 'Attack'])
ax.set_title(f'Confusion Matrix ({best["name"]})')

# 5. AE vs RF 分數比較
ax = axes[1, 1]
sample_size = min(10000, len(ae_score_norm))
sample_idx = np.random.choice(len(ae_score_norm), sample_size, replace=False)
colors_scatter = ['red' if y_test.iloc[i] == 1 else 'green' for i in sample_idx]
ax.scatter(ae_score_norm[sample_idx], rf_score_norm[sample_idx],
          c=colors_scatter, alpha=0.3, s=1)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
ax.set_xlabel('Deep AE Score (normalized)')
ax.set_ylabel('RF Score (probability)')
ax.set_title('AE vs RF Scores')
ax.grid(alpha=0.3)

# 6. 特徵重要性 (RF)
ax = axes[1, 2]
feature_importance = rf.feature_importances_
top_10_idx = np.argsort(feature_importance)[-10:]
ax.barh(range(10), feature_importance[top_10_idx], color='teal')
ax.set_yticks(range(10))
ax.set_yticklabels([f'F{i}' for i in top_10_idx], fontsize=8)
ax.set_xlabel('Importance')
ax.set_title('Top 10 Feature Importance (RF)')
ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('deep_ae_ensemble_analysis.png', dpi=150, bbox_inches='tight')
print(f"✅ deep_ae_ensemble_analysis.png")

print("\n" + "=" * 60)
print("✅ Deep AE + Ensemble 完成!")
print("=" * 60)
print(f"🎯 Deep AE: 6 layers, Bottleneck={encoding_dim}")
print(f"🌲 RF: {rf.n_estimators} trees")
print(f"🏆 Best: {best['name']}")
print(f"📊 TPR: {best['tpr']:.1%}, F1: {best['f1']:.3f}")
print("=" * 60)