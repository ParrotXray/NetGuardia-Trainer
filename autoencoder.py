"""
Autoencoder 訓練 - 雙重裁剪版 v3
策略:
1. Winsorization: 裁剪到 0.5-99.5 百分位
2. StandardScaler 標準化
3. 標準化後再裁剪到 [-5, 5] 範圍
4. Bottleneck = 32 維
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

print("=" * 60)
print("🤖 Autoencoder - 雙重裁剪版 v3")
print("=" * 60)

print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")

# === 1️⃣ 載入 ===
df = pd.read_csv("output_anomaly_combined.csv")
df.columns = df.columns.str.strip()
print(f"\n✅ 資料: {df.shape}")

labels = df['Label'].copy()
print(f"\n📋 標籤:")
print(labels.value_counts())

# === 2️⃣ BENIGN ===
print("\n" + "=" * 60)
print("🎯 準備 BENIGN 資料")
print("=" * 60)

df_benign = df[df['Label'] == 'BENIGN'].copy()
exclude_cols = ['Label', 'anomaly_if']
X_train = df_benign.drop(columns=exclude_cols, errors='ignore')
X_train = X_train.select_dtypes(include=[np.number])

print(f"✅ BENIGN: {len(X_train):,}")
print(f"🔢 特徵: {X_train.shape[1]}")

# === 3️⃣ 清理 ===
print("\n🧹 清理...")
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
print(f"原始範圍: [{X_train.min().min():.2e}, {X_train.max().max():.2e}]")

# === 🆕 4️⃣ 步驟 1: Winsorization ===
print("\n✂️ 步驟 1: Winsorization (0.5%-99.5%)...")

clip_params = {}
for col in X_train.columns:
    lower = X_train[col].quantile(0.005)
    upper = X_train[col].quantile(0.995)
    X_train[col] = np.clip(X_train[col], lower, upper)
    clip_params[col] = {'lower': lower, 'upper': upper}

print(f"✅ 已裁剪")
print(f"裁剪後範圍: [{X_train.min().min():.2e}, {X_train.max().max():.2e}]")

# === 🆕 5️⃣ 步驟 2: StandardScaler ===
print("\n📏 步驟 2: StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print(f"標準化後:")
print(f"  Mean: {X_train_scaled.mean():.4f}")
print(f"  Std: {X_train_scaled.std():.4f}")
print(f"  Range: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")

# === 🆕 6️⃣ 步驟 3: 標準化後裁剪 ===
print("\n✂️ 步驟 3: 標準化後裁剪 (±5σ)...")

# 統計裁剪前的極端值
extreme_count = ((X_train_scaled < -5) | (X_train_scaled > 5)).sum()
print(f"  極端值數量: {extreme_count:,} ({extreme_count/X_train_scaled.size:.2%})")

# 裁剪到 [-5, 5]
X_train_scaled = np.clip(X_train_scaled, -5, 5)

print(f"  最終範圍: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
print(f"  Mean: {X_train_scaled.mean():.4f}")
print(f"  Std: {X_train_scaled.std():.4f}")

# === 7️⃣ Autoencoder ===
print("\n" + "=" * 60)
print("🧩 Autoencoder")
print("=" * 60)

input_dim = X_train_scaled.shape[1]
print(f"輸入: {input_dim} 維")

input_layer = layers.Input(shape=(input_dim,))

# Encoder
x = layers.Dense(128, activation='relu',
                kernel_regularizer=regularizers.l2(0.0001))(input_layer)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)

x = layers.Dense(64, activation='relu',
                kernel_regularizer=regularizers.l2(0.0001))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.15)(x)

# Bottleneck: 32
bottleneck = layers.Dense(32, activation='relu', name='bottleneck')(x)
print(f"Bottleneck: 32 維")

# Decoder
x = layers.Dense(64, activation='relu')(bottleneck)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.15)(x)

x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)

output_layer = layers.Dense(input_dim, activation='linear')(x)

autoencoder = models.Model(inputs=input_layer, outputs=output_layer)
optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
autoencoder.compile(optimizer=optimizer, loss='mse')

autoencoder.summary()

# === 8️⃣ 訓練 ===
print("\n" + "=" * 60)
print("🚀 訓練")
print("=" * 60)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
]

history = autoencoder.fit(
    X_train_scaled, X_train_scaled,
    epochs=100,
    batch_size=1024,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1
)

epochs = len(history.history['loss'])
train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]

print(f"\n✅ 完成: {epochs} epochs")
print(f"  Train: {train_loss:.6f}, Val: {val_loss:.6f}")

# === 9️⃣ 預測 ===
print("\n" + "=" * 60)
print("🔍 預測全部資料")
print("=" * 60)

df_all = df.drop(columns=exclude_cols, errors='ignore')
X_all = df_all.select_dtypes(include=[np.number])
X_all = X_all[X_train.columns]
X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0)

# 套用相同裁剪
for col in X_all.columns:
    if col in clip_params:
        X_all[col] = np.clip(X_all[col], clip_params[col]['lower'], clip_params[col]['upper'])

# 標準化並裁剪
X_all_scaled = scaler.transform(X_all)
X_all_scaled = np.clip(X_all_scaled, -5, 5)

recon = autoencoder.predict(X_all_scaled, batch_size=2048, verbose=1)
mse = np.mean(np.square(X_all_scaled - recon), axis=1)

# === 🔟 分析 ===
print("\n" + "=" * 60)
print("📊 分析")
print("=" * 60)

mse_b = mse[labels == 'BENIGN']
mse_a = mse[labels != 'BENIGN']

print(f"\n🟢 BENIGN (n={len(mse_b):,}):")
print(f"  Mean: {mse_b.mean():.6f}")
print(f"  Median: {np.median(mse_b):.6f}")
print(f"  P95: {np.percentile(mse_b, 95):.6f}")
print(f"  P99: {np.percentile(mse_b, 99):.6f}")
print(f"  Range: [{mse_b.min():.6f}, {mse_b.max():.6f}]")

print(f"\n🔴 Attack (n={len(mse_a):,}):")
print(f"  Mean: {mse_a.mean():.6f}")
print(f"  Median: {np.median(mse_a):.6f}")
print(f"  Range: [{mse_a.min():.6f}, {mse_a.max():.6f}]")

ratio_mean = mse_a.mean() / mse_b.mean()
ratio_med = np.median(mse_a) / np.median(mse_b)

print(f"\n📈 分離度:")
print(f"  Mean: {ratio_mean:.2f}x", "✅" if ratio_mean > 2 else "⚠️")
print(f"  Median: {ratio_med:.2f}x", "✅" if ratio_med > 3 else "⚠️")

# 各類型
print(f"\n🎯 各攻擊類型:")
print(f"{'Type':<30} {'n':>8} {'Mean':>10} {'Median':>10} {'Ratio':>8}")
print("-" * 70)

stats = []
for at in labels[labels != 'BENIGN'].unique():
    m = mse[labels == at]
    stats.append({
        'type': at,
        'n': len(m),
        'mean': m.mean(),
        'med': np.median(m)
    })

stats.sort(key=lambda x: x['med'], reverse=True)

for s in stats:
    r = s['med'] / np.median(mse_b)
    st = '✅' if r > 3 else '⚠️' if r > 1.5 else '❌'
    print(f"{st} {s['type']:<27} {s['n']:>8,} {s['mean']:>10.4f} {s['med']:>10.4f} {r:>7.1f}x")

# === 1️⃣1️⃣ 門檻 ===
print("\n" + "=" * 60)
print("🎯 門檻")
print("=" * 60)

thresholds = {}
for p in [90, 95, 99]:
    thresholds[f"B{p}"] = np.percentile(mse_b, p)
for p in [85, 90, 95]:
    thresholds[f"A{p}"] = np.percentile(mse, p)

med_b = np.median(mse_b)
mad = np.median(np.abs(mse_b - med_b))
for n in [3, 4, 5, 6]:
    thresholds[f"MAD{n}"] = med_b + n * mad

print(f"\n{'Strategy':<10} {'Thresh':>10} {'TPR':>7} {'FPR':>7} {'Prec':>7} {'F1':>7}")
print("-" * 55)

results = []
for name, th in sorted(thresholds.items(), key=lambda x: x[1]):
    pred = (mse > th).astype(int)
    tp = ((labels != 'BENIGN') & (pred == 1)).sum()
    fp = ((labels == 'BENIGN') & (pred == 1)).sum()
    fn = ((labels != 'BENIGN') & (pred == 0)).sum()
    tn = ((labels == 'BENIGN') & (pred == 0)).sum()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * prec * tpr / (prec + tpr) if (prec + tpr) > 0 else 0

    print(f"{name:<10} {th:>10.6f} {tpr:>6.1%} {fpr:>6.1%} {prec:>6.2f} {f1:>6.3f}")

    results.append({
        'name': name, 'th': th, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'tpr': tpr, 'fpr': fpr, 'prec': prec, 'f1': f1
    })

best = max([r for r in results if r['prec'] > 0.5] or results, key=lambda x: x['f1'])

print(f"\n🏆 {best['name']}: threshold={best['th']:.4f}")
print(f"  TPR: {best['tpr']:.2%}, FPR: {best['fpr']:.2%}")
print(f"  Precision: {best['prec']:.3f}, F1: {best['f1']:.3f}")

# === 1️⃣2️⃣ 儲存 ===
print("\n💾 儲存...")

output = X_all.copy()
output['score'] = mse
output['anomaly'] = (mse > best['th']).astype(int)
output['Label'] = labels.values
output.to_csv("output_v3.csv", index=False)

autoencoder.save("ae_v3.keras")
joblib.dump(scaler, "scaler_v3.pkl")
joblib.dump({
    'best': best,
    'results': results,
    'clip_params': clip_params,
    'features': X_train.columns.tolist(),
    'post_clip': [-5, 5]  # 標準化後裁剪範圍
}, "info_v3.pkl")

print(f"✅ output_v3.csv, ae_v3.keras, scaler_v3.pkl, info_v3.pkl")

# === 視覺化 ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 訓練
ax = axes[0, 0]
ax.plot(history.history['loss'], label='Train', linewidth=2)
ax.plot(history.history['val_loss'], label='Val', linewidth=2)
ax.set_title('Training')
ax.legend()
ax.grid(alpha=0.3)

# MSE
ax = axes[0, 1]
bins = np.linspace(0, np.percentile(mse, 99), 100)
ax.hist(mse_b, bins=bins, alpha=0.7, label='BENIGN', color='green', density=True)
ax.hist(mse_a, bins=bins, alpha=0.7, label='Attack', color='red', density=True)
ax.axvline(best['th'], color='black', linestyle='--', linewidth=2)
ax.set_title('MSE Distribution')
ax.legend()
ax.grid(alpha=0.3)

# 混淆矩陣
ax = axes[1, 0]
cm = np.array([[best['tn'], best['fp']], [best['fn'], best['tp']]])
im = ax.imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i,j]:,}\n({cm[i,j]/cm.sum():.1%})",
               ha='center', va='center',
               color='white' if cm[i,j] > cm.max()/2 else 'black',
               fontweight='bold')
ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_xticklabels(['Normal', 'Attack'])
ax.set_yticklabels(['Normal', 'Attack'])
ax.set_title('Confusion Matrix')

# F1
ax = axes[1, 1]
top10 = sorted(results, key=lambda x: x['f1'], reverse=True)[:10]
names = [r['name'] for r in top10]
f1s = [r['f1'] for r in top10]
colors = ['gold' if r['name']==best['name'] else 'steelblue' for r in top10]
ax.barh(names, f1s, color=colors)
ax.set_xlabel('F1-Score')
ax.set_title('Top Strategies')
ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('ae_v3.png', dpi=150)
print(f"✅ ae_v3.png")

print("\n" + "=" * 60)
print("✅ 完成!")
print("=" * 60)
print(f"📊 訓練: {len(X_train):,} BENIGN, {epochs} epochs")
print(f"✂️ 雙重裁剪: Winsorization + Post-scaling clip")
print(f"🏆 {best['name']}: TPR={best['tpr']:.1%}, FPR={best['fpr']:.2%}")
print(f"🎯 Precision={best['prec']:.3f}, F1={best['f1']:.3f}")
print(f"📊 分離度: Mean={ratio_mean:.1f}x, Median={ratio_med:.1f}x")
print("=" * 60)