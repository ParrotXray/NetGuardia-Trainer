import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

print("=" * 60)
print("🚀 改進版 MLP (SMOTE + 類別權重)")
print("=" * 60)

# === 1️⃣ 載入資料 ===
print("\n📂 載入資料...")
df = pd.read_csv("output_deep_ae_ensemble.csv")
df.columns = df.columns.str.strip()

config = joblib.load("deep_ae_ensemble_config.pkl")
scaler = config['scaler']
clip_params = config['clip_params']

# 只用異常樣本
df_anomaly = df[df['ensemble_anomaly'] == 1].copy()

print(f"異常樣本: {len(df_anomaly):,}")

# === 2️⃣ 準備資料 ===
print("\n🔢 準備特徵...")

exclude_cols = ['Label', 'deep_ae_mse', 'rf_proba', 'ensemble_score',
                'ensemble_anomaly', 'anomaly_if']
X = df_anomaly.drop(columns=exclude_cols, errors='ignore')
y = df_anomaly['Label']

# 清理
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
for col in X.columns:
    if col in clip_params:
        X[col] = np.clip(X[col], clip_params[col]['lower'], clip_params[col]['upper'])

# 標準化
X_scaled = scaler.transform(X)
X_scaled = np.clip(X_scaled, -5, 5)

# 標籤編碼
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print(f"特徵維度: {X_scaled.shape}")
print(f"類別數: {len(encoder.classes_)}")

# 顯示原始類別分布
print(f"\n📊 原始類別分布:")
for idx, label in enumerate(encoder.classes_):
    count = (y_encoded == idx).sum()
    print(f"  {idx:2d}. {label:<35} {count:>7,}")

# === 3️⃣ 分割資料 ===
print("\n📊 分割資料...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"訓練集: {X_train.shape[0]:,}")
print(f"測試集: {X_test.shape[0]:,}")

# === 4️⃣ SMOTE 資料增強 ===
print("\n" + "=" * 60)
print("🔄 SMOTE 資料增強")
print("=" * 60)

# 計算每個類別的樣本數
unique, counts = np.unique(y_train, return_counts=True)
class_counts = dict(zip(unique, counts))

# 設定 SMOTE 策略：將少數類別提升到多數類別的 50%
max_count = max(counts)
sampling_strategy = {}
for cls, count in class_counts.items():
    if count < max_count * 0.5:  # 少於多數類別 50%
        sampling_strategy[cls] = int(max_count * 0.5)

print(f"\nSMOTE 策略:")
for cls in sampling_strategy:
    label = encoder.classes_[cls]
    original = class_counts[cls]
    target = sampling_strategy[cls]
    print(f"  {label:<35} {original:>6,} → {target:>6,} (+{target-original:,})")

# 應用 SMOTE
print("\n執行 SMOTE...")
smote = SMOTE(
    sampling_strategy=sampling_strategy,
    k_neighbors=min(5, min([class_counts[c] for c in sampling_strategy]) - 1),
    random_state=42
)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"\n✅ SMOTE 完成")
print(f"訓練集大小: {X_train.shape[0]:,} → {X_train_balanced.shape[0]:,}")

# 顯示平衡後的分布
print(f"\n📊 平衡後類別分布:")
unique, counts = np.unique(y_train_balanced, return_counts=True)
for cls, count in zip(unique, counts):
    label = encoder.classes_[cls]
    print(f"  {label:<35} {count:>7,}")

# === 5️⃣ 計算類別權重 ===
print("\n" + "=" * 60)
print("⚖️ 計算類別權重")
print("=" * 60)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_balanced),
    y=y_train_balanced
)
class_weight_dict = dict(enumerate(class_weights))

print(f"\n類別權重 (前10個):")
for idx in range(min(10, len(class_weights))):
    label = encoder.classes_[idx]
    weight = class_weights[idx]
    print(f"  {label:<35} {weight:.4f}")

# === 6️⃣ 建立改進版 MLP ===
print("\n" + "=" * 60)
print("🧩 建立改進版 MLP")
print("=" * 60)

n_classes = len(encoder.classes_)
input_dim = X_train_balanced.shape[1]

mlp_improved = Sequential([
    Input(shape=(input_dim,)),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(n_classes, activation='softmax')
], name='mlp_improved')

mlp_improved.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📝 模型架構:")
mlp_improved.summary()

# === 7️⃣ 訓練 ===
print("\n" + "=" * 60)
print("🚀 訓練改進版 MLP")
print("=" * 60)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
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

history = mlp_improved.fit(
    X_train_balanced, y_train_balanced,
    epochs=100,
    batch_size=512,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    class_weight=class_weight_dict,  # 🔥 使用類別權重
    verbose=1
)

print(f"\n✅ 訓練完成")

# === 8️⃣ 評估 ===
print("\n" + "=" * 60)
print("📊 評估結果")
print("=" * 60)

loss, acc = mlp_improved.evaluate(X_test, y_test, verbose=0)
print(f"\n測試集準確率: {acc:.4f}")
print(f"測試集損失: {loss:.4f}")

y_pred = np.argmax(mlp_improved.predict(X_test, verbose=0), axis=1)

print("\n📋 詳細分類報告:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_, digits=4))

# === 9️⃣ 重點檢查 XSS ===
print("\n" + "=" * 60)
print("🎯 XSS 分類結果")
print("=" * 60)

xss_idx = list(encoder.classes_).index('Web Attack � XSS')
xss_mask_test = (y_test == xss_idx)

if xss_mask_test.sum() > 0:
    xss_correct = (y_pred[xss_mask_test] == xss_idx).sum()
    xss_total = xss_mask_test.sum()
    xss_accuracy = xss_correct / xss_total

    print(f"\nXSS 性能:")
    print(f"  測試樣本: {xss_total}")
    print(f"  正確預測: {xss_correct}")
    print(f"  準確率: {xss_accuracy:.1%}")

    # XSS 被誤判成什麼
    xss_predictions = y_pred[xss_mask_test]
    print(f"\n  XSS 被預測為:")
    unique, counts = np.unique(xss_predictions, return_counts=True)
    for cls, count in zip(unique, counts):
        label = encoder.classes_[cls]
        pct = count / xss_total * 100
        print(f"    {label:<35} {count:>3} ({pct:>5.1f}%)")

# === 🔟 比較改進 ===
print("\n" + "=" * 60)
print("📈 改進對比")
print("=" * 60)

print(f"\n{'類別':<35} {'原始':<10} {'改進後':<10} {'變化'}")
print("-" * 70)

# 嘗試載入原始結果
try:
    df_old = pd.read_csv("output_mlp.csv")
    y_old_true = df_old['Label']
    y_old_pred = df_old['predicted_label']

    for label in encoder.classes_:
        # 原始準確率
        mask_old = (y_old_true == label)
        if mask_old.sum() > 0:
            old_acc = (y_old_pred[mask_old] == label).sum() / mask_old.sum()
        else:
            old_acc = 0

        # 新準確率
        label_idx = list(encoder.classes_).index(label)
        mask_new = (y_test == label_idx)
        if mask_new.sum() > 0:
            new_acc = (y_pred[mask_new] == label_idx).sum() / mask_new.sum()
        else:
            new_acc = 0

        change = new_acc - old_acc
        change_str = f"{change:+.1%}" if change != 0 else "  -"

        print(f"{label:<35} {old_acc:>8.1%}  {new_acc:>8.1%}  {change_str}")

except:
    print("⚠️ 找不到原始結果，無法比較")

# === 11️⃣ 儲存 ===
print("\n💾 儲存改進版模型...")

mlp_improved.save("mlp_improved.keras")
joblib.dump(encoder, "label_encoder_improved.pkl")

config_improved = {
    'encoder': encoder,
    'scaler': scaler,
    'clip_params': clip_params,
    'class_weights': class_weight_dict,
    'smote_strategy': sampling_strategy,
    'test_accuracy': acc,
    'test_loss': loss
}
joblib.dump(config_improved, "mlp_improved_config.pkl")

print("✅ 已保存:")
print("  - mlp_improved.keras")
print("  - label_encoder_improved.pkl")
print("  - mlp_improved_config.pkl")

# === 12️⃣ 視覺化 ===
print("\n📊 生成視覺化...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 混淆矩陣
ax = axes[0, 0]
cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=False, cmap='Blues', ax=ax,
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
ax.set_title('Confusion Matrix (Normalized)')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')

# 2. 訓練曲線
ax = axes[0, 1]
ax.plot(history.history['accuracy'], label='Train', linewidth=2)
ax.plot(history.history['val_accuracy'], label='Val', linewidth=2)
ax.set_title('Training History')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.legend()
ax.grid(alpha=0.3)

# 3. 各類別準確率
ax = axes[1, 0]
accuracies = []
labels_list = []
for idx, label in enumerate(encoder.classes_):
    mask = (y_test == idx)
    if mask.sum() > 0:
        acc = (y_pred[mask] == idx).sum() / mask.sum()
        accuracies.append(acc)
        labels_list.append(label[:20])  # 截斷長標籤

y_pos = np.arange(len(labels_list))
colors = ['red' if acc < 0.5 else 'orange' if acc < 0.8 else 'green'
          for acc in accuracies]
ax.barh(y_pos, accuracies, color=colors)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels_list, fontsize=8)
ax.set_xlabel('Accuracy')
ax.set_title('Per-Class Accuracy')
ax.grid(alpha=0.3, axis='x')

# 4. 類別樣本分布
ax = axes[1, 1]
train_dist = np.bincount(y_train_balanced)
test_dist = np.bincount(y_test)
x = np.arange(len(encoder.classes_))
width = 0.35
ax.bar(x - width/2, train_dist, width, label='Train (SMOTE)', alpha=0.8)
ax.bar(x + width/2, test_dist, width, label='Test', alpha=0.8)
ax.set_xlabel('Class')
ax.set_ylabel('Count')
ax.set_title('Class Distribution')
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('mlp_improved_analysis.png', dpi=150, bbox_inches='tight')
print("✅ 已保存: mlp_improved_analysis.png")

print("\n" + "=" * 60)
print("✅ 改進版 MLP 完成！")
print("=" * 60)
print(f"📊 測試準確率: {acc:.4f}")
print(f"🎯 重點: 檢查上方 XSS 的準確率是否改善")
print("=" * 60)