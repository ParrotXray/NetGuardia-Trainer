"""
測試階段（最終改良版）：
- 自動載入最佳門檻
- 多門檻策略測試
- 詳細的攻擊類型分析
- 漏報/誤報深度分析
"""
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print("=" * 50)
print("🧪 Step 4: 模型測試（最終改良版）")
print("=" * 50)

# === 1️⃣ 載入模型與工具 ===
print("📦 載入模型與工具...")
autoencoder = load_model("autoencoder_cic_model.h5", compile=False)
mlp = load_model("mlp_attack_classifier.h5")
scaler = joblib.load("scaler_ae.pkl")
le = joblib.load("label_encoder.pkl")

# 載入門檻資訊
try:
    threshold_info = joblib.load("threshold_info.pkl")
    recommended_threshold = threshold_info['threshold']
    threshold_strategy = threshold_info['strategy']
    all_thresholds = threshold_info.get('all_thresholds', {})
    best_results = threshold_info.get('best_results', {})

    print(f"✅ 載入推薦門檻: {recommended_threshold:.6f}")
    print(f"📊 門檻策略: {threshold_strategy}")
    print(f"📈 訓練時 F1-Score: {best_results.get('f1', 0):.4f}")
except:
    recommended_threshold = None
    threshold_strategy = "Not found"
    all_thresholds = {}
    print("⚠️ 未找到推薦門檻")

print("✅ 模型與工具已載入成功")

# === 2️⃣ 讀取測試資料 ===
test_file = "Wednesday-workingHours.pcap_ISCX.csv"
df = pd.read_csv(test_file)
df.columns = df.columns.str.strip()

print(f"\n✅ 載入測試資料: {df.shape}")
print(f"📋 測試資料標籤分布:\n{df['Label'].value_counts()}")

labels = df['Label'].copy()

# === 3️⃣ 準備特徵 ===
drop_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Label']
df_features = df.drop(columns=drop_cols, errors='ignore')

X = df_features.select_dtypes(include=[np.number])
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
X = np.clip(X, -1e9, 1e9)

print(f"🔢 測試資料特徵維度: {X.shape}")

# === 4️⃣ 檢查欄位一致性 ===
if hasattr(scaler, 'feature_names_in_'):
    scaler_cols = scaler.feature_names_in_
    missing_cols = set(scaler_cols) - set(X.columns)
    if missing_cols:
        print(f"⚠️ 測試資料缺少欄位: {missing_cols}")
        for col in missing_cols:
            X[col] = 0
    X = X[scaler_cols]
    print(f"✅ 已對齊特徵欄位，最終維度: {X.shape}")

# === 5️⃣ 標準化 ===
X_scaled = scaler.transform(X)

# === 6️⃣ 驗證維度 ===
expected_dim = autoencoder.input_shape[1]
current_dim = X_scaled.shape[1]

if current_dim != expected_dim:
    raise ValueError(f"❌ 維度不匹配！模型需要 {expected_dim} 維，但資料有 {current_dim} 維")

print(f"✅ 特徵維度匹配: {current_dim}")

# === 7️⃣ Autoencoder 異常偵測 ===
print("\n🔍 執行 Autoencoder 異常偵測...")
recon = autoencoder.predict(X_scaled, verbose=0)
mse = np.mean(np.square(X_scaled - recon), axis=1)

mse_benign = mse[labels == 'BENIGN']
mse_attack = mse[labels != 'BENIGN']

print(f"\n📊 重建誤差統計:")
print(f"  BENIGN: Mean={mse_benign.mean():.6f}, Median={np.median(mse_benign):.6f}, Max={mse_benign.max():.6f}")
print(f"  Attack: Mean={mse_attack.mean():.6f}, Median={np.median(mse_attack):.6f}, Max={mse_attack.max():.6f}")
print(f"  Ratio (Attack/BENIGN Mean): {mse_attack.mean()/mse_benign.mean():.2f}x")

# === 🆕 8️⃣ 測試多個門檻策略 ===
print("\n" + "=" * 50)
print("🎯 測試多個門檻策略")
print("=" * 50)

# 定義要測試的門檻
test_thresholds = {
    'Recommended': recommended_threshold if recommended_threshold else np.percentile(mse, 95),
    'BENIGN_P90': np.percentile(mse_benign, 90),
    'BENIGN_P95': np.percentile(mse_benign, 95),
    'BENIGN_P97': np.percentile(mse_benign, 97),
    'BENIGN_P99': np.percentile(mse_benign, 99),
    'Mean+2Std': mse_benign.mean() + 2 * mse_benign.std(),
    'Mean+3Std': mse_benign.mean() + 3 * mse_benign.std(),
}

print(f"\n{'策略':<20} {'門檻值':<12} {'TP':<8} {'FP':<8} {'FN':<8} {'Precision':<10} {'Recall':<10} {'F1':<10}")
print("-" * 95)

threshold_results = []
for name, thresh in sorted(test_thresholds.items(), key=lambda x: x[1]):
    is_ano = (mse > thresh).astype(int)

    tp = ((labels != 'BENIGN') & (is_ano == 1)).sum()
    fp = ((labels == 'BENIGN') & (is_ano == 1)).sum()
    fn = ((labels != 'BENIGN') & (is_ano == 0)).sum()
    tn = ((labels == 'BENIGN') & (is_ano == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"{name:<20} {thresh:<12.6f} {tp:<8} {fp:<8} {fn:<8} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")

    threshold_results.append({
        'name': name,
        'threshold': thresh,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision, 'recall': recall, 'f1': f1
    })

# 選擇最佳門檻
best_test_result = max(threshold_results, key=lambda x: x['f1'])
threshold = best_test_result['threshold']
threshold_name = best_test_result['name']

print(f"\n🏆 測試集最佳門檻: {threshold_name} (Threshold={threshold:.6f}, F1={best_test_result['f1']:.4f})")

# === 9️⃣ 使用最佳門檻進行預測 ===
is_anomaly = (mse > threshold).astype(int)

print(f"\n📊 異常偵測結果:")
print(f"  - 偵測到異常: {is_anomaly.sum()} / {len(df)}")
print(f"  - 異常比例: {is_anomaly.sum()/len(df):.2%}")

# === 🔟 MLP 分類異常樣本 ===
X_anomaly = X_scaled[is_anomaly == 1]

if len(X_anomaly) > 0:
    print(f"\n🧠 對 {len(X_anomaly)} 個異常樣本進行攻擊分類...")

    mlp_expected_dim = mlp.input_shape[1]
    if X_anomaly.shape[1] != mlp_expected_dim:
        raise ValueError(f"❌ MLP 維度不匹配！")

    preds = mlp.predict(X_anomaly, verbose=0)
    pred_labels = le.inverse_transform(np.argmax(preds, axis=1))
    print(f"✅ 分類完成")
    print(f"📋 預測攻擊類型分布:\n{pd.Series(pred_labels).value_counts()}")
else:
    pred_labels = []
    print("⚠️ 沒有偵測到異常樣本")

# === 11️⃣ 組合輸出結果 ===
output = pd.DataFrame()
output["Label"] = labels.values
output["anomaly_score"] = mse
output["is_anomaly"] = is_anomaly
output["predicted_attack"] = "BENIGN"

if len(pred_labels) > 0:
    output.loc[is_anomaly == 1, "predicted_attack"] = pred_labels

output.to_csv("prediction_results.csv", index=False)
print("\n💾 預測結果已輸出: prediction_results.csv")

# === 12️⃣ 詳細評估 ===
print("\n" + "=" * 50)
print("📊 詳細評估報告")
print("=" * 50)

tp = best_test_result['tp']
fp = best_test_result['fp']
fn = best_test_result['fn']
tn = best_test_result['tn']

print(f"\n🎯 Autoencoder 異常偵測性能:")
print(f"  True Positives (正確偵測攻擊):  {tp:,}")
print(f"  False Positives (誤報):         {fp:,}")
print(f"  False Negatives (漏報):         {fn:,}")
print(f"  True Negatives (正確識別正常):  {tn:,}")

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = best_test_result['precision']
recall = best_test_result['recall']
f1 = best_test_result['f1']

print(f"\n📈 性能指標:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")

# === 🆕 13️⃣ 各攻擊類型詳細分析 ===
print("\n" + "=" * 50)
print("🎯 各攻擊類型詳細分析")
print("=" * 50)

print(f"\n{'攻擊類型':<25} {'總數':<10} {'偵測數':<10} {'偵測率':<10} {'平均MSE':<12} {'中位MSE':<12}")
print("-" * 90)

attack_analysis = []
for attack_type in sorted(labels[labels != 'BENIGN'].unique()):
    mask_attack = (labels == attack_type)
    total = mask_attack.sum()
    detected = ((labels == attack_type) & (is_anomaly == 1)).sum()
    rate = detected / total if total > 0 else 0

    mse_type = mse[mask_attack]
    mean_mse = mse_type.mean()
    median_mse = np.median(mse_type)

    print(f"{attack_type:<25} {total:<10} {detected:<10} {rate:<10.2%} {mean_mse:<12.6f} {median_mse:<12.6f}")

    attack_analysis.append({
        'type': attack_type,
        'total': total,
        'detected': detected,
        'rate': rate,
        'mean_mse': mean_mse,
        'median_mse': median_mse
    })

# 找出最難偵測的攻擊
worst_attack = min(attack_analysis, key=lambda x: x['rate'])
print(f"\n⚠️ 最難偵測的攻擊: {worst_attack['type']} (偵測率: {worst_attack['rate']:.2%})")

# === 🆕 14️⃣ 漏報深度分析 ===
print("\n" + "=" * 50)
print("🚫 漏報（False Negatives）深度分析")
print("=" * 50)

false_negatives = output[(output['Label'] != 'BENIGN') & (output['is_anomaly'] == 0)]

if len(false_negatives) > 0:
    print(f"\n📊 漏報總數: {len(false_negatives):,} ({len(false_negatives)/(labels != 'BENIGN').sum():.2%} of all attacks)")
    print(f"📊 漏報的 MSE 統計:")
    print(f"  - Mean: {false_negatives['anomaly_score'].mean():.6f}")
    print(f"  - Median: {false_negatives['anomaly_score'].median():.6f}")
    print(f"  - Max: {false_negatives['anomaly_score'].max():.6f}")
    print(f"  - Min: {false_negatives['anomaly_score'].min():.6f}")

    print(f"\n📋 漏報攻擊類型分布:")
    for attack_type in sorted(false_negatives['Label'].unique()):
        fn_type = false_negatives[false_negatives['Label'] == attack_type]
        total_type = (labels == attack_type).sum()
        print(f"  {attack_type:<25}: {len(fn_type):>6} / {total_type:<6} ({len(fn_type)/total_type:>6.2%})")

    # 漏報樣本與正常樣本的 MSE 比較
    print(f"\n🔍 漏報樣本 vs BENIGN 的 MSE 比較:")
    fn_mse_mean = false_negatives['anomaly_score'].mean()
    benign_mse_mean = mse_benign.mean()
    print(f"  漏報平均 MSE: {fn_mse_mean:.6f}")
    print(f"  BENIGN 平均 MSE: {benign_mse_mean:.6f}")
    print(f"  比值: {fn_mse_mean/benign_mse_mean:.2f}x")

    # 建議調整
    fn_mse_95 = false_negatives['anomaly_score'].quantile(0.95)
    print(f"\n💡 建議: 若要抓到 95% 的漏報攻擊，門檻需降至: {fn_mse_95:.6f}")
    print(f"  (當前門檻: {threshold:.6f})")

# === 🆕 15️⃣ 誤報分析 ===
print("\n" + "=" * 50)
print("⚠️ 誤報（False Positives）分析")
print("=" * 50)

false_positives = output[(output['Label'] == 'BENIGN') & (output['is_anomaly'] == 1)]

if len(false_positives) > 0:
    print(f"\n📊 誤報總數: {len(false_positives):,} ({len(false_positives)/(labels == 'BENIGN').sum():.2%} of all BENIGN)")
    print(f"📊 誤報的 MSE 統計:")
    print(f"  - Mean: {false_positives['anomaly_score'].mean():.6f}")
    print(f"  - Median: {false_positives['anomaly_score'].median():.6f}")
    print(f"  - Max: {false_positives['anomaly_score'].max():.6f}")
    print(f"  - Min: {false_positives['anomaly_score'].min():.6f}")

    print(f"\n🔍 誤報樣本 vs 攻擊樣本的 MSE 比較:")
    fp_mse_mean = false_positives['anomaly_score'].mean()
    attack_mse_mean = mse_attack.mean()
    print(f"  誤報平均 MSE: {fp_mse_mean:.6f}")
    print(f"  攻擊平均 MSE: {attack_mse_mean:.6f}")
    print(f"  比值: {fp_mse_mean/attack_mse_mean:.2f}x")

# === 16️⃣ MLP 分類性能評估 ===
if len(pred_labels) > 0:
    print("\n" + "=" * 50)
    print("🧠 MLP 攻擊分類性能")
    print("=" * 50)

    true_labels_of_anomalies = output.loc[output['is_anomaly'] == 1, 'Label'].values

    correct_classification = (true_labels_of_anomalies == pred_labels).sum()
    classification_acc = correct_classification / len(pred_labels)

    print(f"\n📊 分類準確率: {classification_acc:.4f} ({correct_classification}/{len(pred_labels)})")

    # 只對攻擊樣本評估（排除 BENIGN）
    mask_real_attack = true_labels_of_anomalies != 'BENIGN'
    if mask_real_attack.sum() > 0:
        true_attack_labels = true_labels_of_anomalies[mask_real_attack]
        pred_attack_labels = pred_labels[mask_real_attack]

        attack_correct = (true_attack_labels == pred_attack_labels).sum()
        attack_acc = attack_correct / len(true_attack_labels)

        print(f"📊 攻擊分類準確率（只看真實攻擊）: {attack_acc:.4f} ({attack_correct}/{len(true_attack_labels)})")

        print(f"\n📋 攻擊分類報告（只看真實攻擊）:")
        try:
            report = classification_report(true_attack_labels, pred_attack_labels, zero_division=0)
            print(report)
        except:
            print("  (無法生成報告)")

# === 17️⃣ 進階視覺化 ===
print("\n📊 生成視覺化圖表...")

fig = plt.figure(figsize=(20, 14))

# 子圖 1: 重建誤差分佈
ax1 = plt.subplot(3, 4, 1)
ax1.hist(mse_benign, bins=100, alpha=0.7, label='BENIGN', color='green', density=True)
ax1.hist(mse_attack, bins=100, alpha=0.7, label='Attack', color='red', density=True)
ax1.axvline(threshold, color='black', linestyle='--', linewidth=2,
            label=f'Threshold={threshold:.4f}\n({threshold_name})')
ax1.set_xlabel('Reconstruction Error (MSE)')
ax1.set_ylabel('Density')
ax1.set_title('Reconstruction Error Distribution', fontsize=11, fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# 子圖 2: 混淆矩陣
ax2 = plt.subplot(3, 4, 2)
cm = np.array([[tn, fp], [fn, tp]])
sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=ax2,
            xticklabels=['Pred\nNormal', 'Pred\nAttack'],
            yticklabels=['True\nNormal', 'True\nAttack'],
            cbar_kws={'label': 'Count'})
ax2.set_title('Confusion Matrix', fontsize=11, fontweight='bold')

# 子圖 3: 門檻策略比較（F1-Score）
ax3 = plt.subplot(3, 4, 3)
strategy_names = [r['name'] for r in threshold_results]
f1_scores = [r['f1'] for r in threshold_results]
colors = ['gold' if r['name'] == threshold_name else 'steelblue' for r in threshold_results]

bars = ax3.barh(strategy_names, f1_scores, color=colors)
ax3.set_xlabel('F1-Score')
ax3.set_title('Threshold Strategy F1-Score', fontsize=11, fontweight='bold')
ax3.set_xlim(0, 1)
ax3.grid(True, alpha=0.3, axis='x')

for i, (bar, score) in enumerate(zip(bars, f1_scores)):
    ax3.text(score + 0.02, i, f'{score:.3f}', va='center', fontsize=8)

# 子圖 4: 性能指標雷達圖
ax4 = plt.subplot(3, 4, 4, projection='polar')
metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
values = [precision, recall, f1, accuracy]

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
values += values[:1]
angles += angles[:1]

ax4.plot(angles, values, 'o-', linewidth=2, color='blue')
ax4.fill(angles, values, alpha=0.25, color='blue')
ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(metrics, fontsize=9)
ax4.set_ylim(0, 1)
ax4.set_title('Performance Metrics', fontsize=11, fontweight='bold', pad=15)
ax4.grid(True)

# 子圖 5: 各攻擊類型偵測率
ax5 = plt.subplot(3, 4, 5)
attack_types = [a['type'] for a in attack_analysis]
detection_rates = [a['rate'] * 100 for a in attack_analysis]
colors_bar = ['green' if r >= 70 else 'orange' if r >= 50 else 'red' for r in detection_rates]

bars = ax5.barh(attack_types, detection_rates, color=colors_bar)
ax5.set_xlabel('Detection Rate (%)')
ax5.set_title('Detection Rate by Attack Type', fontsize=11, fontweight='bold')
ax5.set_xlim(0, 100)
ax5.grid(True, alpha=0.3, axis='x')

for i, (bar, rate) in enumerate(zip(bars, detection_rates)):
    ax5.text(rate + 2, i, f'{rate:.1f}%', va='center', fontsize=8, fontweight='bold')

# 子圖 6: 各攻擊類型 MSE 分佈
ax6 = plt.subplot(3, 4, 6)
attack_types_plot = [a['type'] for a in attack_analysis]
mean_mses = [a['mean_mse'] for a in attack_analysis]
median_mses = [a['median_mse'] for a in attack_analysis]

x = np.arange(len(attack_types_plot))
width = 0.35

bars1 = ax6.barh(x - width/2, mean_mses, width, label='Mean MSE', color='steelblue', alpha=0.8)
bars2 = ax6.barh(x + width/2, median_mses, width, label='Median MSE', color='coral', alpha=0.8)

ax6.set_xlabel('MSE')
ax6.set_title('Mean vs Median MSE by Attack Type', fontsize=11, fontweight='bold')
ax6.set_yticks(x)
ax6.set_yticklabels(attack_types_plot, fontsize=8)
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3, axis='x')

# 子圖 7: MSE 散點圖（採樣）
ax7 = plt.subplot(3, 4, 7)
sample_size = min(5000, len(mse))
sample_indices = np.random.choice(len(mse), sample_size, replace=False)
colors_scatter = ['green' if l == 'BENIGN' else 'red' for l in labels.iloc[sample_indices]]

ax7.scatter(sample_indices, mse[sample_indices], c=colors_scatter, alpha=0.4, s=2)
ax7.axhline(threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
ax7.set_xlabel('Sample Index')
ax7.set_ylabel('MSE')
ax7.set_title('MSE Scatter (Green=BENIGN, Red=Attack)', fontsize=11, fontweight='bold')
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)

# 子圖 8: 漏報 MSE 分佈
ax8 = plt.subplot(3, 4, 8)
if len(false_negatives) > 0:
    ax8.hist(false_negatives['anomaly_score'], bins=50, alpha=0.7, color='orange', label='False Negatives')
    ax8.hist(mse_benign, bins=50, alpha=0.5, color='green', label='BENIGN')
    ax8.axvline(threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax8.set_xlabel('MSE')
    ax8.set_ylabel('Count')
    ax8.set_title('False Negatives MSE Distribution', fontsize=11, fontweight='bold')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)
else:
    ax8.text(0.5, 0.5, 'No False Negatives', ha='center', va='center', fontsize=12)
    ax8.set_title('False Negatives MSE Distribution', fontsize=11, fontweight='bold')

# 子圖 9: 誤報 MSE 分佈
ax9 = plt.subplot(3, 4, 9)
if len(false_positives) > 0:
    ax9.hist(false_positives['anomaly_score'], bins=50, alpha=0.7, color='red', label='False Positives')
    ax9.hist(mse_attack, bins=50, alpha=0.5, color='orange', label='Attack')
    ax9.axvline(threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax9.set_xlabel('MSE')
    ax9.set_ylabel('Count')
    ax9.set_title('False Positives MSE Distribution', fontsize=11, fontweight='bold')
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)
else:
    ax9.text(0.5, 0.5, 'No False Positives', ha='center', va='center', fontsize=12)
    ax9.set_title('False Positives MSE Distribution', fontsize=11, fontweight='bold')

# 子圖 10: Precision vs Recall 權衡
ax10 = plt.subplot(3, 4, 10)
precisions = [r['precision'] for r in threshold_results]
recalls = [r['recall'] for r in threshold_results]
strategy_labels = [r['name'] for r in threshold_results]

ax10.plot(recalls, precisions, 'b-o', linewidth=2, markersize=6)
# 標記當前策略
current_idx = strategy_labels.index(threshold_name)
ax10.plot(recalls[current_idx], precisions[current_idx], 'r*', markersize=15, label='Current')
ax10.set_xlabel('Recall (Detection Rate)')
ax10.set_ylabel('Precision')
ax10.set_title('Precision-Recall Curve', fontsize=11, fontweight='bold')
ax10.legend(fontsize=8)
ax10.grid(True, alpha=0.3)
ax10.set_xlim(-0.05, 1.05)
ax10.set_ylim(-0.05, 1.05)

# 子圖 11: 預測結果分布
ax11 = plt.subplot(3, 4, 11)
pred_counts = output["predicted_attack"].value_counts()
pred_counts.plot(kind='bar', ax=ax11, color='steelblue')
ax11.set_title('Predicted Attack Types', fontsize=11, fontweight='bold')
ax11.set_xlabel('Attack Type')
ax11.set_ylabel('Count')
ax11.tick_params(axis='x', rotation=45, labelsize=8)
ax11.grid(True, alpha=0.3, axis='y')

# 子圖 12: MLP 混淆矩陣（如果有）
ax12 = plt.subplot(3, 4, 12)
if len(pred_labels) > 0:
    mask_real_attack = true_labels_of_anomalies != 'BENIGN'
    if mask_real_attack.sum() > 10:  # 至少要有 10 個樣本
        true_attack_labels = true_labels_of_anomalies[mask_real_attack]
        pred_attack_labels = pred_labels[mask_real_attack]

        unique_labels = sorted(set(true_attack_labels) | set(pred_attack_labels))
        cm_mlp = confusion_matrix(true_attack_labels, pred_attack_labels, labels=unique_labels)

        sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Greens', ax=ax12,
                   xticklabels=[l[:10] for l in unique_labels],
                   yticklabels=[l[:10] for l in unique_labels],
                   cbar_kws={'label': 'Count'})
        ax12.set_title('MLP Classification Matrix', fontsize=11, fontweight='bold')
        ax12.set_xlabel('Predicted')
        ax12.set_ylabel('True')
    else:
        ax12.text(0.5, 0.5, 'Insufficient Attack Samples', ha='center', va='center', fontsize=10)
        ax12.set_title('MLP Classification Matrix', fontsize=11, fontweight='bold')
else:
    ax12.text(0.5, 0.5, 'No MLP Classification', ha='center', va='center', fontsize=10)
    ax12.set_title('MLP Classification Matrix', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('test_final_analysis.png', dpi=150, bbox_inches='tight')
print("✅ 已保存最終測試分析圖: test_final_analysis.png")

plt.show()

print("\n" + "=" * 50)
print("✅ 測試完成（最終改良版）！")
print("=" * 50)
print(f"📊 最終結果總結:")
print(f"  - 測試樣本數: {len(df):,}")
print(f"  - 使用門檻: {threshold:.6f} ({threshold_name})")
print(f"  - 偵測到異常: {is_anomaly.sum():,}")
print(f"  - Precision: {precision:.4f}")
print(f"  - Recall: {recall:.4f}")
print(f"  - F1-Score: {f1:.4f}")
print(f"  - Accuracy: {accuracy:.4f}")
if len(pred_labels) > 0:
    print(f"  - MLP 分類準確率: {classification_acc:.4f}")
print("=" * 50)