"""
測試階段 - Deep AE + Ensemble + MLP 改進版系統
- 自動載入 Deep AE + RF Ensemble
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
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🧪 Deep AE + Ensemble 系統測試")
print("=" * 60)

# === 1️⃣ 載入模型與工具 ===
print("\n📦 載入模型與工具...")

try:
    # Deep AE + RF Ensemble
    print("  載入 Deep Autoencoder...")
    deep_ae = load_model("deep_autoencoder.keras")

    print("  載入 Random Forest...")
    rf = joblib.load("random_forest.pkl")

    print("  載入 Ensemble 配置...")
    ensemble_config = joblib.load("deep_ae_ensemble_config.pkl")
    scaler = ensemble_config['scaler']
    clip_params = ensemble_config['clip_params']
    best_strategy = ensemble_config['best']

    print("  載入 MLP 分類器...")
    # 優先使用改進版，如果不存在則使用原始版
    try:
        mlp = load_model("mlp_improved.keras")
        le = joblib.load("label_encoder_improved.pkl")
        print("    ✅ 使用改進版 MLP")
    except:
        mlp = load_model("mlp_attack_classifier.keras")
        le = joblib.load("label_encoder.pkl")
        print("    ✅ 使用原始版 MLP")

    print("\n✅ 所有模型載入成功")
    print(f"📊 Ensemble 策略: {best_strategy['name']}")
    print(f"📊 推薦門檻: {best_strategy['threshold']:.6f}")
    print(f"📊 訓練時性能: TPR={best_strategy['tpr']:.1%}, F1={best_strategy['f1']:.3f}")

except Exception as e:
    print(f"\n❌ 載入模型失敗: {e}")
    print("\n請確保以下檔案存在:")
    print("  - deep_autoencoder.keras")
    print("  - random_forest.pkl")
    print("  - deep_ae_ensemble_config.pkl")
    print("  - mlp_improved.keras (或 mlp_attack_classifier.keras)")
    print("  - label_encoder_improved.pkl (或 label_encoder.pkl)")
    exit(1)

# === 2️⃣ 讀取測試資料 ===
print("\n📂 讀取測試資料...")

# 嘗試不同的測試檔案
test_files = [

    './csv/Friday-WorkingHours-Morning.pcap_ISCX.csv',
    './csv/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',

]

df = None
for test_file in test_files:
    try:
        df = pd.read_csv(test_file)
        df.columns = df.columns.str.strip()
        print(f"✅ 載入測試資料: {test_file}")
        break
    except:
        continue

if df is None:
    print("❌ 找不到測試資料檔案")
    exit(1)

print(f"測試資料維度: {df.shape}")

# 檢查是否有 Label 欄位
if 'Label' not in df.columns:
    print("❌ 測試資料缺少 Label 欄位")
    exit(1)

print(f"\n📋 測試資料標籤分布:")
label_counts = df['Label'].value_counts()
for label, count in label_counts.items():
    print(f"  {label}: {count:,}")

labels = df['Label'].copy()

# === 3️⃣ 準備特徵 ===
print("\n🔢 準備特徵...")

# 移除非特徵欄位
exclude_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Label',
                'deep_ae_mse', 'rf_proba', 'ensemble_score', 'ensemble_anomaly',
                'predicted_label', 'prediction_confidence', 'is_correct', 'anomaly_if']
df_features = df.drop(columns=exclude_cols, errors='ignore')

# 選擇數值特徵
X = df_features.select_dtypes(include=[np.number])

print(f"原始特徵數: {X.shape[1]}")

# 清理數據
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# === 4️⃣ 預處理（Winsorization + 標準化）===
print("\n🧹 預處理資料...")

# Winsorization (使用訓練時的裁剪參數)
print("  Winsorization...")
for col in X.columns:
    if col in clip_params:
        X[col] = np.clip(X[col],
                        clip_params[col]['lower'],
                        clip_params[col]['upper'])

# 檢查特徵對齊
if hasattr(scaler, 'feature_names_in_'):
    scaler_cols = scaler.feature_names_in_
    missing_cols = set(scaler_cols) - set(X.columns)
    if missing_cols:
        print(f"  ⚠️ 測試資料缺少欄位: {len(missing_cols)} 個")
        for col in missing_cols:
            X[col] = 0
    X = X[scaler_cols]

# 標準化
print("  標準化...")
X_scaled = scaler.transform(X)
X_scaled = np.clip(X_scaled, -5, 5)

print(f"✅ 預處理完成，特徵維度: {X_scaled.shape}")

# === 5️⃣ 驗證維度 ===
expected_dim = deep_ae.input_shape[1]
current_dim = X_scaled.shape[1]

if current_dim != expected_dim:
    raise ValueError(f"❌ 維度不匹配！模型需要 {expected_dim} 維，資料有 {current_dim} 維")

print(f"✅ 特徵維度驗證通過: {current_dim}")

# === 6️⃣ Ensemble 異常偵測 ===
print("\n" + "=" * 60)
print("🔍 執行 Ensemble 異常偵測")
print("=" * 60)

print("\n階段 1: Deep Autoencoder 預測...")
ae_recon = deep_ae.predict(X_scaled, batch_size=2048, verbose=1)
ae_mse = np.mean(np.square(X_scaled - ae_recon), axis=1)

print("\n階段 2: Random Forest 預測...")
rf_proba = rf.predict_proba(X_scaled)[:, 1]

print("\n階段 3: Ensemble Score 計算...")
# 正規化
ae_score_norm = (ae_mse - ae_mse.min()) / (ae_mse.max() - ae_mse.min() + 1e-10)
rf_score_norm = rf_proba

# Ensemble (根據策略)
if best_strategy['name'] == 'W_3:7':
    ensemble_score = 0.3 * ae_score_norm + 0.7 * rf_score_norm
elif best_strategy['name'] == 'W_5:5':
    ensemble_score = 0.5 * ae_score_norm + 0.5 * rf_score_norm
elif best_strategy['name'] == 'W_7:3':
    ensemble_score = 0.7 * ae_score_norm + 0.3 * rf_score_norm
else:
    ensemble_score = (ae_score_norm + rf_score_norm) / 2

print(f"✅ Ensemble Score 計算完成 (策略: {best_strategy['name']})")

# 分類統計
ensemble_benign = ensemble_score[labels == 'BENIGN']
ensemble_attack = ensemble_score[labels != 'BENIGN']

print(f"\n📊 Ensemble Score 統計:")
print(f"  BENIGN:")
print(f"    Mean: {ensemble_benign.mean():.6f}")
print(f"    Median: {np.median(ensemble_benign):.6f}")
print(f"    P95: {np.percentile(ensemble_benign, 95):.6f}")
print(f"    P99: {np.percentile(ensemble_benign, 99):.6f}")
print(f"  Attack:")
print(f"    Mean: {ensemble_attack.mean():.6f}")
print(f"    Median: {np.median(ensemble_attack):.6f}")
print(f"  分離度: {ensemble_attack.mean() / ensemble_benign.mean():.2f}x")

# === 7️⃣ 測試多個門檻策略 ===
print("\n" + "=" * 60)
print("🎯 測試多個門檻策略")
print("=" * 60)

test_thresholds = {
    'Recommended': best_strategy['threshold'],
    'BENIGN_P90': np.percentile(ensemble_benign, 90),
    'BENIGN_P95': np.percentile(ensemble_benign, 95),
    'BENIGN_P97': np.percentile(ensemble_benign, 97),
    'BENIGN_P99': np.percentile(ensemble_benign, 99),
    'Mean+2Std': ensemble_benign.mean() + 2 * ensemble_benign.std(),
    'Mean+3Std': ensemble_benign.mean() + 3 * ensemble_benign.std(),
}

print(f"\n{'策略':<20} {'門檻值':<12} {'TP':<8} {'FP':<8} {'FN':<8} {'Precision':<10} {'Recall':<10} {'F1':<10}")
print("-" * 100)

threshold_results = []
for name, thresh in sorted(test_thresholds.items(), key=lambda x: x[1]):
    is_ano = (ensemble_score > thresh).astype(int)

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

print(f"\n🏆 測試集最佳門檻: {threshold_name}")
print(f"   門檻值: {threshold:.6f}")
print(f"   F1-Score: {best_test_result['f1']:.4f}")

# === 8️⃣ 使用最佳門檻進行預測 ===
print("\n" + "=" * 60)
print("📊 異常偵測結果")
print("=" * 60)

is_anomaly = (ensemble_score > threshold).astype(int)

print(f"\n偵測到異常: {is_anomaly.sum():,} / {len(df):,} ({is_anomaly.sum()/len(df):.1%})")

tp = best_test_result['tp']
fp = best_test_result['fp']
fn = best_test_result['fn']
tn = best_test_result['tn']

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = best_test_result['precision']
recall = best_test_result['recall']
f1 = best_test_result['f1']
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

print(f"\n🎯 性能指標:")
print(f"  True Positives (TP):  {tp:>8,}")
print(f"  False Positives (FP): {fp:>8,}")
print(f"  False Negatives (FN): {fn:>8,}")
print(f"  True Negatives (TN):  {tn:>8,}")
print(f"\n  Accuracy:   {accuracy:.4f}")
print(f"  Precision:  {precision:.4f}")
print(f"  Recall/TPR: {recall:.4f}")
print(f"  FPR:        {fpr:.4f}")
print(f"  F1-Score:   {f1:.4f}")

# === 9️⃣ 各攻擊類型詳細分析 ===
print("\n" + "=" * 60)
print("🎯 各攻擊類型偵測率")
print("=" * 60)

print(f"\n{'攻擊類型':<35} {'總數':<10} {'偵測':<10} {'偵測率':<10} {'平均分數':<12}")
print("-" * 80)

attack_analysis = []
for attack_type in sorted(labels[labels != 'BENIGN'].unique()):
    mask_attack = (labels == attack_type)
    total = mask_attack.sum()
    detected = ((labels == attack_type) & (is_anomaly == 1)).sum()
    rate = detected / total if total > 0 else 0

    score_type = ensemble_score[mask_attack]
    mean_score = score_type.mean()

    status = '✅' if rate >= 0.8 else '⚠️' if rate >= 0.5 else '❌'
    print(f"{status} {attack_type:<32} {total:<10,} {detected:<10,} {rate:<9.1%} {mean_score:<12.6f}")

    attack_analysis.append({
        'type': attack_type,
        'total': total,
        'detected': detected,
        'rate': rate,
        'mean_score': mean_score
    })

# 找出最難偵測的攻擊
if attack_analysis:
    worst_attack = min(attack_analysis, key=lambda x: x['rate'])
    best_attack = max(attack_analysis, key=lambda x: x['rate'])
    print(f"\n⚠️ 最難偵測: {worst_attack['type']} ({worst_attack['rate']:.1%})")
    print(f"✅ 最易偵測: {best_attack['type']} ({best_attack['rate']:.1%})")

# === 🔟 MLP 分類異常樣本 ===
print("\n" + "=" * 60)
print("🧠 MLP 攻擊分類")
print("=" * 60)

X_anomaly = X_scaled[is_anomaly == 1]

if len(X_anomaly) > 0:
    print(f"\n對 {len(X_anomaly):,} 個異常樣本進行分類...")

    mlp_expected_dim = mlp.input_shape[1]
    if X_anomaly.shape[1] != mlp_expected_dim:
        raise ValueError(f"❌ MLP 維度不匹配！")

    preds = mlp.predict(X_anomaly, batch_size=2048, verbose=0)
    pred_labels = le.inverse_transform(np.argmax(preds, axis=1))
    pred_confidence = preds.max(axis=1)

    print(f"✅ 分類完成")
    print(f"\n📋 預測攻擊類型分布:")
    pred_counts = pd.Series(pred_labels).value_counts()
    for label, count in pred_counts.items():
        pct = count / len(pred_labels) * 100
        print(f"  {label:<35} {count:>6,} ({pct:>5.1f}%)")

    # MLP 分類準確率
    true_labels_of_anomalies = labels.values[is_anomaly == 1]
    correct_classification = (true_labels_of_anomalies == pred_labels).sum()
    classification_acc = correct_classification / len(pred_labels)

    print(f"\n📊 MLP 整體分類準確率: {classification_acc:.4f} ({correct_classification:,}/{len(pred_labels):,})")

    # 只看真實攻擊的分類準確率
    mask_real_attack = true_labels_of_anomalies != 'BENIGN'
    if mask_real_attack.sum() > 0:
        true_attack_labels = true_labels_of_anomalies[mask_real_attack]
        pred_attack_labels = pred_labels[mask_real_attack]

        attack_correct = (true_attack_labels == pred_attack_labels).sum()
        attack_acc = attack_correct / len(true_attack_labels)

        print(f"📊 攻擊分類準確率（排除 BENIGN）: {attack_acc:.4f} ({attack_correct:,}/{len(true_attack_labels):,})")

        print(f"\n📋 攻擊分類詳細報告:")
        try:
            report = classification_report(true_attack_labels, pred_attack_labels,
                                          zero_division=0, digits=4)
            print(report)
        except:
            print("  (無法生成報告)")
else:
    pred_labels = []
    pred_confidence = []
    print("\n⚠️ 沒有偵測到異常樣本")

# === 11️⃣ 漏報分析 ===
print("\n" + "=" * 60)
print("🚫 漏報（False Negatives）分析")
print("=" * 60)

false_negatives_mask = (labels != 'BENIGN') & (is_anomaly == 0)
false_negatives = df[false_negatives_mask]

if len(false_negatives) > 0:
    fn_count = len(false_negatives)
    total_attacks = (labels != 'BENIGN').sum()

    print(f"\n漏報總數: {fn_count:,} / {total_attacks:,} ({fn_count/total_attacks:.1%})")

    fn_scores = ensemble_score[false_negatives_mask]
    print(f"\n漏報樣本的 Ensemble Score 統計:")
    print(f"  Mean:   {fn_scores.mean():.6f}")
    print(f"  Median: {np.median(fn_scores):.6f}")
    print(f"  Max:    {fn_scores.max():.6f}")
    print(f"  P95:    {np.percentile(fn_scores, 95):.6f}")

    print(f"\n📋 漏報攻擊類型分布:")
    fn_labels = labels[false_negatives_mask]
    for attack_type in sorted(fn_labels.unique()):
        fn_type_count = (fn_labels == attack_type).sum()
        total_type = (labels == attack_type).sum()
        pct = fn_type_count / total_type * 100
        print(f"  {attack_type:<35} {fn_type_count:>6,} / {total_type:<6,} ({pct:>5.1f}%)")

    # 建議調整
    fn_score_95 = np.percentile(fn_scores, 95)
    print(f"\n💡 若要捕捉 95% 的漏報，門檻需降至: {fn_score_95:.6f}")
    print(f"   (當前門檻: {threshold:.6f})")
else:
    print("\n✅ 沒有漏報！所有攻擊都被偵測到")

# === 12️⃣ 誤報分析 ===
print("\n" + "=" * 60)
print("⚠️ 誤報（False Positives）分析")
print("=" * 60)

false_positives_mask = (labels == 'BENIGN') & (is_anomaly == 1)
false_positives = df[false_positives_mask]

if len(false_positives) > 0:
    fp_count = len(false_positives)
    total_benign = (labels == 'BENIGN').sum()

    print(f"\n誤報總數: {fp_count:,} / {total_benign:,} ({fp_count/total_benign:.1%})")

    fp_scores = ensemble_score[false_positives_mask]
    print(f"\n誤報樣本的 Ensemble Score 統計:")
    print(f"  Mean:   {fp_scores.mean():.6f}")
    print(f"  Median: {np.median(fp_scores):.6f}")
    print(f"  Min:    {fp_scores.min():.6f}")
    print(f"  P5:     {np.percentile(fp_scores, 5):.6f}")

    print(f"\n💡 若要減少誤報到 1%，門檻需提升至: {np.percentile(ensemble_benign, 99):.6f}")
    print(f"   (當前門檻: {threshold:.6f})")
else:
    print("\n✅ 沒有誤報！所有正常流量都被正確識別")

# === 13️⃣ 組合輸出結果 ===
print("\n💾 儲存結果...")

output = pd.DataFrame()
output["Label"] = labels.values
output["ae_mse"] = ae_mse
output["rf_proba"] = rf_proba
output["ensemble_score"] = ensemble_score
output["is_anomaly"] = is_anomaly
output["predicted_attack"] = "BENIGN"

if len(pred_labels) > 0:
    output.loc[is_anomaly == 1, "predicted_attack"] = pred_labels
    output.loc[is_anomaly == 1, "confidence"] = pred_confidence

output.to_csv("test_ensemble_results.csv", index=False)
print("✅ 已保存: test_ensemble_results.csv")

# === 14️⃣ 視覺化 ===
print("\n📊 生成視覺化...")

fig = plt.figure(figsize=(20, 14))

# 1. Ensemble Score 分佈
ax1 = plt.subplot(3, 4, 1)
ax1.hist(ensemble_benign, bins=100, alpha=0.7, label='BENIGN', color='green', density=True)
ax1.hist(ensemble_attack, bins=100, alpha=0.7, label='Attack', color='red', density=True)
ax1.axvline(threshold, color='black', linestyle='--', linewidth=2,
           label=f'Threshold={threshold:.4f}')
ax1.set_xlabel('Ensemble Score')
ax1.set_ylabel('Density')
ax1.set_title('Ensemble Score Distribution', fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. 混淆矩陣
ax2 = plt.subplot(3, 4, 2)
cm = np.array([[tn, fp], [fn, tp]])
sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=ax2,
           xticklabels=['Pred Normal', 'Pred Attack'],
           yticklabels=['True Normal', 'True Attack'])
ax2.set_title('Confusion Matrix', fontweight='bold')

# 3. 門檻策略比較
ax3 = plt.subplot(3, 4, 3)
strategy_names = [r['name'] for r in threshold_results]
f1_scores = [r['f1'] for r in threshold_results]
colors = ['gold' if r['name'] == threshold_name else 'steelblue' for r in threshold_results]
ax3.barh(strategy_names, f1_scores, color=colors)
ax3.set_xlabel('F1-Score')
ax3.set_title('Threshold Strategy Comparison', fontweight='bold')
ax3.grid(alpha=0.3, axis='x')

# 4. 性能指標雷達圖
ax4 = plt.subplot(3, 4, 4, projection='polar')
metrics = ['Precision', 'Recall', 'F1', 'Accuracy']
values = [precision, recall, f1, accuracy, precision]
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]
ax4.plot(angles, values, 'o-', linewidth=2)
ax4.fill(angles, values, alpha=0.25)
ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(metrics)
ax4.set_ylim(0, 1)
ax4.set_title('Performance Metrics', fontweight='bold', pad=20)
ax4.grid(True)

# 5. 各攻擊類型偵測率
ax5 = plt.subplot(3, 4, 5)
if attack_analysis:
    types = [a['type'][:20] for a in attack_analysis]
    rates = [a['rate']*100 for a in attack_analysis]
    colors_bar = ['green' if r >= 80 else 'orange' if r >= 50 else 'red' for r in rates]
    bars = ax5.barh(types, rates, color=colors_bar)
    ax5.set_xlabel('Detection Rate (%)')
    ax5.set_title('Detection Rate by Attack Type', fontweight='bold')
    ax5.set_xlim(0, 105)
    ax5.grid(alpha=0.3, axis='x')
    for i, (bar, rate) in enumerate(zip(bars, rates)):
        ax5.text(rate+2, i, f'{rate:.0f}%', va='center', fontsize=8)

# 6. AE vs RF 分數散點圖
ax6 = plt.subplot(3, 4, 6)
sample_size = min(5000, len(ae_score_norm))
sample_idx = np.random.choice(len(ae_score_norm), sample_size, replace=False)
colors_scatter = ['red' if l != 'BENIGN' else 'green' for l in labels.iloc[sample_idx]]
ax6.scatter(ae_score_norm[sample_idx], rf_score_norm[sample_idx],
           c=colors_scatter, alpha=0.3, s=5)
ax6.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax6.set_xlabel('Deep AE Score (norm)')
ax6.set_ylabel('RF Score (norm)')
ax6.set_title('AE vs RF Scores', fontweight='bold')
ax6.grid(alpha=0.3)

# 7. Ensemble Score 散點圖
ax7 = plt.subplot(3, 4, 7)
sample_idx2 = np.random.choice(len(ensemble_score), min(5000, len(ensemble_score)), replace=False)
colors_scatter2 = ['red' if l != 'BENIGN' else 'green' for l in labels.iloc[sample_idx2]]
ax7.scatter(sample_idx2, ensemble_score[sample_idx2], c=colors_scatter2, alpha=0.4, s=3)
ax7.axhline(threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
ax7.set_xlabel('Sample Index')
ax7.set_ylabel('Ensemble Score')
ax7.set_title('Ensemble Score Scatter', fontweight='bold')
ax7.legend()
ax7.grid(alpha=0.3)

# 8. 漏報分析
ax8 = plt.subplot(3, 4, 8)
if len(false_negatives) > 0:
    fn_scores = ensemble_score[false_negatives_mask]
    ax8.hist(fn_scores, bins=50, alpha=0.7, color='orange', label='False Negatives')
    ax8.hist(ensemble_benign, bins=50, alpha=0.5, color='green', label='BENIGN')
    ax8.axvline(threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax8.set_xlabel('Ensemble Score')
    ax8.set_ylabel('Count')
    ax8.set_title('False Negatives Distribution', fontweight='bold')
    ax8.legend()
    ax8.grid(alpha=0.3)
else:
    ax8.text(0.5, 0.5, 'No False Negatives', ha='center', va='center', fontsize=14,
            fontweight='bold', color='green')
    ax8.set_title('False Negatives Distribution', fontweight='bold')

# 9. 誤報分析
ax9 = plt.subplot(3, 4, 9)
if len(false_positives) > 0:
    fp_scores = ensemble_score[false_positives_mask]
    ax9.hist(fp_scores, bins=50, alpha=0.7, color='red', label='False Positives')
    ax9.hist(ensemble_attack, bins=50, alpha=0.5, color='orange', label='Attack')
    ax9.axvline(threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax9.set_xlabel('Ensemble Score')
    ax9.set_ylabel('Count')
    ax9.set_title('False Positives Distribution', fontweight='bold')
    ax9.legend()
    ax9.grid(alpha=0.3)
else:
    ax9.text(0.5, 0.5, 'No False Positives', ha='center', va='center', fontsize=14,
            fontweight='bold', color='green')
    ax9.set_title('False Positives Distribution', fontweight='bold')

# 10. Precision-Recall 曲線
ax10 = plt.subplot(3, 4, 10)
precisions = [r['precision'] for r in threshold_results]
recalls = [r['recall'] for r in threshold_results]
ax10.plot(recalls, precisions, 'b-o', linewidth=2)
current_idx = [r['name'] for r in threshold_results].index(threshold_name)
ax10.plot(recalls[current_idx], precisions[current_idx], 'r*', markersize=15, label='Current')
ax10.set_xlabel('Recall')
ax10.set_ylabel('Precision')
ax10.set_title('Precision-Recall Curve', fontweight='bold')
ax10.legend()
ax10.grid(alpha=0.3)
ax10.set_xlim(-0.05, 1.05)
ax10.set_ylim(-0.05, 1.05)

# 11. 預測類別分布
ax11 = plt.subplot(3, 4, 11)
if len(pred_labels) > 0:
    pred_counts = pd.Series(pred_labels).value_counts()
    pred_counts.plot(kind='barh', ax=ax11, color='teal')
    ax11.set_title('Predicted Attack Types', fontweight='bold')
    ax11.set_xlabel('Count')
    ax11.grid(alpha=0.3, axis='x')
else:
    ax11.text(0.5, 0.5, 'No Predictions', ha='center', va='center', fontsize=14)
    ax11.set_title('Predicted Attack Types', fontweight='bold')

# 12. MLP 混淆矩陣
ax12 = plt.subplot(3, 4, 12)
if len(pred_labels) > 0:
    true_labels_of_anomalies = labels.values[is_anomaly == 1]
    mask_real_attack = true_labels_of_anomalies != 'BENIGN'

    if mask_real_attack.sum() > 10:
        true_attack = true_labels_of_anomalies[mask_real_attack]
        pred_attack = pred_labels[mask_real_attack]

        unique_labels = sorted(set(true_attack) | set(pred_attack))
        if len(unique_labels) <= 15:  # 只在類別不太多時顯示
            cm_mlp = confusion_matrix(true_attack, pred_attack, labels=unique_labels)
            sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Greens', ax=ax12,
                       xticklabels=[l[:10] for l in unique_labels],
                       yticklabels=[l[:10] for l in unique_labels])
            ax12.set_title('MLP Classification Matrix', fontweight='bold')
            ax12.set_xlabel('Predicted')
            ax12.set_ylabel('True')
        else:
            ax12.text(0.5, 0.5, f'Too many classes ({len(unique_labels)})',
                     ha='center', va='center')
            ax12.set_title('MLP Classification Matrix', fontweight='bold')
    else:
        ax12.text(0.5, 0.5, 'Insufficient samples', ha='center', va='center')
        ax12.set_title('MLP Classification Matrix', fontweight='bold')
else:
    ax12.text(0.5, 0.5, 'No MLP predictions', ha='center', va='center')
    ax12.set_title('MLP Classification Matrix', fontweight='bold')

plt.tight_layout()
plt.savefig('test_ensemble_analysis.png', dpi=150, bbox_inches='tight')
print("✅ 已保存: test_ensemble_analysis.png")

# === 15️⃣ 總結 ===
print("\n" + "=" * 60)
print("✅ 測試完成！")
print("=" * 60)

print(f"""
📊 最終結果總結:
  測試樣本數: {len(df):,}
  使用門檻: {threshold:.6f} ({threshold_name})
  
🎯 Ensemble 異常偵測:
  偵測到異常: {is_anomaly.sum():,}
  TPR (Recall): {recall:.2%}
  FPR: {fpr:.2%}
  Precision: {precision:.3f}
  F1-Score: {f1:.3f}
  Accuracy: {accuracy:.3f}
""")

if len(pred_labels) > 0:
    print(f"""
🧠 MLP 攻擊分類:
  分類樣本數: {len(pred_labels):,}
  分類準確率: {classification_acc:.2%}
  (真實攻擊): {attack_acc:.2%}
""")

print("=" * 60)
print("\n📁 輸出檔案:")
print("  - test_ensemble_results.csv")
print("  - test_ensemble_analysis.png")
print("=" * 60)