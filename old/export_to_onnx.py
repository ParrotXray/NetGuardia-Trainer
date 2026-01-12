"""
NetGuardia 模型導出工具
- 將所有 Keras/sklearn 模型轉為 ONNX
- 將所有預處理參數導出為 JSON
- 供 Rust 程式載入使用
"""
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tf2onnx
import onnx
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

print("=" * 60)
print("🚀 NetGuardia 模型導出工具")
print("=" * 60)

# ============================================================
# 1️⃣ 載入所有模型和配置
# ============================================================
print("\n📦 載入模型和配置...")

try:
    deep_ae = tf.keras.models.load_model("../deep_autoencoder.keras")
    print("✅ Deep Autoencoder 載入")
except Exception as e:
    print(f"❌ 無法載入 Deep Autoencoder: {e}")
    exit(1)

try:
    rf = joblib.load("../random_forest.pkl")
    print("✅ Random Forest 載入")
except Exception as e:
    print(f"❌ 無法載入 Random Forest: {e}")
    exit(1)

try:
    mlp = tf.keras.models.load_model("../mlp_improved.keras")
    le = joblib.load("../label_encoder_improved.pkl")
    mlp_name = "mlp_improved"
    print("✅ MLP Improved 載入")
except:
    try:
        mlp = tf.keras.models.load_model("mlp_attack_classifier.keras")
        le = joblib.load("label_encoder.pkl")
        mlp_name = "mlp_classifier"
        print("✅ MLP Classifier 載入")
    except Exception as e:
        print(f"❌ 無法載入 MLP: {e}")
        exit(1)

try:
    config = joblib.load("../deep_ae_ensemble_config.pkl")
    scaler = config['scaler']
    clip_params = config['clip_params']
    best_strategy = config['best']
    ae_normalization = config.get('ae_normalization', None)  # 🔥 載入 AE 正規化參數
    print("✅ Ensemble 配置載入")

    if ae_normalization:
        print(f"   ✅ AE 正規化參數已找到:")
        print(f"      Min: {ae_normalization['min']:.6f}")
        print(f"      Max: {ae_normalization['max']:.6f}")
        print(f"      Mean: {ae_normalization['mean']:.6f}")
        print(f"      Std: {ae_normalization['std']:.6f}")
    else:
        print("   ⚠️  警告：配置中缺少 ae_normalization，請重新執行 ensemble.py")

except Exception as e:
    print(f"❌ 無法載入配置: {e}")
    exit(1)

# ============================================================
# 2️⃣ 轉換 Deep Autoencoder 為 ONNX
# ============================================================
print("\n" + "=" * 60)
print("🔄 轉換 Deep Autoencoder → ONNX")
print("=" * 60)

input_dim = deep_ae.input.shape[1]
spec = (tf.TensorSpec((None, input_dim), tf.float32, name="input"),)

model_proto, _ = tf2onnx.convert.from_keras(
    deep_ae,
    input_signature=spec,
    opset=13
)

onnx.save(model_proto, "../deep_autoencoder.onnx")
print("✅ 已儲存: deep_autoencoder.onnx")

# 驗證
onnx_model = onnx.load("../deep_autoencoder.onnx")
onnx.checker.check_model(onnx_model)
print("✅ ONNX 模型驗證通過")

# ============================================================
# 3️⃣ 轉換 Random Forest 為 ONNX
# ============================================================
print("\n" + "=" * 60)
print("🔄 轉換 Random Forest → ONNX")
print("=" * 60)

n_features = len(clip_params)
initial_type = [('float_input', FloatTensorType([None, n_features]))]

onx = convert_sklearn(
    rf,
    initial_types=initial_type,
    target_opset=13
)

with open("../random_forest.onnx", "wb") as f:
    f.write(onx.SerializeToString())

print("✅ 已儲存: random_forest.onnx")

# 驗證
onnx_model = onnx.load("../random_forest.onnx")
onnx.checker.check_model(onnx_model)
print("✅ ONNX 模型驗證通過")

# ============================================================
# 4️⃣ 轉換 MLP Classifier 為 ONNX
# ============================================================
print("\n" + "=" * 60)
print("🔄 轉換 MLP Classifier → ONNX")
print("=" * 60)

mlp_input_dim = mlp.input_shape[1]
spec = (tf.TensorSpec((None, mlp_input_dim), tf.float32, name="input"),)

model_proto, _ = tf2onnx.convert.from_keras(
    mlp,
    input_signature=spec,
    opset=13
)

mlp_onnx_filename = f"{mlp_name}.onnx"
onnx.save(model_proto, mlp_onnx_filename)
print(f"✅ 已儲存: {mlp_onnx_filename}")

# 驗證
onnx_model = onnx.load(mlp_onnx_filename)
onnx.checker.check_model(onnx_model)
print("✅ ONNX 模型驗證通過")

# ============================================================
# 5️⃣ 導出預處理參數為 JSON
# ============================================================
print("\n" + "=" * 60)
print("📝 導出預處理參數 → JSON")
print("=" * 60)

# 整理 Scaler 參數
scaler_params = {
    "mean": scaler.mean_.tolist(),
    "std": scaler.scale_.tolist(),
    "feature_names": scaler.feature_names_in_.tolist() if hasattr(scaler, 'feature_names_in_') else []
}

# 整理 Clip 參數
clip_params_json = {}
for col, params in clip_params.items():
    clip_params_json[col] = {
        "lower": float(params['lower']),
        "upper": float(params['upper'])
    }

# 整理 Ensemble 參數
ensemble_params = {
    "strategy_name": best_strategy['name'],
    "threshold": float(best_strategy['threshold']),
    "tpr": float(best_strategy['tpr']),
    "fpr": float(best_strategy['fpr']),
    "precision": float(best_strategy['precision']),  # 🔥 修正這裡
    "f1": float(best_strategy['f1'])
}

# 🔥 整理 AE 正規化參數
if ae_normalization:
    ae_normalization_json = {
        "min": float(ae_normalization['min']),
        "max": float(ae_normalization['max']),
        "mean": float(ae_normalization['mean']),
        "std": float(ae_normalization['std']),
        "median": float(ae_normalization.get('median', 0.0)),
        "p90": float(ae_normalization.get('p90', 0.0)),
        "p95": float(ae_normalization.get('p95', 0.0)),
        "p99": float(ae_normalization.get('p99', 0.0))
    }
else:
    print("⚠️  使用預設的 AE 正規化參數（不建議）")
    ae_normalization_json = {
        "min": 0.0,
        "max": 1.0,
        "mean": 0.0,
        "std": 1.0,
        "median": 0.0,
        "p90": 0.0,
        "p95": 0.0,
        "p99": 0.0
    }

# 整理攻擊類型映射
attack_labels = {
    str(i): label for i, label in enumerate(le.classes_)
}

# 組合所有配置
config_json = {
    "version": "1.0.0",
    "created_at": pd.Timestamp.now().isoformat(),

    "models": {
        "deep_autoencoder": {
            "file": "deep_autoencoder.onnx",
            "input_dim": int(input_dim),
            "encoding_dim": int(config.get('encoding_dim', 16))
        },
        "random_forest": {
            "file": "random_forest.onnx",
            "n_estimators": int(rf.n_estimators),
            "n_features": int(n_features)
        },
        "mlp_classifier": {
            "file": mlp_onnx_filename,
            "input_dim": int(mlp_input_dim),
            "n_classes": int(len(le.classes_))
        }
    },

    "preprocessing": {
        "clip_params": clip_params_json,
        "scaler": scaler_params,
        "post_scaling_clip": {
            "min": -5.0,
            "max": 5.0
        }
    },

    "ensemble": ensemble_params,

    "ae_normalization": ae_normalization_json,  # 🔥 使用實際參數

    "attack_labels": attack_labels,

    "feature_order": scaler_params["feature_names"]
}

# 儲存為 JSON
with open("../netguardia_config.json", "w", encoding='utf-8') as f:
    json.dump(config_json, f, indent=2, ensure_ascii=False)

print("✅ 已儲存: netguardia_config.json")

# 同時儲存一個精簡版（只包含推論所需參數）
inference_config = {
    "threshold": ensemble_params["threshold"],
    "strategy_name": ensemble_params["strategy_name"],
    "clip_params": clip_params_json,
    "scaler_mean": scaler_params["mean"],
    "scaler_std": scaler_params["std"],
    "post_clip_min": -5.0,
    "post_clip_max": 5.0,
    "ae_normalization": ae_normalization_json,  # 🔥 加入正規化參數
    "attack_labels": attack_labels,
    "feature_names": scaler_params["feature_names"]
}

with open("../netguardia_inference.json", "w", encoding='utf-8') as f:
    json.dump(inference_config, f, indent=2, ensure_ascii=False)

print("✅ 已儲存: netguardia_inference.json (精簡版)")

# ============================================================
# 6️⃣ 驗證導出的模型
# ============================================================
print("\n" + "=" * 60)
print("🧪 驗證 ONNX 模型")
print("=" * 60)

import onnxruntime as ort

# 生成測試資料
test_input = np.random.randn(1, n_features).astype(np.float32)

# 預處理
for i, col in enumerate(scaler_params["feature_names"]):
    if col in clip_params_json:
        test_input[0, i] = np.clip(
            test_input[0, i],
            clip_params_json[col]['lower'],
            clip_params_json[col]['upper']
        )

# 標準化
test_input_scaled = (test_input - np.array(scaler_params["mean"])) / np.array(scaler_params["std"])
test_input_scaled = np.clip(test_input_scaled, -5, 5).astype(np.float32)

# 測試 Deep AE
print("\n1. 測試 Deep Autoencoder...")
session_ae = ort.InferenceSession("../deep_autoencoder.onnx")
ae_output = session_ae.run(None, {"input": test_input_scaled})[0]
ae_mse = np.mean((test_input_scaled - ae_output) ** 2)
print(f"   ✅ AE MSE: {ae_mse:.6f}")

# 測試 RF
print("\n2. 測試 Random Forest...")
session_rf = ort.InferenceSession("../random_forest.onnx")
rf_output = session_rf.run(None, {"float_input": test_input_scaled})
rf_proba = rf_output[1][0][1]  # probabilities, attack class
print(f"   ✅ RF Attack Probability: {rf_proba:.6f}")

# 測試 MLP
print("\n3. 測試 MLP Classifier...")
session_mlp = ort.InferenceSession(mlp_onnx_filename)
mlp_output = session_mlp.run(None, {"input": test_input_scaled})[0]
predicted_class = np.argmax(mlp_output[0])
confidence = mlp_output[0][predicted_class]
print(f"   ✅ Predicted Class: {predicted_class} ({le.classes_[predicted_class]})")
print(f"   ✅ Confidence: {confidence:.6f}")

# 測試 Ensemble（🔥 使用正確的正規化）
print("\n4. 測試 Ensemble...")
ae_score_norm = (ae_mse - ae_normalization_json['min']) / \
                (ae_normalization_json['max'] - ae_normalization_json['min'] + 1e-10)
ae_score_norm = np.clip(ae_score_norm, 0, 1)  # 裁剪到 [0, 1]
rf_score_norm = rf_proba

print(f"   AE Score (normalized): {ae_score_norm:.6f}")
print(f"   RF Score: {rf_score_norm:.6f}")

# 根據策略計算 ensemble score
if ensemble_params["strategy_name"] == "W_7:3":
    ensemble_score = 0.7 * rf_score_norm + 0.3 * ae_score_norm
elif ensemble_params["strategy_name"] == "W_5:5":
    ensemble_score = 0.5 * rf_score_norm + 0.5 * ae_score_norm
elif ensemble_params["strategy_name"] == "W_3:7":
    ensemble_score = 0.3 * rf_score_norm + 0.7 * ae_score_norm
else:
    ensemble_score = (rf_score_norm + ae_score_norm) / 2.0

is_anomaly = ensemble_score > ensemble_params["threshold"]

print(f"   ✅ Ensemble Score: {ensemble_score:.6f}")
print(f"   ✅ Threshold: {ensemble_params['threshold']:.6f}")
print(f"   ✅ Is Anomaly: {is_anomaly}")

if is_anomaly:
    print(f"   🚨 預測為攻擊: {le.classes_[predicted_class]} (信心度: {confidence:.2%})")
else:
    print(f"   ✅ 預測為正常流量")

# ============================================================
# 7️⃣ 輸出摘要
# ============================================================
print("\n" + "=" * 60)
print("✅ 導出完成！")
print("=" * 60)

print("\n📦 導出的檔案:")
print("  ONNX 模型:")
print("    - deep_autoencoder.onnx")
print("    - random_forest.onnx")
print(f"    - {mlp_onnx_filename}")
print("\n  JSON 配置:")
print("    - netguardia_config.json (完整配置)")
print("    - netguardia_inference.json (推論專用)")

print("\n📊 模型資訊:")
print(f"  Deep AE: {input_dim} 維 → {config.get('encoding_dim', 16)} 維 bottleneck")
print(f"  Random Forest: {rf.n_estimators} 棵樹")
print(f"  MLP: {len(le.classes_)} 個攻擊類別")
print(f"  Ensemble: {ensemble_params['strategy_name']} (threshold={ensemble_params['threshold']:.4f})")

print("\n🎯 性能指標:")
print(f"  TPR: {ensemble_params['tpr']:.2%}")
print(f"  FPR: {ensemble_params['fpr']:.2%}")
print(f"  Precision: {ensemble_params['precision']:.3f}")
print(f"  F1-Score: {ensemble_params['f1']:.3f}")

print("\n🔥 AE 正規化參數:")
print(f"  Min: {ae_normalization_json['min']:.6f}")
print(f"  Max: {ae_normalization_json['max']:.6f}")
print(f"  Mean: {ae_normalization_json['mean']:.6f}")
print(f"  Std: {ae_normalization_json['std']:.6f}")

print("\n🚀 下一步:")
print("  1. 將 ONNX 模型和 JSON 配置複製到 Rust 專案")
print("  2. 使用 'ort' crate 載入 ONNX 模型")
print("  3. 使用 'serde_json' 載入 JSON 配置")
print("  4. 確保 Rust 使用相同的 AE 正規化參數")

print("=" * 60)