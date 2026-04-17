import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

# ===== 路径 =====
good_path = "Data_good"
bad_path = "Data_bad"

# ===== 读取 CSV =====
def load_signal(file_path):
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)  # 没表头改0

    time = data[:, 0]
    disp = data[:, 1]

    dt = np.mean(np.diff(time))
    fs = 1.0 / (dt + 1e-12)

    return disp, fs

# ===== 预处理 =====
def preprocess(sig):
    sig = sig - np.mean(sig)
    sig = sig / (np.std(sig) + 1e-8)
    return sig

# ===== 特征提取（稳定版）=====
def extract_features(sig, fs):
    fft = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), 1/fs)

    peak_freq = freqs[np.argmax(fft)]

    mask = (freqs > 0.8) & (freqs < 2)
    energy_ratio = np.sum(fft[mask]) / (np.sum(fft) + 1e-8)

    std = np.std(sig)
    rms = np.sqrt(np.mean(sig**2))
    peak2peak = np.max(sig) - np.min(sig)

    sharpness = np.max(fft) / (np.mean(fft) + 1e-8)

    return [peak_freq, energy_ratio, std, rms, peak2peak, sharpness]

# ===== 构建数据集 =====
X = []
y = []

def process_folder(folder, label):
    for file in os.listdir(folder):
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(folder, file)

        try:
            sig, fs = load_signal(file_path)

            if len(sig) < 1000:
                continue

            sig = preprocess(sig)
            feat = extract_features(sig, fs)

            X.append(feat)
            y.append(label)

        except Exception as e:
            print("跳过:", file, e)

process_folder(good_path, 1)
process_folder(bad_path, 0)

X = np.array(X)
y = np.array(y)

print("数据集大小:", X.shape)
print("好数据:", np.sum(y==1), "坏数据:", np.sum(y==0))

# ===== 模型 =====
model = RandomForestClassifier(
    n_estimators=120,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# ===== 交叉验证 =====
scores = cross_val_score(model, X, y, cv=5)
print("\n交叉验证准确率:", scores)
print("平均准确率:", scores.mean())
print("波动:", scores.std())

# ===== 单次划分（用于分析）=====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model.fit(X_train, y_train)

# ===== 概率输出 =====
y_prob = model.predict_proba(X_test)[:, 1]

# ===== 可调阈值 =====
threshold = 0.5   # 👉 可以改成 0.6 / 0.7（更严格）
y_pred = (y_prob > threshold).astype(int)

# ===== 混淆矩阵 =====
cm = confusion_matrix(y_test, y_pred)
print("\n混淆矩阵:\n", cm)

TN, FP, FN, TP = cm.ravel()

print("\n===== 详细分析 =====")
print("坏 → 坏 (TN):", TN)
print("坏 → 好 (FP)【误判❗】:", FP)
print("好 → 坏 (FN)【漏检】:", FN)
print("好 → 好 (TP):", TP)

# ===== 比率 =====
fp_rate = FP / (FP + TN + 1e-8)
fn_rate = FN / (FN + TP + 1e-8)

print("\n误判率（坏→好，危险）:", fp_rate)
print("漏检率（好→坏）:", fn_rate)

# ===== 准确率 =====
acc = accuracy_score(y_test, y_pred)
print("\n当前阈值 =", threshold)
print("测试集准确率:", acc)

model.fit(X, y)

joblib.dump(model, "rf_model.pkl")
joblib.dump(threshold, "threshold.pkl")