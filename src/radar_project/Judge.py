import numpy as np
import joblib

# ===== 加载模型（程序启动时执行一次）=====
model = joblib.load(r"models/rf_model.pkl")
threshold = joblib.load(r"models/threshold.pkl")  # 比如 0.6

#print("模型加载成功")

def judge_channel(sig, fs):
    """
    输入:
        sig: 1D numpy数组（微位移）
        fs: 采样率（Hz），例如 250

    输出:
        is_good: True / False
        prob: 好数据概率
    """

    # ===== 防御（避免异常数据）=====
    if len(sig) < 1000:
        return False, 0.0

    # ===== 预处理 =====
    sig = sig - np.mean(sig)
    sig = sig / (np.std(sig) + 1e-8)

    # ===== FFT =====
    fft = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), 1/fs)

    # ===== 特征 =====
    peak_freq = freqs[np.argmax(fft)]

    mask = (freqs > 0.8) & (freqs < 2)
    energy_ratio = np.sum(fft[mask]) / (np.sum(fft) + 1e-8)

    std = np.std(sig)
    rms = np.sqrt(np.mean(sig**2))
    peak2peak = np.max(sig) - np.min(sig)

    sharpness = np.max(fft) / (np.mean(fft) + 1e-8)

    feat = np.array([peak_freq, energy_ratio, std, rms, peak2peak, sharpness]).reshape(1, -1)

    # ===== 模型预测 =====
    prob = model.predict_proba(feat)[0, 1]

    # ===== 判定 =====
    is_good = prob > threshold

    return is_good, prob