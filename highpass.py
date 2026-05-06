import sys
import os
import time
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.ndimage import median_filter

def fast_hampel_filter(data, window_size=101, n_sigmas=3):
    """
    极速版 Hampel 滤波器：专为百万级大数组设计，内存零拷贝/极少拷贝
    :param data: 1D numpy array (必须已处理完 NaN)
    :param window_size: 滑动窗口大小 (100Hz 下 101 为 1 秒窗口，需为奇数)
    :param n_sigmas: 判定为异常值的 MAD 倍数
    :return: 剔除异常值后的 numpy array
    """
    # 1. 计算滑动中位数 (底层 C 优化，mode='nearest' 处理边缘)
    rolling_median = median_filter(data, size=window_size, mode='nearest')
    
    # 2. 计算绝对残差
    abs_diff = np.abs(data - rolling_median)
    
    # 3. 计算滑动绝对中位差 (MAD)
    rolling_mad = median_filter(abs_diff, size=window_size, mode='nearest')
    
    # 4. 动态阈值
    threshold = n_sigmas * rolling_mad * 1.4826
    
    # 5. 替换异常值
    outliers = abs_diff > threshold
    s_clean = data.copy()
    s_clean[outliers] = rolling_median[outliers]
    
    return s_clean

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    巴特沃斯带通滤波器 (零相移，原样保留心跳波峰位置)
    """
    nyq = 0.5 * fs  
    low = lowcut / nyq
    high = highcut / nyq
    
    b, a = butter(order, [low, high], btype='band')
    
    # filtfilt 前后向滤波，如果输入含有哪怕一个 NaN，输出将全部变成 NaN
    y = filtfilt(b, a, data)
    return y

def process_ppg_signal_large_scale(raw_ppg, fs=100):
    """
    长时数据不分段连续处理流水线
    """
    # 步骤 1: 极速去除极端运动伪影和漏光尖峰
    despiked_ppg = fast_hampel_filter(raw_ppg, window_size=fs+1, n_sigmas=3)
    
    # 步骤 2: 零相移带通滤波，彻底压平基线漂移并滤除高频电噪声
    filtered_ppg = butter_bandpass_filter(despiked_ppg, lowcut=0.5, highcut=8.0, fs=fs, order=4)
    
    return despiked_ppg, filtered_ppg

# ==========================================
# 终端执行主入口
# ==========================================
if __name__ == "__main__":
    # 1. 命令行参数解析
    if len(sys.argv) < 2:
        print("【错误】缺少文件路径参数。")
        print("用法: python highpass.py <csv_or_parquet_path> [fs] [output_path]")
        print("示例: python highpass.py raw_data.parquet output.parquet")
        print("      python highpass.py raw_data.parquet 100 output.parquet")
        sys.exit(1)

    file_path = sys.argv[1]
    fs = 100
    out_file_path = None

    for arg in sys.argv[2:]:
        if arg.isdigit():
            fs = int(arg)
        else:
            out_file_path = arg

    if not os.path.exists(file_path):
        print(f"【错误】找不到文件: {file_path}")
        sys.exit(1)

    # 2. 读取文件
    print(f"正在加载文件: {file_path} ...")
    if file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path).reset_index()
    else:
        df = pd.read_csv(file_path)

    # 3. 提取特征并进行防御性 NaN 处理 (极其重要)
    if "ied" not in df.columns:
        print("【错误】数据中找不到 'ied' 列，请检查表头名称。")
        sys.exit(1)

    # 线性插值修补丢包断点 -> 前后向填充边缘空值 -> 转换为 C 连续的 float64 数组
    raw_ppg = df["ied"].interpolate(method='linear').bfill().ffill().to_numpy(dtype=np.float64)
    total_points = len(raw_ppg)
    print(f"✅ 已加载 {total_points} 个 ied 数据点 (采样率: {fs}Hz)")

    # 4. 核心处理计时开始
    print("🚀 开始执行滤波流水线 (去极值 + 零相移带通)...")
    start_time = time.time()

    despiked, clean_ppg = process_ppg_signal_large_scale(raw_ppg, fs=fs)

    end_time = time.time()
    print(f"⏱️ 核心处理完成！总耗时: {end_time - start_time:.3f} 秒")

    # 5. 数据落盘保存
    print("💾 正在将处理结果写入本地文件...")
    # 追加为新列，不破坏原始数据
    df['ied_despiked'] = despiked
    df['ied_clean'] = clean_ppg

    if out_file_path is None:
        base_name, ext = os.path.splitext(file_path)
        out_file_path = f"{base_name}_processed{ext}"

    if out_file_path.endswith(".parquet"):
        df.to_parquet(out_file_path)
    else:
        df.to_csv(out_file_path, index=False)
        
    print(f"🎉 数据已成功保存至: {out_file_path}")