import sys
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python extract_peaks.py <processed_parquet_path> [output_path] [fs]")
        print("示例: python extract_peaks.py data_processed.parquet output.parquet")
        sys.exit(1)

    input_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "output.parquet"
    fs = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    print(f"正在加载文件: {input_path} ...")
    df = pd.read_parquet(input_path).reset_index()

    if "ied_clean" not in df.columns:
        print("【错误】输入文件中找不到 'ied_clean' 列，请先运行滤波脚本。")
        sys.exit(1)

    signal = df["ied_clean"].to_numpy(dtype=np.float64)
    total = len(signal)
    print(f"✅ 已加载 {total} 个滤波后数据点 (采样率: {fs}Hz)")

    # 初始化所有标记列
    df["is_peak"] = False
    df["peak_value"] = np.nan
    df["is_valley"] = False
    df["valley_value"] = np.nan
    df["rr_interval_ms"] = np.nan
    df["vv_interval_ms"] = np.nan
    df["branch_phase"] = "unknown"  # 记录上升支(ascending)或下降支(descending)

    # ====================================================
    # 步骤 1: 带约束条件的自适应波峰检测
    # ====================================================
    print("\n正在执行自适应波峰提取...")
    
    # 限制最高心率 240 BPM
    min_distance = int(0.25 * fs)

    # 基于 MAD 动态分布计算突起度，免疫局部极端伪影
    median_val = np.median(signal)
    mad = np.median(np.abs(signal - median_val))
    estimated_std = mad * 1.4826
    min_prominence = estimated_std * 0.6 

    peaks, properties = find_peaks(
        signal,
        distance=min_distance,
        prominence=min_prominence,
    )
    print(f"✅ 波峰检测完成！共提取到 {len(peaks)} 个原始波峰")

    # ====================================================
    # 步骤 2: 原始 RR 间期计算与医疗级 HRV 质控
    # ====================================================
    # 临时写入所有原始波峰（后面被淘汰的会被抹除）
    df.loc[peaks, "is_peak"] = True
    df.loc[peaks, "peak_value"] = signal[peaks]

    rr_intervals = np.diff(peaks) / fs * 1000

    # 默认全部波峰是 valid，稍后通过质控掩码(nn_mask)剔除
    valid_peaks_mask = np.ones(len(peaks), dtype=bool)

    if len(rr_intervals) > 0:
        # 规则 A: 绝对生理极限过滤 (300ms ~ 2000ms)
        abs_mask = (rr_intervals >= 300) & (rr_intervals <= 2000)
        
        # 规则 B: 相对突变过滤 (Quotient Filter) -> 变化率 <= 20%
        diff_ratio = np.zeros_like(rr_intervals)
        diff_ratio[1:] = np.abs(np.diff(rr_intervals)) / rr_intervals[:-1]
        diff_ratio[0] = 0  
        
        rel_mask = diff_ratio <= 0.20
        
        # 最终合格的 NN (Normal-to-Normal) 掩码
        nn_mask = abs_mask & rel_mask
        nn_intervals = rr_intervals[nn_mask]
        
        # 统计剔除情况
        total_rr = len(rr_intervals)
        failed_abs = total_rr - np.sum(abs_mask)
        failed_rel = np.sum(abs_mask) - np.sum(nn_mask)
        drop_rate = ((total_rr - len(nn_intervals)) / total_rr) * 100 if total_rr > 0 else 0
        
        print("\n🛡️ HRV 级质控 (QC) 报告:")
        print(f"  - 原始 RR 间期总数: {total_rr}")
        print(f"  - 剔除: 绝对生理异常 (>2s 或 <0.3s): {failed_abs} 个")
        print(f"  - 剔除: 相对突变异常 (跳变 > 20%): {failed_rel} 个")
        print(f"  - 最终保留 NN 间期: {len(nn_intervals)} 个 (总剔除率 {drop_rate:.2f}%)")

        # 将质控失败的波峰从有效列表中剔除
        valid_peaks_mask[1:] = nn_mask
        invalid_peaks_indices = peaks[~valid_peaks_mask]
        
        # 从 DataFrame 中抹除异常波峰的标记
        df.loc[invalid_peaks_indices, "is_peak"] = False
        df.loc[invalid_peaks_indices, "peak_value"] = np.nan

        # 只记录合规的 RR/NN 间期
        valid_peaks = peaks[valid_peaks_mask]
        if len(valid_peaks) > 1:
            df.loc[valid_peaks[1:], "rr_interval_ms"] = nn_intervals

        if len(nn_intervals) > 0:
            mean_nn = np.mean(nn_intervals)
            heart_rate = 60000 / mean_nn  
            sdnn = np.std(nn_intervals)  
            print("\n📊 核心生理特征统计 (基于极净 NN 间期):")
            print(f"  - 平均 NN 间期 : {mean_nn:.1f} ms")
            print(f"  - 平均心率 (HR): {heart_rate:.1f} BPM")
            print(f"  - SDNN (HRV)   : {sdnn:.1f} ms")
    else:
        print("⚠️ 警告：提取到的波峰过少，无法进行质控。")
        valid_peaks = peaks

    # ====================================================
    # 步骤 3: 严格对齐的波谷检测与升降支 (Branch) 划分
    # ====================================================
    print("\n正在划分上升支与下降支，并提取交替波谷...")
    
    valid_valleys = []
    
    if len(valid_peaks) > 1:
        # 遍历每一对有效的、相邻的波峰
        for i in range(len(valid_peaks) - 1):
            p_start = valid_peaks[i]
            p_end = valid_peaks[i+1]
            
            # 在两个波峰之间寻找实际的最低点作为波谷
            segment = signal[p_start:p_end]
            v_idx = p_start + np.argmin(segment)
            valid_valleys.append(v_idx)
            
            # 标记下降支：从当前波峰到区间波谷
            df.loc[p_start:v_idx, "branch_phase"] = "descending"
            # 标记上升支：从区间波谷到下一个波峰
            df.loc[v_idx:p_end, "branch_phase"] = "ascending"

        # 记录波谷点到 DataFrame
        df.loc[valid_valleys, "is_valley"] = True
        df.loc[valid_valleys, "valley_value"] = signal[valid_valleys]

        # 计算并记录 VV 间期 (谷到谷)
        vv_intervals = np.diff(valid_valleys) / fs * 1000
        if len(valid_valleys) > 1:
            df.loc[valid_valleys[1:], "vv_interval_ms"] = vv_intervals

        print(f"✅ 成功提取有效波谷 {len(valid_valleys)} 个")
        print("✅ 升降支(ascending/descending)标记完成！")
    else:
        print("⚠️ 警告：有效波峰不足两个，无法划分升降支和波谷。")

    # ====================================================
    # 步骤 4: 结果落盘保存
    # ====================================================
    print(f"\n💾 正在写入本地文件...")
    df.to_parquet(out_path)
    print(f"🎉 任务完成！带有干净形态特征的数据已保存至: {out_path}")
    