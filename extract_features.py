import sys
import pandas as pd
import numpy as np

if __name__ == "__main__":
    input_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "features.parquet"

    df = pd.read_parquet(input_path).reset_index()

    # 信号无需取反（已经在预处理层统一处理为了真实的生理正向波形）
    signal = df["ied_clean"].to_numpy(dtype=np.float64)

    peaks_idx = df[df["is_peak"] == True].index.values
    valleys_idx = df[df["is_valley"] == True].index.values

    results = []

    # 🌟 核心逻辑更新：按照 "波谷 -> 波峰 -> 波谷" 的周期遍历
    for i in range(len(valleys_idx) - 1):
        v_start = valleys_idx[i]
        v_end = valleys_idx[i + 1]

        # 寻找被夹在这两个波谷之间的真实波峰
        p_between = peaks_idx[(peaks_idx > v_start) & (peaks_idx < v_end)]
        if len(p_between) == 0:
            continue
        p = p_between[0]

        # --- 时间戳 ---
        # 使用当前周期核心波峰的时间作为特征的时间戳
        ts = df.iloc[p]["datetime"]
        timestamp = pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

        # --- RRI (ms) ---
        # RR间期直接读取在 extract_peaks.py 中质控过的数据
        rri = df.iloc[p]["rr_interval_ms"]
        if np.isnan(rri):
            # 兜底：如果波峰未计算 RRI，用当前的 VV 间期估算
            vvi = df.iloc[v_end]["vv_interval_ms"]
            rri = vvi if not np.isnan(vvi) else (v_end - v_start) / 100 * 1000

        # --- 局部动态基线拉平与面积计算 ---
        segment_len = v_end - v_start + 1
        raw_segment = signal[v_start : v_end + 1]

        # 1. 计算斜率，构造一条连接左右波谷的倾斜基线
        val_start = signal[v_start]
        val_end = signal[v_end]
        baseline = np.linspace(val_start, val_end, segment_len)

        # 2. 变换拉平：原信号减去基线。变换后，左右两端的波谷值严丝合缝变为 0
        flattened_segment = raw_segment - baseline

        # 波峰在当前切片中的相对索引
        p_rel = p - v_start

        # 3. 面积计算 (因为基线是 0，且波峰向上突起，此时用梯形积分出来的面积必定为绝对正值)
        # 上升支面积 (波谷 v_start -> 波峰 p)
        asc_segment = flattened_segment[: p_rel + 1]
        asc_area = float(np.trapezoid(asc_segment))

        # 下降支面积 (波峰 p -> 波谷 v_end)
        desc_segment = flattened_segment[p_rel :]
        desc_area = float(np.trapezoid(desc_segment))

        # --- 运动伪影 Motion ---
        ax = df.iloc[v_start:v_end + 1]["accX"].to_numpy(dtype=np.float64)
        ay = df.iloc[v_start:v_end + 1]["accY"].to_numpy(dtype=np.float64)
        az = df.iloc[v_start:v_end + 1]["accZ"].to_numpy(dtype=np.float64)
        vmu = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
        motion = float(np.mean(vmu))

        results.append({
            "timestamp": timestamp,
            "rri_ms": float(rri),
            "asc_area": asc_area,
            "desc_area": desc_area,
            "motion": motion,
        })

    out_df = pd.DataFrame(results)
    out_df.to_parquet(out_path, index=False)

    print(f"提取完成: {len(out_df)} 个心跳周期 -> {out_path}")
    
    # 打印简要数据验证
    if not out_df.empty:
        print(f"\nRRI (ms):     均值={out_df['rri_ms'].mean():.1f}, 标准差={out_df['rri_ms'].std():.1f}, 范围=[{out_df['rri_ms'].min():.0f}, {out_df['rri_ms'].max():.0f}]")
        print(f"上升面积:     均值={out_df['asc_area'].mean():.1f}, 标准差={out_df['asc_area'].std():.1f}")
        print(f"下降面积:     均值={out_df['desc_area'].mean():.1f}, 标准差={out_df['desc_area'].std():.1f}")
        print(f"Motion (VMU): 均值={out_df['motion'].mean():.1f}, 标准差={out_df['motion'].std():.1f}")