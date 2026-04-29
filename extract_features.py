import sys
import pandas as pd
import numpy as np

if __name__ == "__main__":
    input_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "features.parquet"

    df = pd.read_parquet(input_path).reset_index()

    # IED 信号取反
    signal = -df["ied_clean"].to_numpy(dtype=np.float64)

    peaks_idx = df[df["is_peak"] == True].index.values
    valleys_idx = df[df["is_valley"] == True].index.values

    results = []

    for i in range(len(peaks_idx) - 1):
        p_start = peaks_idx[i]
        p_end = peaks_idx[i + 1]

        # 找 p_start 和 p_end 之间的 valley（取反后的波峰）
        v_between = valleys_idx[(valleys_idx > p_start) & (valleys_idx < p_end)]
        if len(v_between) == 0:
            continue
        v = v_between[0]

        # --- 时间戳 ---
        ts = df.iloc[p_start]["datetime"]
        timestamp = pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

        # --- RRI (ms) ---
        rri = df.iloc[p_end]["rr_interval_ms"]
        if np.isnan(rri):
            rri = (p_end - p_start) / 100 * 1000  # 用间距估算

        # --- 上升面积 (peak谷 → valley峰) ---
        # 取反后 peak 处是谷，valley 处是峰
        # 上升段: p_start 到 v，面积 = 积分 (signal - 基线)
        baseline_asc = signal[p_start]
        asc_segment = signal[p_start : v + 1] - baseline_asc
        asc_area = float(np.trapezoid(asc_segment))

        # --- 下降面积 (valley峰 → next peak谷) ---
        baseline_desc = signal[p_end]
        desc_segment = signal[v : p_end + 1] - baseline_desc
        desc_area = float(np.trapezoid(desc_segment))

        # --- Motion ---
        # 三轴加速度向量幅度 (VMU)
        ax = df.iloc[p_start:p_end + 1]["accX"].to_numpy(dtype=np.float64)
        ay = df.iloc[p_start:p_end + 1]["accY"].to_numpy(dtype=np.float64)
        az = df.iloc[p_start:p_end + 1]["accZ"].to_numpy(dtype=np.float64)
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
    print(f"\nRRI (ms):     均值={out_df['rri_ms'].mean():.1f}, 标准差={out_df['rri_ms'].std():.1f}, 范围=[{out_df['rri_ms'].min():.0f}, {out_df['rri_ms'].max():.0f}]")
    print(f"上升面积:     均值={out_df['asc_area'].mean():.1f}, 标准差={out_df['asc_area'].std():.1f}")
    print(f"下降面积:     均值={out_df['desc_area'].mean():.1f}, 标准差={out_df['desc_area'].std():.1f}")
    print(f"Motion (VMU): 均值={out_df['motion'].mean():.1f}, 标准差={out_df['motion'].std():.1f}")
