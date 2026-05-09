import sys
import pandas as pd
import numpy as np


def extract_beat_features(input_parquet, output_parquet):
    """从波峰波谷数据中提取逐拍特征（RRI、面积、motion）"""
    df = pd.read_parquet(input_parquet).reset_index()

    signal = df["ied_clean"].to_numpy(dtype=np.float64)

    peaks_idx = df[df["is_peak"] == True].index.values
    valleys_idx = df[df["is_valley"] == True].index.values

    results = []

    for i in range(len(valleys_idx) - 1):
        v_start = valleys_idx[i]
        v_end = valleys_idx[i + 1]

        p_between = peaks_idx[(peaks_idx > v_start) & (peaks_idx < v_end)]
        if len(p_between) == 0:
            continue
        p = p_between[0]

        # --- 时间戳 ---
        ts = df.iloc[p]["datetime"]
        timestamp = pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

        # --- RRI (ms) ---
        rri = df.iloc[p]["rr_interval_ms"]
        if np.isnan(rri):
            vvi = df.iloc[v_end]["vv_interval_ms"]
            rri = vvi if not np.isnan(vvi) else (v_end - v_start) / 100 * 1000

        # --- 局部动态基线拉平与面积计算 ---
        segment_len = v_end - v_start + 1
        raw_segment = signal[v_start : v_end + 1]

        val_start = signal[v_start]
        val_end = signal[v_end]
        baseline = np.linspace(val_start, val_end, segment_len)

        flattened_segment = raw_segment - baseline

        p_rel = p - v_start

        asc_segment = flattened_segment[: p_rel + 1]
        asc_area = float(np.trapezoid(asc_segment))

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
    out_df.to_parquet(output_parquet, index=False)

    print(f"提取完成: {len(out_df)} 个心跳周期 -> {output_parquet}")

    if not out_df.empty:
        print(f"\nRRI (ms):     均值={out_df['rri_ms'].mean():.1f}, 标准差={out_df['rri_ms'].std():.1f}, 范围=[{out_df['rri_ms'].min():.0f}, {out_df['rri_ms'].max():.0f}]")
        print(f"上升面积:     均值={out_df['asc_area'].mean():.1f}, 标准差={out_df['asc_area'].std():.1f}")
        print(f"下降面积:     均值={out_df['desc_area'].mean():.1f}, 标准差={out_df['desc_area'].std():.1f}")
        print(f"Motion (VMU): 均值={out_df['motion'].mean():.1f}, 标准差={out_df['motion'].std():.1f}")

    return out_df


if __name__ == "__main__":
    input_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "features.parquet"

    extract_beat_features(input_path, out_path)
