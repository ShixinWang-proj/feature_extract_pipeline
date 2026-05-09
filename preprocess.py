import sys
import os
import pandas as pd
import numpy as np

# ============================================================
# 分段参数
# ============================================================
GAP_THRESHOLD_SEC = 5.0       # 时间间隔超过此秒数才切段
MIN_SEGMENT_DURATION_SEC = 10.0  # 段时长短于此秒数则丢弃
MIN_SEGMENT_ROWS = 50         # 段行数少于此值则丢弃（兜底）
MIN_AVG_RATE_HZ = 50          # 平均采样率低于此值则丢弃（防稀疏段导致重采样为空）

def preprocess_and_split(input_path, out_dir):
    """加载原始CSV，预处理并拆分为连续片段，返回输出目录"""
    os.makedirs(out_dir, exist_ok=True)

    print(f"正在加载文件: {input_path} ...")
    df = pd.read_csv(input_path)

    # IED 取反，使生理收缩期变为正向波峰
    if "ied" in df.columns:
        df["ied"] = -df["ied"]

    # --- 时间解析 ---
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%d-%m-%Y %H:%M:%S")
    df.drop(["date", "time"], axis=1, inplace=True)

    # 去掉首尾各一行（边界脏数据）
    df = df.iloc[1:-1].reset_index(drop=True)

    # --- 衍生特征 ---
    df["motion"] = np.sqrt(df.accX ** 2 + df.accY ** 2 + df.accZ ** 2)
    df = df[["datetime", "duration", "red", "ied", "accX", "accY", "accZ", "motion"]]

    # --- 按时间连续性拆分片段 ---
    diff_seconds = df["datetime"].diff().dt.total_seconds()
    break_mask = diff_seconds > GAP_THRESHOLD_SEC
    segment_ids = break_mask.cumsum()

    total_groups = 0
    kept = 0
    for seg_id, group in df.groupby(segment_ids):
        total_groups += 1
        n_rows = len(group)
        duration = (group["datetime"].iloc[-1] - group["datetime"].iloc[0]).total_seconds()

        avg_rate = n_rows / duration if duration > 0 else 0
        if n_rows < MIN_SEGMENT_ROWS or duration < MIN_SEGMENT_DURATION_SEC or avg_rate < MIN_AVG_RATE_HZ:
            continue

        kept += 1
        seg_name = f"segment_{kept:03d}"
        seg_path = os.path.join(out_dir, f"{seg_name}.csv")
        group.to_csv(seg_path, index=False)
        t_start = group["datetime"].iloc[0].strftime("%H:%M:%S")
        t_end = group["datetime"].iloc[-1].strftime("%H:%M:%S")
        print(f"  {seg_name}: {n_rows} 行, {duration:.1f}s, ~{avg_rate:.0f}Hz, {t_start} → {t_end}")

    skipped = total_groups - kept
    if skipped:
        print(f"  ⚠ 丢弃 {skipped} 个不合格片段（阈值: ≥{MIN_SEGMENT_DURATION_SEC:.0f}s, ≥{MIN_SEGMENT_ROWS}行, ≥{MIN_AVG_RATE_HZ}Hz）")
    print(f"\n拆分完成: 共 {kept} 个有效片段 -> {out_dir}/")
    return out_dir


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python preprocess.py <input_csv> [output_dir]")
        sys.exit(1)

    input_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "segments"
    preprocess_and_split(input_path, out_dir)