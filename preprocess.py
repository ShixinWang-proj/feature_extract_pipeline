import sys
import os
import pandas as pd
import numpy as np

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python preprocess.py <input_csv> [output_dir]")
        print("示例: python preprocess.py raw.csv ./segments/")
        sys.exit(1)

    input_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "segments"
    os.makedirs(out_dir, exist_ok=True)

    print(f"正在加载文件: {input_path} ...")
    df = pd.read_csv(input_path)

    # --- 时间解析 ---
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%d-%m-%Y %H:%M:%S")
    df.drop(["date", "time"], axis=1, inplace=True)

    # 去掉首尾各一行（边界脏数据）
    df = df.iloc[1:-1].reset_index(drop=True)

    # --- 衍生特征 ---
    df["motion"] = np.sqrt(df.accX ** 2 + df.accY ** 2 + df.accZ ** 2)
    df = df[["datetime", "duration", "red", "ied", "accX", "accY", "accZ", "motion"]]

    # --- 按时间连续性拆分片段 ---
    # 正常情况：同秒内 diff=0，跨秒 diff=1，都算连续
    # 只有 diff > 1 才说明有数据丢失，需要拆分
    diff_seconds = df["datetime"].diff().dt.total_seconds()
    break_mask = diff_seconds > 1.0
    segment_ids = break_mask.cumsum()

    for seg_id, group in df.groupby(segment_ids):
        if len(group) < 2:
            continue
        seg_name = f"segment_{seg_id:03d}"
        seg_path = os.path.join(out_dir, f"{seg_name}.csv")
        group.to_csv(seg_path, index=False)
        t_start = group["datetime"].iloc[0].strftime("%H:%M:%S")
        t_end = group["datetime"].iloc[-1].strftime("%H:%M:%S")
        print(f"  {seg_name}: {len(group)} 行, {t_start} → {t_end}")

    print(f"\n拆分完成: 共 {segment_ids.nunique()} 个片段 -> {out_dir}/")
