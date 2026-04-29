import sys
import os
import pandas as pd
import numpy as np
from utils import resample_ppg_100hz

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python resample.py <input_csv> [output_parquet]")
        sys.exit(1)

    input_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None

    if out_path is None:
        base = os.path.splitext(input_path)[0]
        out_path = f"{base}_100hz.parquet"

    print(f"正在加载文件: {input_path} ...")
    df = pd.read_csv(input_path)

    if "datetime" not in df.columns:
        print("【错误】找不到 'datetime' 列，请先运行 preprocess.py。")
        sys.exit(1)

    # 以 datetime 为索引进行 100Hz 重采样
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()

    # 去重（同秒内重复时间戳）
    df = df[~df.index.duplicated(keep="first")]

    print(f"原始: {len(df)} 行, {df.index[0]} → {df.index[-1]}")

    df_100hz = resample_ppg_100hz(df)

    print(f"重采样后: {len(df_100hz)} 行, 采样率: 100Hz")
    df_100hz.to_parquet(out_path)
    print(f"已保存至: {out_path}")
