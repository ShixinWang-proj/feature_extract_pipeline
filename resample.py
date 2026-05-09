import sys
import os
import pandas as pd
from utils import resample_ppg_100hz


def resample_to_100hz(input_csv, output_parquet):
    """将 segment CSV 重采样为 100Hz parquet，返回 True 表示成功"""
    df = pd.read_csv(input_csv)

    if "datetime" not in df.columns:
        print("【错误】找不到 'datetime' 列，请先运行 preprocess.py。")
        return False

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()

    print(f"原始: {len(df)} 行, {df.index[0]} → {df.index[-1]}")

    df_100hz = resample_ppg_100hz(df)

    if df_100hz.empty:
        print("警告：重采样后数据为空。")
        return False

    print(f"重采样后: {len(df_100hz)} 行, 采样率: 100Hz")
    df_100hz.to_parquet(output_parquet)
    print(f"已保存至: {output_parquet}")
    return True


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else None
    out_path = sys.argv[2] if len(sys.argv) > 2 else None

    if input_path is None:
        print("用法: python resample.py <input_csv> [output_parquet]")
        sys.exit(1)

    if out_path is None:
        base = os.path.splitext(input_path)[0]
        out_path = f"{base}_100hz.parquet"

    print(f"正在加载文件: {input_path} ...")
    ok = resample_to_100hz(input_path, out_path)
    sys.exit(0 if ok else 1)