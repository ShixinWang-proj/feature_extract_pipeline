import sys
import pandas as pd


def extract_rri_timeseries(input_parquet, output_parquet):
    """从 peaks 数据中提取 RRI 时间序列"""
    df = pd.read_parquet(input_parquet).reset_index()
    peaks = df[df["is_peak"]].copy()
    peaks["timestamp"] = pd.to_datetime(peaks["datetime"]).dt.floor("s")

    rri = peaks[["timestamp", "rr_interval_ms"]].copy()
    rri = rri.rename(columns={"rr_interval_ms": "rri_ms"})

    rri.to_parquet(output_parquet, index=False)
    print(f"Saved {len(rri)} rows to {output_parquet}")
    print(rri.head())

    return rri


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "peaks.parquet"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "rri.parquet"

    extract_rri_timeseries(input_path, out_path)
