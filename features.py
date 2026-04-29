import sys
import pandas as pd

if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "peaks.parquet"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "rri.parquet"

    df = pd.read_parquet(input_path).reset_index()
    peaks = df[df["is_peak"]].copy()
    peaks["timestamp"] = pd.to_datetime(peaks["datetime"]).dt.floor("s")

    rri = peaks[["timestamp", "rr_interval_ms"]].copy()
    rri = rri.rename(columns={"rr_interval_ms": "rri_ms"})

    rri.to_parquet(out_path, index=False)
    print(f"Saved {len(rri)} rows to {out_path}")
    print(rri.head())
