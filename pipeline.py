import sys
import os
import glob
import numpy as np
import pandas as pd
import subprocess

def find_jump_absolute_diff(data):
    """
    使用绝对一阶差分寻找跳跃点 (寻找绝对数值差距最大的一步)
    """
    sorted_data = np.sort(data)
    diffs = np.diff(sorted_data)
    jump_index = np.argmax(diffs)
    threshold = sorted_data[jump_index]
    next_value = sorted_data[jump_index + 1]
    max_diff = diffs[jump_index]
    return threshold, next_value, max_diff, jump_index

def filter_outliers_by_jump(df, col):
    """
    对指定列用跳变点找阈值，剔除上方异常值
    """
    clean_data = df[col].dropna().values
    if len(clean_data) < 10:
        return df, 0
    threshold, next_val, max_diff, idx = find_jump_absolute_diff(clean_data)
    outlier_count = int((clean_data > threshold).sum())
    filtered = df[df[col] <= threshold].copy()
    return filtered, outlier_count

def run(script, *args, check=True):
    """执行一个子脚本"""
    cmd = [sys.executable, script, *args]
    print(f"  ▶ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)

def main():
    if len(sys.argv) < 2:
        print("用法: python pipeline.py <input_csv> [output_dir]")
        print("示例: python pipeline.py raw.csv ./output/")
        sys.exit(1)

    input_csv = sys.argv[1]
    base_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    os.makedirs(base_dir, exist_ok=True)

    # ========== Step 0: 预处理 & 片段拆分 ==========
    seg_dir = os.path.join(base_dir, "segments")
    print("\n=== [0/5] 预处理 & 片段拆分 ===")
    run("preprocess.py", input_csv, seg_dir)

    segments = sorted(glob.glob(os.path.join(seg_dir, "segment_*.csv")))
    if not segments:
        print("【错误】未找到任何连续片段。")
        sys.exit(1)
    print(f"共 {len(segments)} 个片段\n")

    # ========== 逐片段处理 ==========
    all_features = []
    all_rri = []

    for i, seg_path in enumerate(segments):
        seg_name = os.path.splitext(os.path.basename(seg_path))[0]
        prefix = os.path.join(base_dir, seg_name)
        print(f"\n{'='*50}")
        print(f"=== 片段 {i+1}/{len(segments)}: {seg_name} ===")
        print(f"{'='*50}")

        # Step 1: 100Hz 重采样
        print("\n[1/5] 100Hz 重采样")
        resampled = f"{prefix}_100hz.parquet"
        run("resample.py", seg_path, resampled)

        # Step 2: 滤波清洗
        print("\n[2/5] 滤波清洗")
        cleaned = f"{prefix}_clean.parquet"
        run("highpass.py", resampled, "100", cleaned)

        # Step 3: 峰谷检测 + HRV 质控
        print("\n[3/5] 峰谷检测 + HRV 质控")
        peaks = f"{prefix}_peaks.parquet"
        run("extract_peaks.py", cleaned, peaks, "100")

        # Step 4: 逐拍特征提取
        print("\n[4/5] 逐拍特征提取")
        features_path = f"{prefix}_features.parquet"
        run("extract_features.py", peaks, features_path)

        # Step 5: RRI 时间序列
        print("\n[5/5] RRI 时间序列")
        rri_path = f"{prefix}_rri.parquet"
        run("features.py", peaks, rri_path)

        # 收集结果
        if os.path.exists(features_path):
            all_features.append(pd.read_parquet(features_path))
        if os.path.exists(rri_path):
            all_rri.append(pd.read_parquet(rri_path))

    # ========== 拼接 & 输出 ==========
    print(f"\n{'='*50}")
    print("=== 拼接所有片段特征 ===")
    print(f"{'='*50}")

    if all_features:
        feat_df = pd.concat(all_features, ignore_index=True)
        feat_df["timestamp"] = pd.to_datetime(feat_df["timestamp"])
        feat_df = feat_df.sort_values("timestamp").reset_index(drop=True)

        # 异常值过滤 (asc_area / desc_area)
        print("\n--- 异常值过滤 (asc_area / desc_area) ---")
        feat_df, n_out_asc = filter_outliers_by_jump(feat_df, "asc_area")
        max_val = feat_df["asc_area"].max() if n_out_asc > 0 else None
        print(f"  asc_area: 剔除 {n_out_asc} 条 (阈值 {f'{max_val:.0f}' if max_val is not None else 'N/A'})")
        
        feat_df, n_out_desc = filter_outliers_by_jump(feat_df, "desc_area")
        max_val_d = feat_df["desc_area"].max() if n_out_desc > 0 else None
        print(f"  desc_area: 剔除 {n_out_desc} 条 (阈值 {f'{max_val_d:.0f}' if max_val_d is not None else 'N/A'})")
        
        print(f"  过滤后剩余: {len(feat_df)} / {len(feat_df) + n_out_asc + n_out_desc} 条")

        feat_out = os.path.join(base_dir, "all_features.parquet")
        feat_df.to_parquet(feat_out, index=False)
        print(f"\nall_features: {len(feat_df)} 行 -> {feat_out}")
        print(feat_df.head())

    if all_rri:
        rri_df = pd.concat(all_rri, ignore_index=True)
        rri_df["timestamp"] = pd.to_datetime(rri_df["timestamp"])
        rri_df = rri_df.sort_values("timestamp").reset_index(drop=True)
        rri_out = os.path.join(base_dir, "all_rri.parquet")
        rri_df.to_parquet(rri_out, index=False)
        print(f"\nall_rri: {len(rri_df)} 行 -> {rri_out}")
        print(rri_df.head())

    print(f"\n{'='*50}")
    print("Pipeline 完成！")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()