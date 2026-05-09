import os
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def _read_auto(path, **kwargs):
    """自动识别文件类型读取数据（parquet / CSV 带编码兜底）"""
    path = Path(path)
    if path.suffix in ('.parquet', '.pq'):
        return pd.read_parquet(path, **kwargs)

    # CSV 路径：依次尝试编码 + 引擎兜底
    kwargs.setdefault('on_bad_lines', 'warn')
    for enc in ['utf-8', 'gbk', 'latin-1']:
        for engine in ['c', 'python']:
            try:
                return pd.read_csv(path, encoding=enc, engine=engine, **kwargs)
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
    raise ValueError(f"无法识别文件格式或编码: {path}")

# ==========================================
# 阶段 A: 获取 Target 分布
# ==========================================
_TARGET_FEATURES = ["area_up", "area_down", "motion"]


def extract_target_features(file_path):
    try:
        df = _read_auto(file_path)
        if df.empty:
            return None
        result = {}
        for feat in _TARGET_FEATURES:
            if feat in df.columns:
                raw = df[feat].dropna().values
                result[feat] = raw[::100]
            else:
                result[feat] = np.array([])
        return result
    except Exception:
        return None

def build_target_distributions(target_dir, use_cache=True):
    target_base = Path(target_dir)
    cache_path = target_base / "_target_dist_cache.pkl"

    # --- 尝试从缓存加载 ---
    if use_cache and cache_path.exists():
        cache_mtime = cache_path.stat().st_mtime
        csv_files = list(target_base.rglob("*.csv"))
        # 检查是否有 CSV 文件比缓存新（说明数据有更新）
        newer = [f for f in csv_files if f.stat().st_mtime > cache_mtime]
        if not newer:
            print(f"📦 从缓存加载 Target 分布: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        else:
            print(f"🔄 检测到 {len(newer)} 个文件已更新，重新构建分布...")

    # --- 从原始文件构建 ---
    csv_files = list(target_base.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"在 {target_dir} 中未找到 Target CSV 文件。")

    print(f"📊 正在从 {len(csv_files)} 个文件中构建 Target 分布...")
    global_data = {f: [] for f in _TARGET_FEATURES}
    workers = max(1, os.cpu_count() - 1)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(extract_target_features, fp): fp for fp in csv_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="提取 Target", colour="green"):
            res = future.result()
            if res:
                for feat in _TARGET_FEATURES:
                    if len(res.get(feat, [])) > 0:
                        global_data[feat].append(res[feat])

    target_dists = {}
    for feat in _TARGET_FEATURES:
        if global_data[feat]:
            merged_data = np.concatenate(global_data[feat])
            p01 = np.percentile(merged_data, 1)
            p99 = np.percentile(merged_data, 99)
            if p01 != p99:
                merged_data = merged_data[(merged_data >= p01) & (merged_data <= p99)]
            target_dists[feat] = merged_data
        else:
            target_dists[feat] = np.array([])

    # --- 保存缓存 ---
    with open(cache_path, "wb") as f:
        pickle.dump(target_dists, f)
    print(f"💾 分布已缓存至: {cache_path}")

    return target_dists

# ==========================================
# 阶段 B: Wasserstein 域转换核心算法
# ==========================================
def wasserstein_1d_mapping(source_data, target_data):
    source_clean = source_data[~np.isnan(source_data)]
    target_clean = np.sort(target_data[~np.isnan(target_data)])
    
    if len(source_clean) == 0 or len(target_clean) == 0:
        return source_data 
        
    source_sorted_indices = np.argsort(source_clean)
    
    x_source = np.linspace(0, 1, len(source_clean))
    x_target = np.linspace(0, 1, len(target_clean))
    
    mapped_sorted = np.interp(x_source, x_target, target_clean)
    
    mapped_source = np.empty_like(source_clean)
    mapped_source[source_sorted_indices] = mapped_sorted
    
    result = np.full_like(source_data, np.nan, dtype=np.float64)
    result[~np.isnan(source_data)] = mapped_source
    
    return result

# ==========================================
# 阶段 C: 内存级 DataFrame 转换 API (支持列名映射)
# ==========================================
def apply_domain_adaptation(df: pd.DataFrame, target_dists: dict, feature_mapping: dict = None) -> pd.DataFrame:
    """
    直接接收内存中的 DataFrame 进行映射，支持 Source 和 Target 列名不一致的情况。
    """
    if feature_mapping is None:
        feature_mapping = {"area_up": "area_up", "area_down": "area_down"}
        
    df_mapped = df.copy()
    
    for src_col, tgt_col in feature_mapping.items():
        if src_col in df_mapped.columns and len(target_dists.get(tgt_col, [])) > 0:
            source_array = df_mapped[src_col].values
            mapped_array = wasserstein_1d_mapping(source_array, target_dists[tgt_col])
            df_mapped[src_col] = mapped_array
        elif src_col not in df_mapped.columns:
            print(f"⚠️ 跳过映射: 源 DataFrame 中不存在 '{src_col}' 列。")
        elif len(target_dists.get(tgt_col, [])) == 0:
            print(f"⚠️ 跳过映射: Target 字典中不存在 '{tgt_col}' 的分布。")
            
    return df_mapped

# ==========================================
# 🚀 启动台 (命令行 CLI 版本)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="领域适应转换引擎 (支持异构列名映射)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-s", "--source", required=True, type=str, help="[必填] 源 CSV 文件路径")
    parser.add_argument("-t", "--target_dir", required=True, type=str, help="[必填] Target 域分布文件夹路径")
    parser.add_argument("-o", "--output", required=False, type=str, help="[选填] 输出的 CSV 路径。")
    parser.add_argument("--no-cache", action="store_true", help="强制重新构建 Target 分布，忽略缓存。")

    args = parser.parse_args()

    source_path = Path(args.source)
    target_dir = Path(args.target_dir)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = source_path.with_name(f"{source_path.stem}_aligned{source_path.suffix}")

    if not source_path.exists():
        print(f"❌ 错误: 找不到源文件 {source_path}")
        exit(1)

    print("="*50)
    print(f"🔧 Target 域文件夹: {target_dir}")
    print(f"📥 输入的源数据: {source_path}")
    print(f"📤 计划输出路径: {output_path}")
    print("="*50)

    # 1. 构建 Target 分布
    target_distributions = build_target_distributions(target_dir, use_cache=not args.no_cache)

    # 2. 读取源 CSV
    print("\n⏳ 正在加载源数据...")
    input_df = _read_auto(source_path)

    # 3. 定义源列名到目标列名的映射关系
    my_column_mapping = {
        "asc_area": "area_up",
        "desc_area": "area_down",
        "motion": "motion",
    }

    # 4. 执行转换
    print("⏳ 正在执行跨列名 Wasserstein 域转换...")
    aligned_df = apply_domain_adaptation(
        df=input_df, 
        target_dists=target_distributions,
        feature_mapping=my_column_mapping
    )

    # 5. 保存结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned_df.to_csv(output_path, index=False)
    print(f"\n✅ 转换圆满完成！数据已保存至:\n{output_path}")