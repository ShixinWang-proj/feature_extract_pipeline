```python pipeline.py <input_csv> [output_dir]```

e.g.

```python pipeline.py raw.csv ./output/```

```python feature_da.py -s <source_csv> -t <target_dir> [-o <output_csv>] [--no-cache]```

首次运行（自动构建并缓存 Target 分布）：

```bash
python feature_da.py -s ./output/all_features.parquet -t ./target_distribution_folder/ -o ./output/aligned_features.csv
```

之后再次运行相同命令，直接命中缓存，跳过文件读取：

```bash
# 输出 "📦 从缓存加载 Target 分布" — 秒级加载
python feature_da.py -s ./output/all_features.parquet -t ./target_distribution_folder/ -o ./output/aligned_features.csv
```

Target 分布缓存为 `target_dir/_target_dist_cache.pkl`。若 target 文件夹内 CSV 有新增或修改，下次运行会自动重建缓存。如需强制重建，加 `--no-cache`：

```bash
python feature_da.py -s ./output/all_features.parquet -t ./target_distribution_folder/ --no-cache
```