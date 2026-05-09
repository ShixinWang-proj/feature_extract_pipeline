```python pipeline.py <input_csv> [output_dir]```

e.g.

```python pipeline.py raw.csv ./output/```

```python feature_da.py -s <source_csv> -t <target_dir> [-o <output_csv>]```

e.g.

```python feature_da.py -s ./output/all_features.parquet -t ./target_distribution_folder/ -o ./output/aligned_features.csv```