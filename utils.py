import pandas as pd
import random

def resample_ppg_100hz(piece: pd.DataFrame) -> pd.DataFrame:
    """
    将非均匀的秒级 PPG/IMU 数据重采样为严格的 100Hz
    """
    df = piece.copy()
    if 'duration' in df.columns:
        df = df.drop(columns=['duration'])

    df['sec_time'] = df.index

    # 🌟 修复：剔除帧数不足的“不完整秒” (防拉伸)
    sec_counts = df['sec_time'].value_counts()
    valid_secs = sec_counts[sec_counts >= 80].index
    df = df[df['sec_time'].isin(valid_secs)].copy()
    
    if df.empty:
        return pd.DataFrame()

    # 微观时间戳分配
    grouped = df.groupby('sec_time')
    df['sub_sec_offset'] = grouped.cumcount() / grouped['sec_time'].transform('count')

    df['exact_time'] = df['sec_time'] + pd.to_timedelta(df['sub_sec_offset'], unit='s')
    df = df.set_index('exact_time')
    df = df.drop(columns=['sec_time', 'sub_sec_offset'])

    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]

    start_time = df.index.min().floor('s')
    end_time = df.index.max().ceil('s')
    target_index = pd.date_range(start=start_time, end=end_time, freq='10ms', name='datetime')

    df_combined = df.reindex(df.index.union(target_index).unique()).sort_index()
    df_interpolated = df_combined.interpolate(method='pchip')
    df_100hz = df_interpolated.reindex(target_index).dropna()

    return df_100hz

