import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

"""重采样，差值"""


def resample_ppg_100hz(piece: pd.DataFrame) -> pd.DataFrame:
    """
    将非均匀的秒级 PPG/IMU 数据重采样为严格的 100Hz
    """
    # 1. 保护原始数据，并剔除无法数值插值的字符串列（如 'duration'）
    df = piece.copy()
    if 'duration' in df.columns:
        df = df.drop(columns=['duration'])

    # 2. 微观时间戳分配 (Micro-timestamping)
    # 将当前的 index (秒) 提取为列，方便计算
    df['sec_time'] = df.index

    # 按秒分组，计算这一秒内的总帧数 (N) 和当前帧的序号 (k)
    # 给每一行计算亚秒级偏移: offset = k / N
    grouped = df.groupby('sec_time')
    df['sub_sec_offset'] = grouped.cumcount() / grouped['sec_time'].transform('count')

    # 生成精确到微秒的绝对时间戳
    df['exact_time'] = df['sec_time'] + pd.to_timedelta(df['sub_sec_offset'], unit='s')

    # 设置新的精确时间戳为索引，并清理辅助列
    df = df.set_index('exact_time')
    df = df.drop(columns=['sec_time', 'sub_sec_offset'])

    # 确保时间戳单调递增（处理极少数由于设备回环可能导致的乱序）
    df = df.sort_index()
    # 去除由于极端情况可能产生的完全重复的时间戳
    df = df[~df.index.duplicated(keep='first')]

    # 3. 构建严格的 100Hz 目标时间轴 (间隔 10ms)
    # 从数据的最开始一秒到最后一秒
    start_time = df.index.min().floor('s')
    end_time = df.index.max().ceil('s')
    target_index = pd.date_range(start=start_time, end=end_time, freq='10ms', name='datetime')

    # 4. 合并索引并进行插值
    # 将原始时间戳和目标时间戳合并，这会产生大量 NaN
    df_combined = df.reindex(df.index.union(target_index).unique()).sort_index()

    # 使用 PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) 插值
    # 它比普通 spline 更稳定，不会在低帧率(如 60fps)段产生过大的假波峰
    df_interpolated = df_combined.interpolate(method='pchip')

    # 5. 提取出目标 100Hz 的数据
    df_100hz = df_interpolated.reindex(target_index)

    # (可选) 掐头去尾：插值可能会在最首尾产生 NaN，可以直接 drop
    df_100hz = df_100hz.dropna()

    return df_100hz


def generate_hex_string():
    # 1. 2025年随机秒级 Unix 时间戳（16进制固定8位）
    ts_start = 1735689600  # 2025-01-01 00:00:00
    ts_end = 1767225599  # 2025-12-31 23:59:59
    timestamp = random.randint(ts_start, ts_end)
    hex_ts = f"{timestamp:08x}"  # 8位小写16进制

    # 2. N(60, 100) 非负整数 → 3位16进制补0
    mu, sigma = 60, 100
    while True:
        num2 = int(random.gauss(mu, sigma))
        if num2 >= 0:
            break
    hex_num2 = f"{num2:03x}"

    # 3. 280~380 均匀整数 → 5位16进制补0
    num3 = random.randint(280, 380)
    hex_num3 = f"{num3:05x}"

    # 4. 262~531 均匀整数 → 5位16进制补0
    num4 = random.randint(262, 531)
    hex_num4 = f"{num4:05x}"

    # 5. 0~200 均匀整数 → 4位16进制补0
    num5 = random.randint(0, 200)
    hex_num5 = f"{num5:04x}"

    # 拼接
    result = hex_ts + hex_num2 + hex_num3 + hex_num4 + hex_num5

    return result


