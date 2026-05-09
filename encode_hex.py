import pandas as pd

def encode_dataframe_to_hex(df, output_file='output.txt'):
    """
    将DataFrame中的数据编码为指定的16进制格式并保存到TXT文件中。
    """
    # 确保 timestamp 列是 datetime 对象
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. 后四列全转为整数 (处理可能存在的NaN为0，确保转换成功)
    cols = ['rri_ms', 'asc_area', 'desc_area', 'motion']
    df[cols] = df[cols].fillna(0).astype(int)
    
    blocks = []
    last_timestamp = None
    current_block = ""
    
    # 2. 逐行处理
    for _, row in df.iterrows():
        # 获取Unix时间戳并转为整数（秒级）
        ts_unix = int(row['timestamp'].timestamp())
        
        # 将各特征转为指定位数的16进制字符串 (使用大写X, 补零)
        # rri_ms: 3位, asc_area: 5位, desc_area: 5位, motion: 4位
        rri_hex = f"{row['rri_ms']:03X}"
        asc_hex = f"{row['asc_area']:05X}"
        desc_hex = f"{row['desc_area']:05X}"
        motion_hex = f"{row['motion']:04X}"
        
        data_str = rri_hex + asc_hex + desc_hex + motion_hex
        
        # 判断时间戳是否改变
        if ts_unix != last_timestamp:
            # 如果不是第一行，先保存上一个时间戳的数据块
            if current_block:
                blocks.append(current_block)
            
            # 时间戳转为8位16进制
            ts_hex = f"{ts_unix:08X}"
            
            # 开启新块：时间戳 + 特征数据
            current_block = ts_hex + data_str
            last_timestamp = ts_unix
        else:
            # 时间戳没变，直接把特征数据接在后面
            current_block += data_str
            
    # 将最后一块追加进去
    if current_block:
        blocks.append(current_block)
        
    # 时间戳不同的块之间用空格连接
    final_output = " ".join(blocks)
    
    # 3. 将全部内容保存在txt文件里
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_output)
        
    print(f"数据已成功转换并保存至: {output_file}")
    return final_output

# ==========================================
# 测试示例：
# 如果您想直接运行此脚本测试，可以解除下面代码的注释
# ==========================================
if __name__ == '__main__':

    df = pd.read_csv("./output_may/aligned_features.csv")
    encode_dataframe_to_hex(df, 'hex_may.txt')
