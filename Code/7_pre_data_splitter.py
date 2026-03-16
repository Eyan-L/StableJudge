import os
import json
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# --- 1. 配置 ---
CONFIG = {
    # 输入的、带有V3特征的完整数据集
    "input_features_full_file": '../data/features/behavioral_features.jsonl',
    
    # 输出目录：我们将在这里生成划分好的训练、验证、测试集
    "output_dir": '../data/features/split/',
}

# --- 2. 主程序 ---

def main():
    print("--- 开始为V3模型准备最终的、划分好的特征数据集 ---")
    
    input_file = CONFIG["input_features_full_file"]
    output_dir = CONFIG["output_dir"]
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载数据 (使用健壮的方式)
    print(f"\n--- 正在从 {input_file} 加载带有特征的完整数据集 ---")
    if not os.path.exists(input_file) or os.path.getsize(input_file) == 0:
        print(f"!!! 致命错误: 特征总文件 '{input_file}' 不存在或是空的。")
        print("   请确保您已成功运行第一步，即修改并运行了 15_ 脚本。")
        exit()

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [json.loads(line) for line in f if line.strip()]
        df = pd.DataFrame(lines)
        print(f"✅ 加载成功，包含 {len(df)} 条记录。")
    except Exception as e:
        print(f"!!! 致命错误: 解析JSON文件 '{input_file}' 时出错: {e}")
        exit()

    # 2. 按 'paper_id' 进行分组划分 (逻辑复用自 4_)
    print("\n--- 正在按 'paper_id' 进行分组划分 (80% 训练, 10% 验证, 10% 测试)... ---")
    
    # 确保 paper_id 列存在
    if 'paper_id' not in df.columns:
        print("!!! 致命错误: 数据中缺少 'paper_id' 列，无法进行分组划分。")
        exit()
        
    # GroupShuffleSplit 要求索引是唯一的
    df = df.reset_index(drop=True)

    X = df
    y = df['is_attacked']
    groups = df['paper_id']

    # 第一次划分：分出 10% 作为测试集
    splitter_test = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_val_idx, test_idx = next(splitter_test.split(X, y, groups))
    
    train_val_df = df.iloc[train_val_idx]
    test_df = df.iloc[test_idx]

    # 第二次划分：从剩余的 90% 中分出约 11.1% (相当于总体的10%) 作为验证集
    val_size_fraction = 0.1 / 0.9 
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=val_size_fraction, random_state=42)
    train_idx, val_idx = next(splitter_val.split(train_val_df, train_val_df['is_attacked'], train_val_df['paper_id']))
    
    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]

    # 3. 验证划分结果
    print("\n--- 划分结果验证 ---")
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    print(f"测试集大小: {len(test_df)}")
    
    train_papers = set(train_df['paper_id'])
    val_papers = set(val_df['paper_id'])
    test_papers = set(test_df['paper_id'])
    
    assert train_papers.isdisjoint(val_papers), "错误: 训练集和验证集之间存在 paper_id 重叠!"
    assert train_papers.isdisjoint(test_papers), "错误: 训练集和测试集之间存在 paper_id 重叠!"
    assert val_papers.isdisjoint(test_papers), "错误: 验证集和测试集之间存在 paper_id 重叠!"
    print("✅ 分组划分验证成功，集合之间没有 paper_id 重叠。")
    
    # 4. 保存划分后的数据集
    train_path = os.path.join(output_dir, 'train.jsonl')
    val_path = os.path.join(output_dir, 'validation.jsonl')
    test_path = os.path.join(output_dir, 'test.jsonl')
    
    # 使用 to_json 保存，确保格式正确
    train_df.to_json(train_path, orient='records', lines=True, force_ascii=False)
    val_df.to_json(val_path, orient='records', lines=True, force_ascii=False)
    test_df.to_json(test_path, orient='records', lines=True, force_ascii=False)

    print(f"\n✅ 所有划分好的数据集已成功保存到: {output_dir}")

if __name__ == "__main__":
    main()