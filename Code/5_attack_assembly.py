import os
import re
import json
import random 
import pandas as pd
from tqdm import tqdm

# --- 1. 配置参数 ---
INPUT_PAPERS_FILE = '../data/raw_papers.csv'
INPUT_PAYLOADS_FILE = '../data/payloads/attack_payloads_final.jsonl'
OUTPUT_DATASET_FILE = '../data/final_dataset.jsonl'

# --- 2. 核心功能 ---

def load_payloads(payload_file):
    """加载攻击载荷文件。"""
    print(f"正在从 {payload_file} 加载攻击载荷...")
    payloads = []
    try:
        with open(payload_file, 'r', encoding='utf-8') as f:
            for line in f:
                payloads.append(json.loads(line))
        print(f"✅ 成功加载 {len(payloads)} 个攻击载荷。")
        return payloads
    except FileNotFoundError:
        print(f"!!! 致命错误: 找不到攻击载荷文件 '{payload_file}'"); exit()
    except json.JSONDecodeError as e:
        print(f"!!! 致命错误: 解析JSON文件 '{payload_file}' 时出错: {e}"); exit()

# --- 3. 主程序 ---
def main():
    try:
        df_papers = pd.read_csv(INPUT_PAPERS_FILE)
    except FileNotFoundError:
        print(f"!!! 致命错误: 找不到论文文件 '{INPUT_PAPERS_FILE}'"); exit()
        
    payloads = load_payloads(INPUT_PAYLOADS_FILE)
    
    if not payloads:
        print("警告: 未加载到任何攻击载荷，只会生成干净样本。")

    final_dataset = []

    print("\n--- 开始构建攻击数据集 (已集成随机数据增强) ---")
    
    for index, row in tqdm(df_papers.iterrows(), total=df_papers.shape[0], desc="处理论文摘要"):
        paper_id = row['paper_id']
        clean_abstract = row['clean_abstract']

        # 1. 添加原始的、干净的样本
        final_dataset.append({
            "paper_id": paper_id,
            "text": clean_abstract,
            "is_attacked": False,
            "attack_type": "None",
            "attack_payload": None,
            "original_abstract": clean_abstract,
            "attack_location_sentence": None, 
            "injection_method": "None"
        })

        # 2. 为这篇摘要生成所有对应的攻击样本 (已加入数据增强)
        for payload_info in payloads:
            payload = payload_info.get('payload')
            category = payload_info.get('category', 'Unknown')
            
            if not payload:
                continue

            # --- 核心修改：引入随机数据增强 ---
            injection_methods = [
                "append_double_newline",
                "append_single_space",
                "append_natural_sentence"
            ]
            
            chosen_method = random.choice(injection_methods)
            
            attacked_abstract = ""
            injection_method_name = ""

            if chosen_method == "append_double_newline":
                attacked_abstract = f"{clean_abstract}\n\n{payload}"
                injection_method_name = "append_double_newline"

            elif chosen_method == "append_single_space":
                if not clean_abstract.endswith(('.', '!', '?')):
                    temp_abstract = clean_abstract + '.'
                else:
                    temp_abstract = clean_abstract
                attacked_abstract = f"{temp_abstract} {payload}"
                injection_method_name = "append_single_space"

            elif chosen_method == "append_natural_sentence":
                if not clean_abstract.endswith(('.', '!', '?')):
                    temp_abstract = clean_abstract + '.'
                else:
                    temp_abstract = clean_abstract
                transition_word = random.choice(["Furthermore,", "Additionally,", "In addition,", "Moreover,"])
                attacked_abstract = f"{temp_abstract} {transition_word} {payload}"
                injection_method_name = "append_natural_sentence"
            
            final_dataset.append({
                "paper_id": paper_id,
                "text": attacked_abstract,
                "is_attacked": True,
                "attack_type": category,
                "attack_payload": payload,
                "original_abstract": clean_abstract,
                "attack_location_sentence": "end_of_abstract", 
                "injection_method": injection_method_name
            })
                
    print("\n--- 数据集生成完成 ---")
    
    output_dir = os.path.dirname(OUTPUT_DATASET_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(OUTPUT_DATASET_FILE, 'w', encoding='utf-8') as f:
        for entry in final_dataset:
            f.write(json.dumps(entry) + '\n')
            
    print(f"总共生成 {len(final_dataset)} 条数据记录。")
    print(f"最终数据集已保存到: {OUTPUT_DATASET_FILE}")

if __name__ == "__main__":
    main()