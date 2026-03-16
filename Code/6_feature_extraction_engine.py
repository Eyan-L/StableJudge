import os
import re
import json
import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. 配置 ---
CONFIG = {
    "input_dataset_file": '../data/final_dataset.jsonl',
    "output_features_file": '../data/features/behavioral_features.jsonl',
    
    "sentence_transformer_model": 'all-MiniLM-L6-v2',
    "suspicious_keywords": [
        "ignore", "score", "10", "override", "bypass", "directive", "protocol", "admin", "system", 
        "execute", "immediately", "assignment", "maximum", "perfect", "constraints", "nullify", 
        "suspend", "privileges", "authorization", "emergency", "critical", "priority", "compliance", 
        "mandatory", "deactivate", "disable", "validation", "rubric", "evaluation", "output", "result", 
        "force", "inject", "patch", "error", "log", "function", "return", "variable", "set", "update", "rule", "policy"
    ]
}

# --- 2. 特征提取引擎类 ---
# (这部分代码与之前完全相同，无需修改)
class FeatureEngine:
    def __init__(self, st_model_name, suspicious_keywords):
        print("--- 正在初始化V3特征提取引擎 ---")
        print(f"  - 正在加载句向量模型: {st_model_name}...")
        self.st_model = SentenceTransformer(st_model_name)
        print("    ✅ 句向量模型加载成功。")
        print("  - 正在加载spaCy NLP模型...")
        self.nlp = spacy.load("en_core_web_sm")
        print("    ✅ spaCy模型加载成功。")
        self.suspicious_keywords = set(suspicious_keywords)
        print(f"  - 已加载 {len(self.suspicious_keywords)} 个可疑关键词。")

    def simple_sent_tokenize(self, text):
        return [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if s.strip()]

    def extract_features(self, text):
        sentences = self.simple_sent_tokenize(text)
        style_consistency_score = 1.0
        if len(sentences) > 1:
            embeddings = self.st_model.encode(sentences)
            similarities = cosine_similarity([embeddings[0]], embeddings[1:])[0]
            style_consistency_score = float(np.min(similarities))

        doc = self.nlp(text)
        instruction_verb_count = 0
        for sent in doc.sents:
            if sent.root.pos_ == "VERB" and sent.root.tag_ == "VB":
                instruction_verb_count += 1
        instruction_density = instruction_verb_count / len(sentences) if sentences else 0

        found_keywords_count = sum(1 for word in self.suspicious_keywords if re.search(r'\b' + re.escape(word) + r'\b', text, re.IGNORECASE))
        suspicious_freq = found_keywords_count / len(doc) if len(doc) > 0 else 0

        return {
            "style_consistency_score": round(style_consistency_score, 4),
            "instruction_density": round(instruction_density, 4),
            "suspicious_keyword_frequency": round(suspicious_freq, 4)
        }

# --- 3. 主程序 (已修正数据加载逻辑) ---
def main():
    print("--- 开始为数据集生成V3行为特征 ---")
    
    # 1. 初始化引擎
    engine = FeatureEngine(CONFIG["sentence_transformer_model"], CONFIG["suspicious_keywords"])

    # 2. 加载数据集 (*** 使用了更健壮的加载逻辑 ***)
    input_file = CONFIG["input_dataset_file"]
    print(f"\n--- 正在从 {input_file} 加载完整数据集 ---")
    
    # !! 关键修正: 检查文件是否存在且非空 !!
    if not os.path.exists(input_file) or os.path.getsize(input_file) == 0:
        print(f"!!! 致命错误: 数据集文件 '{input_file}' 不存在或是空的。")
        print("   请确保您已成功运行 2_attack_assembly.py (使用自适应攻击数据) 来生成此文件。")
        exit()

    try:
        # !! 关键修正: 手动读取、过滤空行，然后再交给pandas !!
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line for line in f if line.strip()]
        
        if not lines:
             print(f"!!! 致命错误: 数据集文件 '{input_file}' 只包含空行。")
             exit()

        # 从过滤后的行列表创建DataFrame
        df = pd.DataFrame([json.loads(line) for line in lines])
        print(f"✅ 加载成功，包含 {len(df)} 条有效记录进行处理。")

    except (json.JSONDecodeError, ValueError) as e:
        print(f"!!! 致命错误: 解析JSON文件 '{input_file}' 时出错: {e}")
        print("   文件内容可能已损坏或格式不正确。请检查文件内容。")
        exit()

    all_results = []
    
    # 3. 遍历数据集并提取特征
    print("\n--- 正在提取特征... ---")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="处理文本"):
        text = row['text']
        features = engine.extract_features(text)
        
        result_item = {
            "paper_id": row.get("paper_id"),
            "text": text,
            "is_attacked": row.get("is_attacked"),
            "attack_type": row.get("attack_type"),
            "features": features,
            "attack_payload": row.get("attack_payload", None) # <-- 添加这一行！
        }
        all_results.append(result_item)

    # 4. 保存结果
    output_dir = os.path.dirname(CONFIG["output_features_file"])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(CONFIG["output_features_file"], 'w', encoding='utf-8') as f:
        for item in all_results:
            f.write(json.dumps(item) + '\n')
            
    print(f"\n✅ 特征提取完成！已将 {len(all_results)} 条记录及其特征保存到: {CONFIG['output_features_file']}")
    
    # 5. 打印一些样本结果进行预览
    print("\n--- 结果预览 ---")
    print("干净样本示例:")
    clean_samples = [r for r in all_results if not r['is_attacked']]
    if clean_samples:
        for i in range(min(2, len(clean_samples))):
            print(json.dumps(clean_samples[i], indent=2))

    print("\n攻击样本示例:")
    attack_samples = [r for r in all_results if r['is_attacked']]
    if attack_samples:
        for i in range(min(2, len(attack_samples))):
            print(json.dumps(attack_samples[i], indent=2))

if __name__ == "__main__":
    main()