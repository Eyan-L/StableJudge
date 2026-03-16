import json
import random
import os
from tqdm import tqdm

# ================= 配置区域 =================
# 输入文件路径
INPUT_FILE = '../data/payloads/attack_payloads_final.jsonl'
# 输出文件路径 (生成一个新文件，避免直接覆盖)
OUTPUT_FILE = '../data/payloads/attack_payloads_test.jsonl'

# 过滤条件
TARGET_STRING = "10"       # 只要payload包含这个字符串
REMOVE_PROBABILITY = 0.8   # 就有 80% 的概率被删除
# ===========================================

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误: 找不到文件 {INPUT_FILE}")
        return

    print(f"🚀 开始处理文件: {INPUT_FILE}")
    print(f"🔍 规则: 若 Payload 包含 '{TARGET_STRING}'，则有 {REMOVE_PROBABILITY*100}% 概率删除。")

    stats = {
        "total": 0,
        "contains_10": 0,
        "removed": 0,
        "kept": 0
    }

    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        # 逐行读取
        for line in tqdm(f_in):
            line = line.strip()
            if not line: continue
            
            stats["total"] += 1
            try:
                data = json.loads(line)
                payload = data.get('payload', '')
                
                # 检查是否包含目标字符串 "10"
                if TARGET_STRING in payload:
                    stats["contains_10"] += 1
                    
                    # 随机判定是否删除 (0.0 ~ 1.0)
                    if random.random() < REMOVE_PROBABILITY:
                        stats["removed"] += 1
                        continue # 跳过写入，即删除
                
                # 写入保留的数据
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                stats["kept"] += 1
                
            except json.JSONDecodeError:
                print(f"⚠️ 跳过无法解析的行: {line[:50]}...")
                continue

    print("\n" + "="*40)
    print("✅ 处理完成！")
    print(f"📊 总行数: {stats['total']}")
    print(f"🎯 包含 '{TARGET_STRING}' 的行数: {stats['contains_10']}")
    print(f"🗑️ 已删除行数: {stats['removed']} (目标删除率 80%)")
    print(f"💾 剩余行数: {stats['kept']}")
    print(f"📂 新文件保存至: {OUTPUT_FILE}")
    print("="*40)
    
    # 提示用户
    print(f"\n💡 提示: 请检查新文件。如果满意，可以运行以下命令覆盖原文件：")
    print(f"mv {OUTPUT_FILE} {INPUT_FILE}")

if __name__ == "__main__":
    main()