import json
import re
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_FILE = "../data/sft_consensus_abstract_cot.json"
OUTPUT_FILE = "../data/sft_consensus_ten_scale.json"
# ===========================================

def custom_round(score_100):
    """
    100分制转10分制逻辑：
    1. 转为浮点数，如 45 -> 4.5
    2. 如果在 [5.0, 5.99] 之间，强制设为 5 (不可进入 Accept 档位)
    3. 其他分数采用标准四舍五入
    """
    val = score_100 / 10.0
    
    # 特殊逻辑：5.x 分强制截断为 5
    if 5.0 <= val < 6.0:
        return 5
    
    # 标准四舍五入
    return int(val + 0.5)

def process_data():
    print(f"📖 正在读取: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"🚀 开始转换数据格式 (System Prompt + Assistant Scores)...")
    
    converted_scores_count = 0
    updated_system_count = 0

    for item in tqdm(data):
        # 1. 修改 System Prompt (通常在 item['system'] 或 conversations 的第一个条目)
        # 检查 item 级别的 system 字段
        if "system" in item:
            old_sys = item["system"]
            item["system"] = old_sys.replace("(0-100)", "(0-10)")
            if item["system"] != old_sys:
                updated_system_count += 1
        
        # 遍历对话列表
        if "conversations" in item:
            for msg in item["conversations"]:
                # 检查消息条目里的 system
                if msg["from"] == "system":
                    old_val = msg["value"]
                    msg["value"] = old_val.replace("(0-100)", "(0-10)")
                    if msg["value"] != old_val:
                        updated_system_count += 1
                
                # 2. 修改 Assistant 的分数输出
                elif msg["from"] == "assistant":
                    content = msg["value"]
                    # 正则查找 **Final Score**: XX/100
                    match = re.search(r"\*\*Final Score\*\*: (\d+)/100", content)
                    if match:
                        original_score_100 = int(match.group(1))
                        new_score_10 = custom_round(original_score_100)
                        
                        # 替换字符串：**Final Score**: X/10
                        new_string = f"**Final Score**: {new_score_10}/10"
                        msg["value"] = content.replace(match.group(0), new_string)
                        converted_scores_count += 1

    print(f"💾 正在保存到: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("\n" + "="*40)
    print(f"✅ 转换完成！")
    print(f"📊 修改 System Prompt 数: {updated_system_count}")
    print(f"📊 修改 Assistant 分数数: {converted_scores_count}")
    print(f"⚖️ 规则: (0-100) -> (0-10), 5.x 强制归 5")
    print("="*40)

if __name__ == "__main__":
    process_data()