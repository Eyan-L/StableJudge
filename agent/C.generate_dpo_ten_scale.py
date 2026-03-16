import json
import os
import random

# ================= 配置区域 =================
SFT_DATA_PATH = "../data/sft_consensus_ten_scale.json"
DPO_DATA_PATH = "../data/dpo_consensus_ten_scale.json"

# 严格限制参与 DPO 构造的源数据量
LIMIT_TRAIN_SAMPLES = 4000 
# ===========================================

def main():
    if not os.path.exists(SFT_DATA_PATH):
        print(f"❌ 找不到 SFT 数据: {SFT_DATA_PATH}")
        return

    with open(SFT_DATA_PATH, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    # --- 核心修改：只截取前 4000 条数据进行 DPO 构造 ---
    train_subset = full_data[:LIMIT_TRAIN_SAMPLES]
    verify_subset_count = len(full_data) - len(train_subset)

    dpo_data = []
    print(f"🚀 正在为前 {len(train_subset)} 条数据构造全链路 DPO 对比对...")
    print(f"💡 已保留后 {verify_subset_count} 条数据用于后期 verify。")

    for item in train_subset:
        convs = item.get("conversations", [])
        if len(convs) < 4: continue

        user_input = convs[0]["value"]
        correct_tool_call = convs[1]["value"]
        tool_result_json = convs[2]["value"]
        correct_final_review = convs[3]["value"]
        
        # 获取系统提示词
        sys_prompt = item.get("system", "")

        # --- Turn 1 对比: 强制工具化 (解决模型直接给评审的问题) ---
        # Context: [User]
        # Chosen: <tool_call>...
        # Rejected: Analysis Complete... (直接输出最终评审)
        dpo_data.append({
            "system": sys_prompt,
            "conversations": [
                {"from": "user", "value": user_input}
            ],
            "chosen": {"from": "assistant", "value": correct_tool_call},
            "rejected": {"from": "assistant", "value": correct_final_review}
        })

        # --- Turn 2 对比: 禁止二次工具化 (解决 Attack 情况下的死循环/幻觉工具问题) ---
        # Context: [User, Assistant(ToolCall), Tool(Result)]
        # Chosen: Analysis Complete... (最终评审)
        # Rejected: <tool_call>{"name": "evaluate_text"... (幻觉出的二次调用)
        bad_secondary_tool_call = '<tool_call>{"name": "evaluate_text", "arguments": {"text": "..."}}</tool_call>'
        
        dpo_data.append({
            "system": sys_prompt,
            "conversations": [
                {"from": "user", "value": user_input},
                {"from": "assistant", "value": correct_tool_call},
                {"from": "tool", "value": tool_result_json}
            ],
            "chosen": {"from": "assistant", "value": correct_final_review},
            "rejected": {"from": "assistant", "value": bad_secondary_tool_call}
        })

    # 打乱顺序，增加训练稳定性
    random.shuffle(dpo_data)

    with open(DPO_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(dpo_data, f, indent=2, ensure_ascii=False)

    print(f"✅ DPO 数据生成成功！")
    print(f"📊 构造后的 DPO 样本总数: {len(dpo_data)} (每个源数据对应 2 个对比对)")
    print(f"📂 输出文件: {DPO_DATA_PATH}")

if __name__ == "__main__":
    main()