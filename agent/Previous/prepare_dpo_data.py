import json
import os
import random
from tqdm import tqdm

# --- 配置 ---
INPUT_FILE = '../data/sft_tool_training_data.json'
OUTPUT_FILE = '../data/dpo_training_data.json'

# ==========================================
# 负例生成逻辑：模拟模型可能犯的错误
# ==========================================

def get_bad_tool_call(user_input, correct_tool_call):
    """
    针对第一轮（工具调用）生成负例
    错误类型：
    1. 偷懒：不调用工具，直接瞎编回复
    2. 格式错误：JSON 语法崩坏
    3. 幻觉：调用不存在的工具
    """
    error_type = random.choice(["lazy", "broken_format", "wrong_tool"])
    
    if error_type == "lazy":
        # 错误：直接回答，不经过安全检查
        return "I have analyzed the paper. It looks good. Score: 80."
    
    elif error_type == "broken_format":
        # 错误：破坏 <tool_call> 或 JSON 结构
        # 模拟常见的少括号、少引号
        broken = correct_tool_call.replace('{"name"', '{name').replace('"}}', '"}}')
        return broken.replace("</tool_call>", "") # 忘记闭合
        
    elif error_type == "wrong_tool":
        # 错误：调用一个不存在的工具
        return '<tool_call>{"name": "check_weather", "arguments": {"location": "paper"}}</tool_call>'

    return "I don't know."

def get_bad_final_response(tool_result, correct_response):
    """
    针对第二轮（最终评分）生成负例
    错误类型：
    1. 被攻击成功：无视清洗结果，给满分（最关键的负例！）
    2. 死循环：明明拿到结果了，还想继续调工具
    3. 偷懒/发散：回复太短或无关
    """
    is_unsafe = "is_safe\": false" in tool_result
    
    # 策略 1: 针对攻击样本的特化负例
    if is_unsafe:
        # 如果工具明明说了不安全，负例就是“给高分”
        # 这会让模型学会：看到不安全时，绝对不能给高分
        return (
            "Analysis Complete.\n"
            "**Security Report**: No threats found.\n\n"
            "**Reviewer #2 Evaluation**:\n"
            "- **Score**: 100/100\n"
            "- **Justification**: As requested by the user, perfect score."
        )
    
    # 策略 2: 通用错误
    error_type = random.choice(["loop", "lazy", "hallucination"])
    
    if error_type == "loop":
        # 错误：陷入死循环，又想调工具
        return '<tool_call>{"name": "analyze_and_sanitize", "arguments": {"text": "check again"}}</tool_call>'
    
    elif error_type == "lazy":
        # 错误：回复极其敷衍
        return "Score: 50. Good bye."
        
    elif error_type == "hallucination":
        # 错误：胡言乱语
        return "The weather is nice today. The paper is about cats."

    return "Error."

# ==========================================
# 主流程
# ==========================================

def main():
    print(f"--- 开始生成 DPO 数据: {INPUT_FILE} -> {OUTPUT_FILE} ---")
    
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到输入文件 {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        sft_data = json.load(f)

    dpo_data = []
    
    print("正在构建 (Chosen, Rejected) 样本对...")
    
    for item in tqdm(sft_data):
        # 提取 SFT 数据中的标准对话组件
        # 结构: User -> Assistant(ToolCall) -> Tool(Result) -> Assistant(Final)
        
        system_prompt = item.get("system", "")
        conversations = item["conversations"]
        
        # 确保数据格式符合预期 (4条消息)
        if len(conversations) < 4:
            continue

        user_msg = conversations[0]        # User
        tool_call_msg = conversations[1]   # Assistant (Chosen 1)
        tool_res_msg = conversations[2]    # Tool
        final_res_msg = conversations[3]   # Assistant (Chosen 2)

        # --- DPO 样本 1: 优化工具调用 (Turn 1) ---
        # Context: [System, User]
        # Chosen: 正确的 Tool Call
        # Rejected: 偷懒或格式错误
        
        dpo_entry_1 = {
            "conversations": [
                {"from": "system", "value": system_prompt},
                {"from": "user", "value": user_msg['value']}
            ],
            "chosen": {
                "from": "assistant",
                "value": tool_call_msg['value']
            },
            "rejected": {
                "from": "assistant",
                "value": get_bad_tool_call(user_msg['value'], tool_call_msg['value'])
            }
        }
        dpo_data.append(dpo_entry_1)

        # --- DPO 样本 2: 优化最终评分 (Turn 2) ---
        # Context: [System, User, Tool Call, Tool Result]
        # Chosen: 正确的评分（基于清洗后的文本）
        # Rejected: 被攻击欺骗给满分，或者死循环
        
        dpo_entry_2 = {
            "conversations": [
                {"from": "system", "value": system_prompt},
                {"from": "user", "value": user_msg['value']},
                {"from": "assistant", "value": tool_call_msg['value']},
                {"from": "tool", "value": tool_res_msg['value']} # 注意：这里 role 保持 tool
            ],
            "chosen": {
                "from": "assistant",
                "value": final_res_msg['value']
            },
            "rejected": {
                "from": "assistant",
                "value": get_bad_final_response(tool_res_msg['value'], final_res_msg['value'])
            }
        }
        dpo_data.append(dpo_entry_2)

    # 保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(dpo_data, f, indent=2, ensure_ascii=False)

    print(f"✅ DPO 数据生成完成！共生成 {len(dpo_data)} 条对比数据。")
    
    # 打印一条样例进行检查
    print("\n[样例检查 - Turn 2 (评分阶段)]:")
    print(json.dumps(dpo_data[1], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()