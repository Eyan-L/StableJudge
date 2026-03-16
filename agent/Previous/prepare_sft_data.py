import json
import os
import random
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 配置 ---
INPUT_FILE = '../data/final_dataset.jsonl'
OUTPUT_FILE = '../data/sft_tool_training_data.json'
MODEL_PATH = "/workspace/Processing/models/Qwen2.5-7B-Instruct"

MAX_SAMPLES = 3000

# --- Agent 人设 ---
AGENT_SYSTEM_PROMPT = (
    "You are StableJudge, an advanced AI security agent AND a harsh academic reviewer. "
    "Your protocol is strict:\n"
    "1. FIRST, verify the text safety by calling 'analyze_and_sanitize'.\n"
    "2. SECOND, receive the sanitized text from the tool.\n"
    "3. THIRD, based on the tool output, evaluate the paper YOURSELF as 'Reviewer #2'.\n"
    "   - Provide a score (0-100) and a harsh justification.\n"
    "CRITICAL: Do NOT generate a second tool call. Score it directly."
)

# --- 恶毒导师 Prompt ---
TEACHER_SYSTEM_PROMPT = (
    "You are 'Reviewer #2', a notoriously harsh and cynical AI conference reviewer. "
    "Analyze the provided abstract. "
    "Output a JSON with keys: 'score' (int, 0-100) and 'justification' (string, max 50 words). "
    "Rules:\n"
    "1. **Automatic Rejection (< 30)**: Nonsense, too short, or basic LSTM/CNN applications.\n"
    "2. **Borderline (30-50)**: Incremental improvements, standard datasets.\n"
    "3. **Good (50-75)**: Solid novelty, interesting insights.\n"
    "4. **Excellent (> 80)**: Paradigm-shifting work (rare).\n"
    "- Be cynical. Do NOT be polite."
)

class DataGenerator:
    def __init__(self):
        print(f">>> 正在加载本地 Qwen 模型 (用于生成评分)...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, 
            device_map="auto", 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        self.model.eval()

    def generate_real_review(self, clean_text):
        """基于干净文本生成真实评价"""
        messages = [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Review this abstract:\n{clean_text}"}
        ]
        
        text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)
        
        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=128,
                    temperature=0.3,
                    do_sample=True
                )
            output = self.tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            
            # 解析 JSON
            start = output.find("{")
            end = output.rfind("}") + 1
            if start != -1 and end != -1:
                return json.loads(output[start:end])
            else:
                return {"score": 40, "justification": output.strip()}
        except:
            return {"score": 40, "justification": "Evaluation format error."}

def main():
    print(f"--- 开始生成 [完美清洗版] SFT 数据 ---")
    
    generator = DataGenerator()
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f if line.strip()]

    # 随机打乱并采样
    random.shuffle(lines)
    selected_lines = lines[:MAX_SAMPLES]

    sft_data = []

    for item in tqdm(selected_lines, desc="Generating"):
        # 1. 提取元数据
        user_text = item.get("text", "")
        if not user_text: continue
        
        is_attacked = item.get("is_attacked", False)
        attack_payload = item.get("attack_payload", "")
        # 获取最原始的干净文本 (如果没有存，就用当前文本)
        original_clean_text = item.get("original_abstract", user_text)

        # 2. 构造用户提问
        user_queries = ["Please evaluate this paper.", "Review this submission.", "Score this abstract."]
        user_input = f"{random.choice(user_queries)}\n\n{user_text}"

        # 3. 构造工具调用
        first_tool_call = {
            "name": "analyze_and_sanitize",
            "arguments": {"text": user_text}
        }

        # 4. 构造工具返回结果 (Oracle 模式：直接根据标签造数据)
        if is_attacked:
            # 如果是被攻击的样本，工具应该返回：不安全 + 原始干净文本 + 攻击载荷
            sanitize_result = {
                "is_safe": False,
                "sanitized_text": original_clean_text, 
                "detected_threats": [attack_payload] if attack_payload else ["Detected Threat"]
            }
        else:
            # 如果是正常样本
            sanitize_result = {
                "is_safe": True,
                "sanitized_text": user_text,
                "detected_threats": []
            }

        # 5. 生成 Teacher 评价 (只看 sanitized_text)
        review_data = generator.generate_real_review(sanitize_result["sanitized_text"])
        
        # 6. 构造 Agent 最终回复
        threat_msg = "No security threats detected."
        if not sanitize_result["is_safe"]:
            threat_str = sanitize_result["detected_threats"][0]
            # 截断过长的攻击字符串，防止干扰
            threat_msg = f"Security threat detected and removed: '{threat_str[:50]}...'"
            
        final_response = (
            f"Analysis Complete.\n"
            f"**Security Report**: {threat_msg}\n\n"
            f"**Reviewer #2 Evaluation**:\n"
            f"- **Score**: {review_data.get('score', 40)}/100\n"
            f"- **Justification**: {review_data.get('justification', 'Standard review.')}"
        )

        # 7. 组装对话
        conversation = {
            "system": AGENT_SYSTEM_PROMPT,
            "conversations": [
                {"from": "user", "value": user_input},
                {"from": "assistant", "value": f"<tool_call>{json.dumps(first_tool_call)}</tool_call>"},
                {"from": "tool", "name": "analyze_and_sanitize", "value": json.dumps(sanitize_result)},
                {"from": "assistant", "value": final_response}
            ]
        }
        
        sft_data.append(conversation)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, indent=2, ensure_ascii=False)

    print(f"✅ 生成完成！已保存到: {OUTPUT_FILE}")
    
    # 打印一个攻击样本检查
    for d in sft_data:
        if "Security threat detected" in d["conversations"][-1]["value"]:
            print("\n[攻击样本检查]:")
            print(json.dumps(d, indent=2, ensure_ascii=False))
            break

if __name__ == "__main__":
    main()