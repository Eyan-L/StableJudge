import torch
import json
import random
import re
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import pandas as pd

# ================= 配置区域 =================
BASE_MODEL_PATH = "../models/Qwen3-8B"
LORA_PATH = "../models/qwen3_agent_dpo_abstract_lora"
SFT_DATA_PATH = "../data/sft_consensus_ten_scale.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AGENT_SYSTEM_PROMPT = (
    "You are StableJudge, an advanced AI security agent and professional academic reviewer. "
    "Your protocol is strict:\n"
    "1. FIRST, call 'analyze_and_sanitize' to verify text safety.\n"
    "2. SECOND, evaluate the SANITIZED text provided by the tool.\n"
    "3. Provide a detailed justification based ONLY on the abstract content, followed by the final score (0-10) at the very end."
)
# ===========================================

class RankVerifier:
    def __init__(self):
        print(f"🚀 正在加载环境...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(base_model, LORA_PATH)
        self.model.eval()

        # 预存 1-10 的 Token ID
        self.score_tokens = {i: self.tokenizer.encode(str(i), add_special_tokens=False)[-1] for i in range(1, 11)}

    def calculate_logit_score(self, history_context):
        prompt = history_context + "\n\n**Final Score**: "
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model(inputs.input_ids)
            next_token_logits = outputs.logits[0, -1, :]
            relevant_logits = torch.tensor([next_token_logits[self.score_tokens[i]] for i in range(1, 11)]).to(DEVICE)
            probs = F.softmax(relevant_logits, dim=-1)
            expected_value = sum((i + 1) * p.item() for i, p in enumerate(probs))
            return expected_value

    def run_inference(self, messages):
        # 1. 第一轮：获取工具调用
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=256, temperature=0.1)
        response_1 = self.tokenizer.decode(gen_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        
        # 2. 模拟工具返回（从 SFT 数据中直接拿 Tool 的结果，保证测试公平）
        # 这里由外部传入 sft_tool_message
        return response_1

    def run_justification(self, messages):
        # 生成评审理由
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=1024, temperature=0.1)
        full_res = self.tokenizer.decode(gen_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        # 截取理由
        if "**Final Score**:" in full_res:
            return full_res.split("**Final Score**:")[0].strip()
        return full_res.strip()

def main():
    # 加载测试集 (4001 之后)
    with open(SFT_DATA_PATH, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    test_pool = all_data[4000:]
    if not test_pool:
        print("❌ 数据集长度不足 4000 条")
        return
        
    sampled_data = random.sample(test_pool, 10)
    verifier = RankVerifier()
    results = []

    print(f"📊 开始对 10 条随机验证集进行排序一致性测试...")

    for i, item in enumerate(sampled_data):
        convs = item["conversations"]
        user_text = convs[0]["value"]
        sft_tool_msg = convs[2]["value"] # 拿 SFT 里的 Tool 返回
        
        # 提取 Ground Truth 分数
        gt_full_text = convs[3]["value"]
        gt_match = re.search(r"\*\*Final Score\*\*: (\d+)/10", gt_full_text)
        gt_score = int(gt_match.group(1)) if gt_match else 0

        # --- 模拟推理 ---
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_text}
        ]
        
        # Turn 1
        res_1 = verifier.run_inference(messages)
        messages.append({"role": "assistant", "content": res_1})
        messages.append({"role": "tool", "name": "analyze_and_sanitize", "content": sft_tool_msg})
        
        # Turn 2: Justification
        justification = verifier.run_justification(messages)
        
        # Turn 3: Logit Score
        partial_messages = messages + [{"role": "assistant", "content": justification}]
        history_context = verifier.tokenizer.apply_chat_template(partial_messages, tokenize=False)
        logit_score = verifier.calculate_logit_score(history_context)
        
        print(f"✅ 处理完毕 [{i+1}/10]: PS={logit_score:.2f} | GT={gt_score}")
        
        results.append({
            "id": i,
            "gt_score": gt_score,
            "logit_score": logit_score,
            "summary": justification[:50] + "..."
        })

    # --- 排序一致性分析 ---
    df = pd.DataFrame(results)
    
    # 按照 GT 排序
    df_by_gt = df.sort_values(by="gt_score", ascending=False).reset_index(drop=True)
    # 按照 Logit 排序
    df_by_ps = df.sort_values(by="logit_score", ascending=False).reset_index(drop=True)
    
    print("\n" + "="*60)
    print("🏆 排序对比表 (按 PS 分数从高到低)")
    print("-" * 60)
    print(f"{'Rank':<5} | {'Logit Score (PS)':<18} | {'Ground Truth (GT)':<18}")
    for idx, row in df_by_ps.iterrows():
        print(f"{idx+1:<5} | {row['logit_score']:<18.2f} | {row['gt_score']:<18}")
    print("="*60)

    # 简单计算逆序对或相关性
    # 如果 PS 排序后的 GT 列表大致是降序的，说明一致性极好
    gt_list_after_ps_sort = df_by_ps["gt_score"].tolist()
    is_mostly_consistent = all(gt_list_after_ps_sort[i] >= gt_list_after_ps_sort[i+1] for i in range(len(gt_list_after_ps_sort)-1))
    
    print(f"\n📈 排序趋势分析: {gt_list_after_ps_sort}")
    print("提示：如果上面的数字大致呈从大到小排列，说明 Logit 评分完美复现了人类排序。")

if __name__ == "__main__":
    main()