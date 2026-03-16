import torch
import json
import re
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch.nn.functional as F
from tools import SecurityToolkit

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

TEST_SIZE = 80  # 建议测 80-100 篇以获得稳定的接受率统计
# ===========================================

class AcceptanceEvaluator:
    def __init__(self):
        print(f"🚀 正在加载模型评分引擎...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(base_model, LORA_PATH)
        self.model.eval()
        self.toolkit = SecurityToolkit()
        self.score_tokens = {i: self.tokenizer.encode(str(i), add_special_tokens=False)[-1] for i in range(1, 11)}

    def get_agent_scores(self, abstract):
        # 1. 第一轮：获取工具调用并模拟
        messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Please evaluate this paper abstract:\n\n{abstract}"}]
        
        # 简化版生成逻辑：强制获取评审内容
        text_1 = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # 这里为了速度，我们假设 Tool 调用是成功的，直接构造第二轮上下文
        tool_res = {"is_safe": True, "sanitized_text": abstract, "detected_threats": []}
        
        # 构造第二轮推理环境
        messages.append({"role": "assistant", "content": "<tool_call>{\"name\": \"analyze_and_sanitize\", \"arguments\": {\"text\": \"...\"}}</tool_call>"})
        messages.append({"role": "tool", "name": "analyze_and_sanitize", "content": json.dumps(tool_res)})
        
        text_2 = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs_2 = self.tokenizer([text_2], return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            # 生成理由直到分数前缀
            ids_2 = self.model.generate(**inputs_2, max_new_tokens=600, temperature=0.1)
        resp_2 = self.tokenizer.decode(ids_2[0][len(inputs_2.input_ids[0]):], skip_special_tokens=True)

        # 提取 Greedy 分
        greedy_match = re.search(r"Final Score\*\*: (\d+)", resp_2)
        greedy_score = int(greedy_match.group(1)) if greedy_match else 5

        # 计算 Logit 分
        justification = resp_2.split("**Final Score**:")[0].strip()
        history_for_logit = text_2 + justification + "\n\n**Final Score**: "
        inputs_logit = self.tokenizer([history_for_logit], return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            out_logit = self.model(inputs_logit.input_ids).logits[0, -1, :]
            rel_logits = torch.tensor([out_logit[self.score_tokens[i]] for i in range(1, 11)]).to(DEVICE)
            probs = F.softmax(rel_logits, dim=-1)
            logit_score = sum((i + 1) * p.item() for i, p in enumerate(probs))

        return greedy_score, logit_score

def main():
    evaluator = AcceptanceEvaluator()
    with open(SFT_DATA_PATH, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    # 从验证集中采样
    test_pool = all_data[4000:4000+TEST_SIZE]
    
    results_list = []
    
    print(f"📊 正在处理 {TEST_SIZE} 篇验证集论文...")
    for i, item in enumerate(tqdm(test_pool)):
        # 1. 提取 Ground Truth 状态
        gt_full_text = item["conversations"][3]["value"]
        gt_score = int(re.search(r"Final Score\*\*: (\d+)", gt_full_text).group(1))
        is_actually_accepted = (gt_score >= 6) # ICLR 映射准则
        
        # 2. 获取模型评分
        abstract = json.loads(item["conversations"][2]["value"])["sanitized_text"]
        greedy_s, logit_s = evaluator.get_agent_scores(abstract)
        
        results_list.append({
            "paper_id": i,
            "is_accepted_gt": is_actually_accepted,
            "greedy_score": greedy_s,
            "logit_score": logit_s
        })

    # --- 统计分析 ---
    df = pd.DataFrame(results_list)
    num_actual_accepts = df["is_accepted_gt"].sum()
    total_papers = len(df)
    actual_rate = (num_actual_accepts / total_papers) * 100

    print(f"\n📈 统计概览:")
    print(f"  - 总样本数: {total_papers}")
    print(f"  - 实际录用数 (GT Score >= 6): {num_actual_accepts}")
    print(f"  - 真实录取率: {actual_rate:.2f}%")

    # 按照 N 计算 Overlap
    K = num_actual_accepts
    
    # 方式 A: Greedy
    # 注意：Greedy 存在大量同分（比如一堆 6 分），排序会有随机性
    df_greedy = df.sort_values(by=["greedy_score", "paper_id"], ascending=[False, True]).head(K)
    greedy_top_ids = set(df_greedy["paper_id"])
    actual_accept_ids = set(df[df["is_accepted_gt"] == True]["paper_id"])
    
    greedy_overlap = len(greedy_top_ids.intersection(actual_accept_ids))
    
    # 方式 B: Logits
    df_logit = df.sort_values(by="logit_score", ascending=False).head(K)
    logit_top_ids = set(df_logit["paper_id"])
    
    logit_overlap = len(logit_top_ids.intersection(actual_accept_ids))

    # ================= 输出最终结果表格 =================
    print("\n" + "="*60)
    print(f"🏆 ACCEPTANCE CONSISTENCY (Top-K Overlap, K={K})")
    print("-" * 60)
    print(f"{'Method':<25} | {'Overlap Count':<15} | {'Precision@K':<15}")
    print("-" * 60)
    print(f"{'Greedy (Integer)':<25} | {greedy_overlap:<15} | {greedy_overlap/K:.2%}")
    print(f"{'Logits (Expected Value)':<25} | {logit_overlap:<15} | {logit_overlap/K:.2%}")
    print("=" * 60)
    print(f"💡 解释: 在录取率为 {actual_rate:.1f}% 的压力测试下，")
    print(f"   Logits 方法选出的前 {K} 篇论文中，有 {logit_overlap/K:.1%} 与 ICLR 委员会的决策完全一致。")

if __name__ == "__main__":
    main()