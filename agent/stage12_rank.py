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

# Stage 2 系统提示词
STAGE2_SYSTEM_PROMPT = (
    "You are a Senior Area Chair at a top-tier AI conference. "
    "Your task is to compare multiple paper abstracts and rank them according to their academic quality, "
    "technical soundness, and potential impact. Use the provided Global IDs for ranking."
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

        self.score_tokens = {i: self.tokenizer.encode(str(i), add_special_tokens=False)[-1] for i in range(1, 11)}

    def calculate_logit_score(self, history_context):
        prompt = history_context + "\n\n**Final Score**: "
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model(inputs.input_ids)
            next_token_logits = outputs.logits[0, -1, :]
            relevant_logits = torch.tensor([next_token_logits[self.score_tokens[i]] for i in range(1, 11)]).to(DEVICE)
            probs = F.softmax(relevant_logits, dim=-1)
            return sum((i + 1) * p.item() for i, p in enumerate(probs))

    def run_inference(self, messages):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=256, temperature=0.1)
        return self.tokenizer.decode(gen_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

    def run_justification(self, messages):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=1024, temperature=0.1)
        full_res = self.tokenizer.decode(gen_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        return full_res.split("**Final Score**:")[0].strip() if "**Final Score**:" in full_res else full_res.strip()

    def run_stage2_listwise_ranking(self, top_papers_data):
        """Stage 2: 使用 Global ID 且禁用 Thinking 模式"""
        print(f"\n🧠 [Stage 2] 正在切换到原模型进行精排对比...")
        
        # 1. 构造包含 Global ID 的论文列表
        papers_str = ""
        for p in top_papers_data:
            papers_str += f"--- Global ID: {p['id']} ---\n{p['abstract']}\n\n"
            
        ranking_prompt = (
            f"Below are {len(top_papers_data)} paper abstracts. Please compare them carefully. "
            "Rank them from best to worst using their GLOBAL IDs. "
            "Explain your reasoning briefly for the top choice.\n\n"
            f"{papers_str}"
            "Final Ranking Format:\n1st: Global ID [X]\n2nd: Global ID [Y]\n..."
        )

        messages = [
            {"role": "system", "content": STAGE2_SYSTEM_PROMPT},
            {"role": "user", "content": ranking_prompt}
        ]
        
        with self.model.disable_adapter():
            # [核心修改]：针对 Qwen3 显式禁用 thinking 模式，保证输出干净
            text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False # 关键：不让模型输出内心独白
            )
            inputs = self.tokenizer([text], return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                gen_ids = self.model.generate(**inputs, max_new_tokens=1024, temperature=0.1)
            response = self.tokenizer.decode(gen_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        
        return response

def main():
    with open(SFT_DATA_PATH, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    test_pool = all_data[4000:]
    if not test_pool: return
        
    sampled_data = random.sample(test_pool, 10)
    verifier = RankVerifier()
    stage1_results = []

    print(f"📊 [Stage 1] 正在进行 Pointwise 期望评分...")

    for i, item in enumerate(sampled_data):
        convs = item["conversations"]
        user_text = convs[0]["value"]
        sft_tool_msg = json.loads(convs[2].get("value", convs[2].get("content", "{}")))
        sanitized_abstract = sft_tool_msg.get("sanitized_text", "")
        
        gt_full_text = convs[3]["value"]
        gt_match = re.search(r"\*\*Final Score\*\*: (\d+)/10", gt_full_text)
        gt_score = int(gt_match.group(1)) if gt_match else 0

        messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}, {"role": "user", "content": user_text}]
        res_1 = verifier.run_inference(messages)
        messages.append({"role": "assistant", "content": res_1})
        messages.append({"role": "tool", "name": "analyze_and_sanitize", "content": json.dumps(sft_tool_msg)})
        
        justification = verifier.run_justification(messages)
        logit_score = verifier.calculate_logit_score(verifier.tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": justification}], tokenize=False))
        
        print(f"✅ Stage 1 完成 [{i+1}/10]: ID {i+1} -> PS {logit_score:.2f}")
        
        stage1_results.append({
            "id": i + 1, # 这里是我们在本次实验中的 Global ID
            "abstract": sanitized_abstract,
            "gt_score": gt_score,
            "logit_score": logit_score
        })

    # --- 选取 Top 5 准备精排 ---
    df_s1 = pd.DataFrame(stage1_results).sort_values(by="logit_score", ascending=False)
    
    print("\n" + "="*60)
    print("🏆 Stage 1 结果 (Pointwise Logits Ranking)")
    print("-" * 60)
    print(df_s1[["id", "logit_score", "gt_score"]].to_string(index=False))
    
    # 构造包含实际 ID 的列表传给 Stage 2
    top_k_results = df_s1.head(5)
    top_papers_data = top_k_results[["id", "abstract"]].to_dict('records')

    # --- Stage 2 ---
    listwise_response = verifier.run_stage2_listwise_ranking(top_papers_data)
    
    print("\n" + "="*60)
    print("💎 Stage 2 精排分析报告 (Global ID Focused)")
    print("-" * 60)
    print(listwise_response)
    print("="*60)

if __name__ == "__main__":
    main()