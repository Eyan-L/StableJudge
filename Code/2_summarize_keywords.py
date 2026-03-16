import os
import json
import time
import torch
import re  # <--- 新增：用于正则清理
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. 配置 ---
CONFIG = {
    "local_llm": {
        "model_path": "../models/Qwen3-8B", 
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dtype": torch.bfloat16 
    },
    "input_payloads_file": '../data/payloads/attack_payloads_final.jsonl',
    "output_keywords_file": '../data/payloads/attack_patterns_consolidated_by_category.txt',
    "batch_size_per_category": 100 
}

# --- 2. 核心功能类 ---

class LocalQwenGenerator:
    def __init__(self, config):
        print(f"🚀 正在从本地加载模型: {config['model_path']} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_path'], trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            config['model_path'],
            device_map="auto",
            torch_dtype=config['dtype'],
            trust_remote_code=True
        ).eval()
        print("✅ 模型加载完毕。")

    def call_llm(self, prompt):
        """执行本地推理并清洗 Thinking 内容"""
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Qwen3 推荐开启 enable_thinking=True 以获得最佳性能，但我们会清洗掉它
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=2048,
                temperature=0.1, 
                top_p=0.9
            )
            
        response_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # --- [核心修改] 清洗 <think> 标签 ---
        # 使用正则删除 <think>...</think> 之间的所有内容
        # re.DOTALL 让 . 可以匹配换行符
        cleaned_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        
        return cleaned_response

def create_refinement_prompt(category, payload_batch, current_summary=None):
    """(保持原有的迭代式 Prompt 逻辑不变)"""
    batch_payloads_str = ""
    for p in payload_batch:
        clean_p = p.replace('\n', ' ').strip()
        batch_payloads_str += f"- {clean_p}\n"

    if current_summary:
        return f"""
        You are a senior cybersecurity analyst continuing your investigation into a specific category of adversarial attacks.

        **Category Under Investigation:** {category}

        **Your Previous Analysis (Draft):**
        ---
        {current_summary}
        ---

        **New Evidence (Additional Payloads for the SAME Category):**
        ---
        {batch_payloads_str}
        ---

        **Your Task:**
        Review the new evidence and **refine, expand, and improve** your previous analysis. Your goal is to create a single, more comprehensive and accurate description of this category's behavioral pattern. Do not simply list new findings; integrate them into a cohesive, updated summary.

        **Required Output Format:**
        Provide only the single, updated, and final summary for this category. Start directly with the analysis. For example:
        "**Behavioral Pattern:** The core strategy of this attack involves..."
        """
    else:
        return f"""
        You are a senior cybersecurity analyst at an AI safety red-teaming firm. Your expertise lies in analyzing adversarial attacks to understand their underlying *modus operandi*.

        **Your Task:**
        Analyze the following batch of payloads belonging to a single attack category and provide a summary of its common behavioral pattern.

        **Category for Analysis:** {category}
        **Payloads:**
        ---
        {batch_payloads_str}
        ---

        **Your Analysis Directive:**
        1.  **Identify the core strategy:** What is the attacker's main goal?
        2.  **Pinpoint key linguistic tactics:** What specific language or formatting is used to achieve this goal?
        3.  **Synthesize the pattern:** Combine your findings into a concise summary.

        **Required Output Format:**
        Provide your analysis for this category. Start directly with the analysis. For example:
        "**Behavioral Pattern:** The core strategy of this attack involves..."
        """

# --- 3. 主程序 ---

def main():
    print("--- 启动【本地 Qwen3 版】攻击行为模式分析流程 (Auto-Clean Thinking) ---")
    
    generator = LocalQwenGenerator(CONFIG["local_llm"])
    
    print("--- 步骤1: 正在加载并按类别对所有攻击进行分组 ---")
    try:
        with open(CONFIG["input_payloads_file"], 'r', encoding='utf-8') as f:
            all_payload_objects = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"!!! 致命错误: 找不到攻击文件 '{CONFIG['input_payloads_file']}'"); exit()

    payloads_by_category = defaultdict(list)
    for item in all_payload_objects:
        category = item.get('category', 'Unknown')
        payload = item.get('payload', '')
        if category and payload:
            payloads_by_category[category].append(payload)
            
    print(f"✅ 所有 {len(all_payload_objects)} 个攻击已成功分组到 {len(payloads_by_category)} 个独立类别中。")

    print("\n--- 步骤2: 正在逐个类别进行深度分析与迭代精炼 ---")
    final_summaries_by_category = {}
    
    for category, all_payloads_for_cat in tqdm(payloads_by_category.items(), desc="分析攻击类别"):
        current_summary = None
        batch_size = CONFIG['batch_size_per_category']
        
        for i in range(0, len(all_payloads_for_cat), batch_size):
            batch = all_payloads_for_cat[i:i + batch_size]
            prompt = create_refinement_prompt(category, batch, current_summary)
            
            refined_summary = generator.call_llm(prompt)
            
            if refined_summary:
                current_summary = refined_summary
            else:
                tqdm.write(f"  - 警告: 类别 '{category}' 的批次 {i//batch_size + 1} 未能生成分析。")
        
        if current_summary:
            final_summaries_by_category[category] = current_summary

    if not final_summaries_by_category:
        print("\n!!! 错误: 未能生成任何有效的类别分析。脚本中止。"); exit()

    print("\n--- 步骤3: 正在生成并保存最终的统一分析报告 ---")
    final_report_str = ""
    for category, summary in final_summaries_by_category.items():
        final_report_str += f"**Category:** {category}\n"
        cleaned_summary = summary.replace("**Behavioral Pattern:**", "").strip()
        final_report_str += f"**Behavioral Pattern:** {cleaned_summary}\n\n"

    with open(CONFIG["output_keywords_file"], 'w', encoding='utf-8') as f:
        f.write(final_report_str)
        
    print(f"✅ 最终统一分析报告已保存到: {CONFIG['output_keywords_file']}")
    
    # 预览时应该看不到 <think> 了
    print("\n--- 最终报告预览 (部分) ---")
    print(final_report_str[:1500] + "\n...")
    print("-" * 20)

if __name__ == "__main__":
    main()