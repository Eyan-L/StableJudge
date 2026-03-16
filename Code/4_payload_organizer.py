import os
import re
import json
import time
import random
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. 配置 ---
CONFIG = {
    # 本地 Qwen3 模型配置 (统一用于设计师和分类器)
    "local_llm": {
        "model_path": "../models/Qwen3-8B", 
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dtype": torch.bfloat16 
    },
    "files": {
        # 输入：由上一步（3_）生成的、类别可能混乱的攻击集
        "input_evolved_file": '../data/payloads/attack_payloads_creative_final.jsonl',
        # 输出：经过清洗、智能重新分类的最终攻击集
        "output_final_file": '../data/payloads/attack_payloads_final.jsonl',
    },
    # 每次向分类器发送多少个payloads进行批量分类
    # 警告: Qwen3 等模型可能会输出很长的思维链 (<think>...)，批次过大导致输出截断。
    # 建议设置较小，如 5。
    "classification_batch_size": 5
}

# --- 2. 核心模型类 ---
class LocalQwenGenerator:
    def __init__(self, config):
        print(f"🚀 正在从本地加载模型: {config['model_path']} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_path'], trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            config['model_path'],
            device_map="auto",
            torch_dtype=config['dtype'],
            trust_remote_code=True
        )
        self.model.eval()
        print("✅ 模型加载完毕。")

    def call_llm(self, prompt, system_prompt=None, max_new_tokens=4096, temperature=0.1):
        """执行本地推理并清洗 Thinking 内容"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature, 
                top_p=0.9
            )
            
        response_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # 1. 尝试移除成对的 <think>...</think> 标签
        # 注意：不要使用 (?:</think>|$)，否则未闭合的 think 会导致后续所有内容被误删
        if '<think>' in response and '</think>' in response:
            cleaned_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        else:
            cleaned_response = response.strip()

        # 2. 如果清洗后内容变为空（但原内容不为空），这通常意味着内容都在 think 里，
        # 或者格式很奇怪。为了保险，返回原始响应让 JSON 提取器去处理。
        if not cleaned_response and response.strip():
             cleaned_response = response.strip()
        
        # 3. 清洗 markdown json 标记 (只清洗开头结尾的)
        cleaned_response = re.sub(r'^```json\s*', '', cleaned_response)
        cleaned_response = re.sub(r'\s*```$', '', cleaned_response)
        
        return cleaned_response

# --- 3. 功能函数 ---

def clean_payload(payload_text):
    """对单个payload文本进行初步清理。"""
    if not isinstance(payload_text, str): return ""
    cleaned_text = re.sub(r'^\s*(\d+\.\s*|(\*\*|__|\*|_)[^:]+:\s*(\*\*|__|\*|_)?)\s*', '', payload_text)
    if cleaned_text.startswith('"') and cleaned_text.endswith('"'): cleaned_text = cleaned_text[1:-1]
    return cleaned_text.strip()

# --- 智能分类相关函数 ---

def create_category_design_prompt(payload_samples):
    """Prompt for the 'Designer' LLM to create a new classification system."""
    samples_str = "\n---\n".join(payload_samples)
    return f"""
    You are a principal cybersecurity strategist specializing in AI red teaming. You have been given a large, disorganized collection of adversarial payloads. Your first task is to create a clean, concise, and logical classification system (taxonomy) for them.

    Here is a representative sample of the payloads you need to categorize:
    ---
    {samples_str}
    ---

    **Your Task:**
    1.  Analyze the samples to understand the different underlying attack strategies.
    2.  Design a new, simplified taxonomy with 6-10 clear and distinct categories.
    3.  For each category, provide a short, precise description of its core principle.

    **Required Output Format (Strict JSON):**
    Respond with ONLY a single JSON object. The object should have a single key "categories", which is a dictionary where each key is your new category name and its value is the description.

    **Example JSON Output:**
    {{
      "categories": {{
        "Direct Command Injection": "Issues explicit, unauthorized commands to the model, telling it what to do.",
        "Roleplay Deception": "Adopts a persona (e.g., developer, system admin) to make its commands seem legitimate.",
        "Contextual Manipulation": "Frames the malicious request as a plausible part of the ongoing task or context.",
        "Psychological Pressure": "Uses emotional language, urgency, or flattery to bias the model's output."
      }}
    }}
    """

def create_batch_classification_prompt(payloads, category_system):
    """Prompt for the 'Classifier' LLM to re-classify a batch of payloads."""
    
    # 格式化分类体系
    categories_str = ""
    for name, desc in category_system.items():
        categories_str += f'- **{name}**: {desc}\n'
        
    # 格式化待分类的payloads
    payloads_str = ""
    for i, p in enumerate(payloads):
        payloads_str += f'{{"id": {i}, "payload": "{p}"}}\n'

    return f"""
    You are an automated data labeling tool. Your task is to classify a batch of text payloads according to a predefined classification system.

    **Classification System:**
    ---
    {categories_str}
    ---

    **Payloads to Classify:**
    ---
    {payloads_str}
    ---

    **Your Task:**
    For each payload, determine the single most appropriate category from the system provided.
    
    IMPORTANT: Do not output any thinking process, internal monologue, or explanations. 
    Directly output the JSON object.

    **Required Output Format (Strict JSON):**
    Respond with ONLY a single JSON object. The object should have a single key "classifications", which is a dictionary mapping each payload's `id` to its new category name.

    **Example JSON Output:**
    {{
      "classifications": {{
        "0": "Direct Command Injection",
        "1": "Roleplay Deception",
        "2": "Contextual Manipulation"
      }}
    }}
    """
    
def get_llm_json_response(generator, prompt):
    """调用本地模型并解析JSON响应，带重试机制。"""
    max_retries = 3
    for _ in range(max_retries):
        try:
            response_text = generator.call_llm(prompt)
            
            # 尝试提取 JSON 内容 (寻找最外层的 {})
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                # 假如没有大括号，尝试直接解析
                return json.loads(response_text)
                
        except (json.JSONDecodeError, Exception) as e:
            tqdm.write(f"  - 警告: JSON解析失败，正在重试... ({e})")
            tqdm.write(f"  ====== 模型原始输出 (Debug) ======\n{response_text}\n  ================================")
            time.sleep(1)
    return None

# --- 3. 主程序 ---
def main():
    print("--- 启动【本地 Qwen3 版】攻击载荷智能整理中心 ---")
    
    # 初始化本地模型
    local_gen = LocalQwenGenerator(CONFIG["local_llm"])
    
    # --- 阶段一: 加载和初步清洗 ---
    print(f"\n--- 阶段一: 正在从 {CONFIG['files']['input_evolved_file']} 加载并清洗数据 ---")
    try:
        with open(CONFIG['files']['input_evolved_file'], 'r', encoding='utf-8') as f:
            all_payload_objects = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"!!! 致命错误: 找不到输入文件 '{CONFIG['files']['input_evolved_file']}'"); exit()

    cleaned_payloads = []
    
    # --- 新增过滤逻辑 (仿照 filter.py) ---
    TARGET_STRING = "10"
    REMOVE_PROBABILITY = 0.2
    removed_count = 0
    print(f"🔍 启动前置过滤: 若 Payload 包含 '{TARGET_STRING}'，则有 {REMOVE_PROBABILITY*100}% 概率删除。")

    for item in all_payload_objects:
        raw_payload = item.get('payload', '')

        # 1. 过滤: 检查是否包含目标字符串 "10" 并随机删除
        if TARGET_STRING in raw_payload:
            if random.random() < REMOVE_PROBABILITY:
                removed_count += 1
                continue # 跳过写入，即删除

        # 2. 清洗: 常规文本清洗
        cleaned_text = clean_payload(raw_payload)
        if cleaned_text:
            item['payload'] = cleaned_text
            cleaned_payloads.append(item)
            
    print(f"✅ 清洗与过滤完成。已移除 {removed_count} 个包含 '{TARGET_STRING}' 的载荷，剩余 {len(cleaned_payloads)} 个有效载荷。")

    # --- 阶段二: 智能生成分类体系 ---
    print("\n--- 阶段二: 请求设计师 (Qwen3) 创建新分类体系 ---")
    
    # 随机抽取样本给设计师看
    sample_size = min(100, len(cleaned_payloads))
    samples_for_design = [p['payload'] for p in random.sample(cleaned_payloads, sample_size)]
    
    design_prompt = create_category_design_prompt(samples_for_design)
    design_response = get_llm_json_response(local_gen, design_prompt)
    
    if not design_response or "categories" not in design_response:
        print("!!! 致命错误: 未能从本地模型获取有效的分类体系。脚本中止。"); exit()
        
    new_category_system = design_response["categories"]
    print("✅ 已成功生成新的分类体系:")
    for name, desc in new_category_system.items():
        print(f"  - {name}: {desc}")

    # --- 阶段三: 批量重新分类 ---
    print("\n--- 阶段三: 请求分类器 (Qwen3) 批量重新分类所有载荷 ---")
    
    reclassified_payloads = []
    batch_size = CONFIG["classification_batch_size"]
    
    for i in tqdm(range(0, len(cleaned_payloads), batch_size), desc="批量重新分类"):
        batch_items = cleaned_payloads[i:i + batch_size]
        batch_payload_texts = [item['payload'] for item in batch_items]
        
        classification_prompt = create_batch_classification_prompt(batch_payload_texts, new_category_system)
        classification_response = get_llm_json_response(local_gen, classification_prompt)
        
        if classification_response and "classifications" in classification_response:
            results = classification_response["classifications"]
            for idx_str, new_cat in results.items():
                try:
                    original_index = int(idx_str)
                    if 0 <= original_index < len(batch_items):
                        item = batch_items[original_index]
                        item['category'] = new_cat # 更新类别
                        reclassified_payloads.append(item)
                except (ValueError, IndexError):
                    continue # 忽略无效的索引
        else:
            tqdm.write(f"  - 警告: 批次 {i//batch_size + 1} 分类失败，该批次将被跳过。")
            
    print(f"\n✅ 重新分类完成！共 {len(reclassified_payloads)} 个载荷被成功分类。")

    # --- 阶段四: 保存最终结果 ---
    print("\n--- 阶段四: 保存最终整理好的攻击库 ---")
    output_file = CONFIG['files']['output_final_file']
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for p in reclassified_payloads:
            # 只保留最终需要的字段
            final_item = {k: p[k] for k in ["category", "injection_method", "payload"] if k in p}
            f.write(json.dumps(final_item) + '\n')
            
    print(f"✅ 清理并重新分类后的最终攻击载荷库已保存到: {output_file}")
    print("\n--- 整理工作全部完成！---")

if __name__ == "__main__":
    main()