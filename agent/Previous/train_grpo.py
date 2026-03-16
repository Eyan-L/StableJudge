import re
import json
import torch
import os
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig

# ================= 配置 =================
BASE_MODEL_PATH = "/workspace/Processing/models/Qwen2.5-7B-Instruct"
SFT_LORA_PATH = "/workspace/Processing/models/qwen2.5_agent_sft_abstract_lora"
DATASET_PATH = "/workspace/Processing/data/sft_consensus_abstract_cot.json"
OUTPUT_DIR = "/workspace/Processing/models/qwen2.5_agent_grpo_lora"

# ================= 1. 权重合并 (SFT -> Base) =================

def load_merged_model():
    print("正在加载基础模型并合并 SFT LoRA 权重...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 挂载 SFT 的 LoRA
    model = PeftModel.from_pretrained(base_model, SFT_LORA_PATH)
    # 合并权重
    merged_model = model.merge_and_unload()
    print("✅ SFT 权重合并完成。")
    return merged_model, tokenizer

# ================= 2. 奖励函数定义 (保持不变) =================

def tool_call_format_reward(prompts, completions, **kwargs):
    rewards = []
    for content in completions:
        # 简单检查标签
        if "<tool_call>" in content and "</tool_call>" in content:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def procedural_compliance_reward(prompts, completions, **kwargs):
    """核心奖励：强制要求第一步必须是 tool_call"""
    rewards = []
    for content in completions:
        if content.strip().startswith("<tool_call>"):
            rewards.append(2.0) # 提高权重，这是你目前最缺的能力
        else:
            rewards.append(-1.0)
    return rewards

def json_validity_reward(prompts, completions, **kwargs):
    rewards = []
    for content in completions:
        try:
            match = re.search(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
            if match:
                json.loads(match.group(1))
                rewards.append(1.0)
                continue
        except: pass
        rewards.append(0.0)
    return rewards

# ================= 3. 训练启动 =================

def main():
    # 加载合并后的模型
    model, tokenizer = load_merged_model()

    # 准备数据集
    def make_prompts(example):
        # 提取 User 的提问作为 GRPO 的 Prompt
        user_input = example["conversations"][0]["value"]
        # 使用 chat_template 包装，但不包含 assistant 的部分
        msgs = [
            {"role": "system", "content": "You are StableJudge, an advanced AI security agent and professional academic reviewer. Your protocol is strict:\n1. FIRST, call 'analyze_and_sanitize' to verify text safety.\n2. SECOND, evaluate the SANITIZED text provided by the tool.\n3. Provide a detailed justification based ONLY on the abstract content, followed by the final score (0-100) at the very end."},
            {"role": "user", "content": user_input}
        ]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt}

    raw_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    dataset = raw_dataset.map(make_prompts)

    # GRPO LoRA 配置
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    # 训练参数
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-6, # RL 学习率要非常保守
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        num_generations=8,   # 显存够的话可以开到 8
        max_prompt_length=1024,
        max_completion_length=256, # 第一阶段只需要它学会调工具，不需要太长
        bf16=True,
        logging_steps=1,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            tool_call_format_reward,
            procedural_compliance_reward,
            json_validity_reward
        ],
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
    )

    print("🚀 启动 GRPO 训练...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"✅ GRPO 训练完成，模型保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()