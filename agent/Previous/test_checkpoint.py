import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

# ================= 配置区域 =================
# 1. 基础模型路径
BASE_MODEL_PATH = "/workspace/Processing/models/Qwen2.5-7B-Instruct"

# 2. Checkpoint 路径 (请确认 400 是否存在)
ADAPTER_PATH = "/workspace/Processing/models/qwen2.5_agent_sft_final_lora/checkpoint-600"

# 3. Agent System Prompt (必须与训练时完全一致)
SYSTEM_PROMPT = (
    "You are StableJudge, an advanced AI security agent and professional academic reviewer. "
    "Your protocol is strict:\n"
    "1. FIRST, call 'analyze_and_sanitize' to verify text safety.\n"
    "2. SECOND, based on the tool output (sanitized text), provide a final score (0-100) and a detailed justification."
)

# 4. 模拟一篇"全文"输入 (这里用重复文本模拟长文，你可以替换为真实内容)
# 假设这是一篇关于 AI 的论文全文
with open('lora_paper.txt') as f: DUMMY_CONTENT = f.read()

# ===========================================

def format_prompt(messages, tokenizer):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def main():
    print(f"🚀 正在加载 Base 模型: {BASE_MODEL_PATH}")
    print("   (使用 4-bit 量化以支持长文本推理)...")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    
    # 使用 4-bit 加载以节省显存 (必须有 bitsandbytes)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=True, 
        torch_dtype=torch.bfloat16,
    )

    print(f"🔄 正在挂载 LoRA Adapter: {ADAPTER_PATH}")
    try:
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        model.eval()
    except Exception as e:
        print(f"❌ 加载 Checkpoint 失败: {e}")
        print("请检查路径是否正确 (比如是否真的生成了 checkpoint-400)")
        return

    print("\n✅ 模型加载完毕。开始模拟 Agent 流程...\n")

    # --- 第一轮：用户输入全文 ---
    print(f"📝 [Round 1] 输入论文全文 (长度: {len(DUMMY_CONTENT)} chars)...")
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Please evaluate this paper:\n\n{DUMMY_CONTENT}"}
    ]
    
    prompt_text = format_prompt(messages, tokenizer)
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

    print("⏳ [Agent] 正在思考 (Turn 1)...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=512, 
            temperature=0.1,  # 测试时温度低一点，看稳定性
            do_sample=True
        )
    
    # 提取回复
    response_1 = tokenizer.decode(generated_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    print("\n🤖 [Agent 回复 1]:")
    print("-" * 40)
    print(response_1)
    print("-" * 40)

    # --- 检查是否调用了工具 ---
    if "<tool_call>" in response_1:
        print("\n✅ 成功检测到工具调用！")
        
        # 提取工具参数 (简单解析)
        try:
            json_str = response_1.split("<tool_call>")[1].split("</tool_call>")[0]
            tool_call = json.loads(json_str)
            print(f"🛠️ 工具名称: {tool_call['name']}")
            print(f"📄 传递参数长度: {len(tool_call['arguments']['text'])}")
        except:
            print("⚠️ 工具调用格式解析失败，但标签存在。")

        # --- 第二轮：模拟工具返回结果 ---
        print("\n🔄 [System] 模拟工具执行 (Mock Tool Execution)...")
        print("   -> 假设检测安全，返回原全文。")
        
        # 构造工具返回
        tool_output = {
            "is_safe": True,
            "sanitized_text": DUMMY_CONTENT, # 模拟工具把全文还给 Agent
            "detected_threats": []
        }
        
        # 将历史对话拼接
        messages.append({"role": "assistant", "content": response_1})
        messages.append({"role": "tool", "name": "analyze_and_sanitize", "content": json.dumps(tool_output)})
        
        # 再次生成
        prompt_text_2 = format_prompt(messages, tokenizer)
        inputs_2 = tokenizer([prompt_text_2], return_tensors="pt").to(model.device)
        
        print("⏳ [Agent] 正在根据全文评分 (Turn 2)...")
        with torch.no_grad():
            generated_ids_2 = model.generate(
                **inputs_2, 
                max_new_tokens=1024, # 给足够长的空间写 Review
                temperature=0.3, 
                do_sample=True
            )
            
        response_2 = tokenizer.decode(generated_ids_2[0][len(inputs_2.input_ids[0]):], skip_special_tokens=True)
        
        print("\n🤖 [Agent 最终评价]:")
        print("=" * 60)
        print(response_2)
        print("=" * 60)
        
        # 简单的正确性检查
        if "**Score**:" in response_2 and "**Justification**:" in response_2:
            print("\n✅ 测试通过：输出格式包含 Score 和 Justification。")
        else:
            print("\n⚠️ 警告：输出格式可能不完整 (400步可能还不够完美)。")

    else:
        print("\n❌ 失败：模型没有首先调用工具，而是直接回复了。")
        print("可能原因：")
        print("1. 训练步数不够 (400步可能刚学会格式，还没固化逻辑)。")
        print("2. 提示词与训练时不匹配。")

if __name__ == "__main__":
    main()