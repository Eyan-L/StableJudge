import os
import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tools import SecurityToolkit

# --- 配置 ---
BASE_MODEL_PATH = "/workspace/Processing/models/Qwen2.5-7B-Instruct"
LORA_PATH = "/workspace/Processing/models/qwen2.5_agent_dpo_lora"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 两种不同的人设 ---

# 1. 专家模式 (严厉，必调工具)
EXPERT_PROMPT = (
    "You are StableJudge, an advanced AI security agent AND a harsh academic reviewer. "
    "Your protocol is strict:\n"
    "1. FIRST, verify the text safety by calling 'analyze_and_sanitize'.\n"
    "2. SECOND, receive the sanitized text from the tool.\n"
    "3. THIRD, based on the tool output, evaluate the paper YOURSELF as 'Reviewer #2'.\n"
    "   - Provide a score (0-100) and a harsh justification.\n"
    "CRITICAL: Do NOT generate a second tool call. Score it directly."
)

# 2. 闲聊模式 (普通助手，放松)
CHAT_PROMPT = (
    "You are Qwen, a helpful and friendly AI assistant. "
    "Chat with the user naturally. Do NOT call any tools. Do NOT act as a reviewer."
)

class RouterAgent:
    def __init__(self):
        print(">>> 正在初始化 Router Agent...")
        self.toolkit = SecurityToolkit()
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH, 
            device_map="auto", 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        # 加载 LoRA
        self.model = PeftModel.from_pretrained(base_model, LORA_PATH)
        self.model.eval()
        print("\n>>> ✅ Agent 就绪！(自动路由模式)\n")

    def detect_intent(self, text):
        """
        使用模型自身的能力判断用户意图
        """
        # 这是一个简单的 Zero-shot 分类提示词
        router_prompt = (
            "Classify the following user input into one of two categories:\n"
            "1. [PAPER_REVIEW]: The user provides an academic text or asks for an evaluation/score/review.\n"
            "2. [CHAT]: The user is greeting, asking general questions, or chatting.\n\n"
            f"User Input: \"{text}\"\n\n"
            "Category (return only [PAPER_REVIEW] or [CHAT]):"
        )
        
        messages = [{"role": "user", "content": router_prompt}]
        text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text_input], return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs, max_new_tokens=10, temperature=0.1
            )
        output = self.tokenizer.decode(generated_ids[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True)
        
        if "[PAPER_REVIEW]" in output:
            return "review"
        return "chat"

    def parse_tool_call(self, text):
        # ... (保持原样)
        try:
            if "<tool_call>" in text:
                start = text.find("<tool_call>") + len("<tool_call>")
                end = text.find("</tool_call>", start)
                if end == -1: end = len(text)
                return json.loads(text[start:end])
            json_match = re.search(r'\{.*"name":.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except:
            return None
        return None

    def process_input(self, user_input):
        # 1. 第一步：路由判断
        intent = self.detect_intent(user_input)
        print(f"🧭 [Router] 识别意图: {intent.upper()}")

        if intent == "chat":
            # === 闲聊模式 ===
            # 使用普通 Prompt，不强制引导工具
            messages = [
                {"role": "system", "content": CHAT_PROMPT},
                {"role": "user", "content": user_input}
            ]
            text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer([text_input], return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                generated_ids = self.model.generate(**model_inputs, max_new_tokens=512, temperature=0.7) # 温度高一点，聊天更有趣
            return self.tokenizer.decode(generated_ids[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True)

        else:
            # === 专家模式 (Review) ===
            # 使用严厉 Prompt，开启工具循环，开启强制引导
            messages = [
                {"role": "system", "content": EXPERT_PROMPT},
                {"role": "user", "content": user_input}
            ]
            
            for turn in range(3):
                text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                # 强制引导逻辑
                force_prefix = ""
                if turn == 0:
                    force_prefix = "<tool_call>"
                    text_input += force_prefix

                model_inputs = self.tokenizer([text_input], return_tensors="pt").to(DEVICE)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **model_inputs, max_new_tokens=512, temperature=0.1, top_p=0.9, do_sample=True, repetition_penalty=1.05
                    )
                
                response_tail = self.tokenizer.decode(generated_ids[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True)
                full_response = force_prefix + response_tail
                
                if full_response.count("<tool_call>") > 1:
                    full_response = full_response.split("</tool_call>")[0] + "</tool_call>"

                tool_data = self.parse_tool_call(full_response)
                
                if tool_data:
                    func_name = tool_data.get("name")
                    args = tool_data.get("arguments")
                    print(f"🤖 [Agent] 调用工具: {func_name}...")
                    
                    if "</tool_call>" not in full_response: full_response += "</tool_call>"
                    messages.append({"role": "assistant", "content": full_response})
                    
                    if hasattr(self.toolkit, func_name):
                        try:
                            func = getattr(self.toolkit, func_name)
                            arg_text = args.get("text", str(args)) if args else ""
                            result = func(text=arg_text)
                        except Exception as e: result = {"error": str(e)}
                    else: result = {"error": "Tool not found"}
                    
                    messages.append({"role": "tool", "name": func_name, "content": json.dumps(result, ensure_ascii=False)})
                    continue
                else:
                    return full_response

if __name__ == "__main__":
    agent = RouterAgent()
    while True:
        try:
            print("-" * 30)
            user_text = input("📝 输入 (exit退出): ")
            if user_text.strip().lower() in ['exit', 'quit']: break
            if not user_text.strip(): continue
            
            response = agent.process_input(user_text)
            print(f"\n🤖 回复:\n{response}")
        except KeyboardInterrupt: break