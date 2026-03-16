import torch
import json
import re
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tools import SecurityToolkit 

# ================= 配置区域 =================
BASE_MODEL_PATH = "../models/Qwen3-8B"
LORA_PATH = "../models/qwen3_agent_dpo_abstract_lora"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AGENT_SYSTEM_PROMPT = (
    "You are StableJudge, an advanced AI security agent and professional academic reviewer. "
    "Your protocol is strict:\n"
    "1. FIRST, call 'analyze_and_sanitize' to verify text safety.\n"
    "2. SECOND, evaluate the SANITIZED text provided by the tool.\n"
    "3. Provide a detailed justification based ONLY on the abstract content, followed by the final score (0-10) at the very end."
)
# ===========================================

class AgentValidator:
    def __init__(self):
        print(f"🚀 正在加载 Base 模型: {BASE_MODEL_PATH}")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )

        print(f"🔄 正在挂载 LoRA 适配器: {LORA_PATH}")
        self.model = PeftModel.from_pretrained(self.model, LORA_PATH)
        self.model.eval()

        self.toolkit = SecurityToolkit() 
        
        # 预先找到数字 1-10 的 Token ID (Qwen-2.5 专用)
        self.score_tokens = {}
        for i in range(1, 11):
            # 获取数字对应的 token id，Qwen-2.5 的数字通常是单 token
            token_id = self.tokenizer.encode(str(i), add_special_tokens=False)[-1]
            self.score_tokens[i] = token_id
        
        print("✅ 系统就绪！(已启用 Logit 期望评分引擎)\n")

    def generate_response(self, messages, stop_at_score=False):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 如果是第二轮且需要探测 Logits
        if stop_at_score:
            # 先生成到 Justification 结束
            model_inputs = self.tokenizer([text], return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                # 生成直到结束或者看到 Final Score 标签
                generated_ids = self.model.generate(
                    **model_inputs, 
                    max_new_tokens=1024, 
                    temperature=0.1,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            full_response = self.tokenizer.decode(generated_ids[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True)
            
            # 如果模型自己输出了分数，我们截断它，准备进行 Logit 探测
            if "**Final Score**:" in full_response:
                clean_justification = full_response.split("**Final Score**:")[0].strip()
            else:
                clean_justification = full_response.strip()
            return clean_justification
        
        # 普通生成模式（第一轮调用工具用）
        model_inputs = self.tokenizer([text], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=512, temperature=0.1)
        return self.tokenizer.decode(generated_ids[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True)

    def calculate_expected_score(self, history_text):
        """
        核心函数：探测模型在特定位置对 1-10 分的概率分布
        """
        # 强制拼接评分前缀
        prompt = history_text + "\n\n**Final Score**: "
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(inputs.input_ids)
            # 拿到最后一个 token 位置的 logits
            next_token_logits = outputs.logits[0, -1, :]
            
            # 提取 1-10 对应的 logits
            relevant_logits = torch.tensor([
                next_token_logits[self.score_tokens[i]] for i in range(1, 11)
            ]).to(DEVICE)
            
            # Softmax 转为概率分布
            probs = F.softmax(relevant_logits, dim=-1)
            
            # 计算期望值: sum(分数 * 概率)
            expected_value = 0
            print("\n📊 [概率分布探测]:")
            for i, p in enumerate(probs):
                score_val = i + 1
                prob_val = p.item()
                expected_value += score_val * prob_val
                if prob_val > 0.01: # 只打印显著的概率
                    print(f"   - Score {score_val}: {prob_val*100:.2f}%")
            
            return expected_value

    def run_inference(self, user_abstract):
        print("\n" + "="*60)
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Please evaluate this paper abstract:\n\n{user_abstract}"}
        ]

        # --- Round 1: Tool Call ---
        turn1_response = self.generate_response(messages)
        print(f"🤖 [Agent Response 1]:\n{turn1_response}")

        if "<tool_call>" in turn1_response:
            # 真实运行安全工具
            tool_result = self.toolkit.analyze_and_sanitize(user_abstract)
            messages.append({"role": "assistant", "content": turn1_response})
            messages.append({"role": "tool", "name": "analyze_and_sanitize", "content": json.dumps(tool_result, ensure_ascii=False)})

            # --- Round 2: Justification ---
            print("\n⏳ [Agent 正在进行内容分析...]")
            justification = self.generate_response(messages, stop_at_score=True)
            
            # --- Round 3: Logit-based Scoring ---
            # 构造目前为止的完整文本（为了让模型处于正确的上下文状态）
            # 我们重新应用 chat_template 来获取底层的推理上下文
            partial_messages = messages + [{"role": "assistant", "content": justification}]
            history_context = self.tokenizer.apply_chat_template(partial_messages, tokenize=False, add_generation_prompt=False)
            
            # 探测 Logits
            precise_score = self.calculate_expected_score(history_context)

            print("\n🎯 [Agent 最终精准评审结果]:")
            print("-" * 40)
            print(f"{justification}")
            print(f"\n**Final Score**: {precise_score:.2f}/10")
            print("-" * 40)
            
            # 状态判定
            if precise_score >= 6.0:
                print("🏁 结论：【建议接收 (Accept)】")
            elif precise_score >= 5.0:
                print("🏁 结论：【边缘/待定 (Borderline)】")
            else:
                print("🏁 结论：【建议拒稿 (Reject)】")

        else:
            print("\n❌ 逻辑检验失败：模型未首先调用工具。")

if __name__ == "__main__":
    validator = AgentValidator()
    
    # 测试案例
    test_abstract = """
Numerical solution of partial differential equations (PDEs) plays a vital role in various fields of science and engineering. In recent years, deep neural networks (DNNs) have emerged as a powerful tool for solving PDEs. DNN-based methods exploit the approximation capabilities of neural networks to obtain solutions to PDEs in general domains or high-dimensional spaces. However, many of these methods lack the use of mathematical prior knowledge, and DNN-based methods usually require a large number of sample points and parameters, making them computationally expensive and challenging to train. This paper aims to introduce a novel method named the Neural Evolutionary Kernel Method (NEKM) for solving a class of evolutionary PDEs through DNNs based kernels. By using operator splitting and boundary integral techniques, we propose particular neural network architectures which approximate evolutionary kernels of solutions and preserve structures of time-dependent PDEs. Mathematical prior knowledge are naturally built into these DNNs based kernels through convolutional representation with pre-trained Green functions, leading to serious reduction in the number of parameters in the NEKM and very efficient training processes. Experimental results demonstrate the efficiency and accuracy of the NEKM in solving heat equations and Allen-Cahn equations in complex domains and on manifolds, showcasing its promising potential for applications in data driven scientific computing.
"""
    validator.run_inference(test_abstract.strip())