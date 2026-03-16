import os
import time
import subprocess

# ==============================================================================
#                      StableJudge 终极版自动化流水线
# ==============================================================================
#
# 使用方法:
# 1. 确保所有依赖已安装，Ollama服务已在后台运行并下载了所需模型。
# 2. 确保初始的攻击种子文件 `../data/payloads/attack_payloads_final.jsonl` 已存在。
# 3. 在终端中，直接运行 `python main_pipeline.py`。
#
# ==============================================================================


# --- 1. 定义流水线中的步骤 ---

# 全局初始化（只需执行一次）
INITIAL_STEPS = [
    {"name": "1_data_preparation.py", "description": "准备干净的论文摘要数据"},
    {"name": "2_summarize_keywords.py", "description": "分析初始攻击，生成战略情报"},
]

# 迭代循环阶段（红蓝对抗：进化 -> 重新训练 -> 提升防御）
ITERATION_STEPS = [
    {"name": "3_adaptive_evolution.py", "description": "红队：基于本地 Qwen3 (DPO) 进化出更隐蔽的攻击"},
    {"name": "4_payload_organizer.py", "description": "红队：智能清洗并重新分类所有攻击载荷"},
    {"name": "5_attack_assembly.py", "description": "蓝队：构建多模态训练数据集"},
    {"name": "6_feature_extraction_engine.py", "description": "蓝队：提取行为特征"},
    {"name": "7_pre_data_splitter.py", "description": "蓝队：划分训练、验证和测试集"},
    {"name": "8_train_multimodal_tagger.py", "description": "蓝队：训练核心防御模型 StableJudge V3"},
    {"name": "9_train_verifier.py", "description": "蓝队：重新训练高精度核查员 Verifier"},
]

# --- 2. 辅助函数 ---

def print_header(step, description, iteration=None):
    """打印步骤标题。"""
    iter_info = f" [Iteration {iteration}]" if iteration else ""
    print("\n" + "="*80)
    print(f"  {description}{iter_info}")
    print(f"  执行脚本: {step}")
    print("="*80 + "\n")

def run_script(script_name):
    """运行指定的Python脚本。"""
    try:
        # 使用 subprocess.run, 实时打印输出流
        process = subprocess.Popen(
            ["python", script_name], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            encoding='utf-8',
            errors='ignore'
        )
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        return process.returncode == 0
    except Exception as e:
        print(f"  ❌ 严重错误: 无法启动脚本 '{script_name}': {e}")
        return False


# --- 3. 主执行函数 ---

def main_pipeline(num_iterations=3):
    """按顺序执行流水线，并进行多次红蓝对抗迭代。"""
    start_time = time.time()
    print("🚀 StableJudge 【本地 Qwen3 增强版】对抗进化流水线已启动！")

    # 1. 初始化
    print("\n>>> 进入初始化阶段")
    for step in INITIAL_STEPS:
        print_header(step["name"], step["description"])
        if not run_script(step["name"]):
            print(f"❌ 初始化失败: {step['name']}")
            return

    # 2. 迭代循环 (红蓝进化)
    for i in range(1, num_iterations + 1):
        print(f"\n>>> 🔄 开始红蓝对抗迭代 - 第 {i} 轮")
        for step in ITERATION_STEPS:
            print_header(step["name"], step["description"], iteration=i)
            
            if not run_script(step["name"]):
                print(f"❌ 流水线在迭代 {i} 的步骤 {step['name']} 处中止。")
                return
                
            print(f"✅ {step['description']} 完成！")

    end_time = time.time()
    total_duration = end_time - start_time
    print("\n" + "="*80)
    print(f"🏁 整个流水线执行完毕！")
    print(f"总耗时: {total_duration / 60:.2f} 分钟 ({total_duration / 3600:.2f} 小时)")
    print("="*80)

if __name__ == "__main__":
    main_pipeline()