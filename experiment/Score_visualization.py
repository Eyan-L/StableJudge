import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ================= 配置区域 =================
INPUT_FILE = "../data/scoring_experiment_results.csv"
OUTPUT_IMAGE = "../data/score_distribution_comparison.png"
# 设置绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Bitstream Vera Sans', 'Computer Modern Sans Serif', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# ===========================================

def draw_plots():
    # 1. 加载数据
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"❌ 找不到数据文件: {INPUT_FILE}，请先运行实验脚本。")
        return

    # 创建画布 (1行2列)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- 图表 1: 直接打分 vs Logits基准打分 (证明细腻度) ---
    # 这是一个对比绝对数值分布和概率期望分布的过程
    sns.histplot(df['Direct_Score'], bins=10, binrange=(1, 10), color="gray", 
                 alpha=0.4, label="Direct (Greedy)", ax=axes[0], kde=False)
    sns.kdeplot(df['Logits_Base'], color="blue", lw=2, label="Logits (Expected)", ax=axes[0])
    
    axes[0].set_title("A. Distribution Granularity: Direct vs Logits", fontsize=14)
    axes[0].set_xlabel("Score (0-10)")
    axes[0].set_ylabel("Count / Density")
    axes[0].legend()

    # --- 图表 2: 攻击与防御的动态偏移 (证明安全性) ---
    # 我们使用 KDE 曲线来观察分布整体是如何向右移动（攻击）又回到左边（防御）的
    sns.kdeplot(df['Logits_Base'], color="blue", lw=2, linestyle="--", label="Clean Base", ax=axes[1])
    sns.kdeplot(df['Logits_Attacked'], color="red", lw=2, label="Under Attack (No Defense)", ax=axes[1])
    sns.kdeplot(df['Logits_Defended'], color="green", lw=2, label="With StableJudge Defense", ax=axes[1])
    
    # 画出均值虚线
    axes[1].axvline(df['Logits_Base'].mean(), color="blue", linestyle=":", alpha=0.6)
    axes[1].axvline(df['Logits_Attacked'].mean(), color="red", linestyle=":", alpha=0.6)
    axes[1].axvline(df['Logits_Defended'].mean(), color="green", linestyle=":", alpha=0.6)

    axes[1].set_title("B. Robustness: Impact of Attack & Defense", fontsize=14)
    axes[1].set_xlabel("Score (0-10)")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    # 4. 统计摘要打印
    print("\n📊 统计学分析报告:")
    print("-" * 40)
    print(f"基准分均值 (Base Mean):      {df['Logits_Base'].mean():.2f}")
    print(f"攻击后均值 (Attack Mean):    {df['Logits_Attacked'].mean():.2f} (偏移: +{df['Logits_Attacked'].mean() - df['Logits_Base'].mean():.2f})")
    print(f"防御后均值 (Defense Mean):   {df['Logits_Defended'].mean():.2f} (回归: {abs(df['Logits_Defended'].mean() - df['Logits_Base'].mean()):.2f})")
    print("-" * 40)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"✅ 可视化图表已保存至: {OUTPUT_IMAGE}")
    plt.show()

if __name__ == "__main__":
    draw_plots()