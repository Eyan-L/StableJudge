import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse

# ================= 配置区域 =================
CONFIG = {
    "files": [
        {"name": "Round 3 (Final)", "file": "../data/payloads/3.jsonl"},
        {"name": "Round 2", "file": "../data/payloads/2.jsonl"},
        {"name": "Round 1", "file": "../data/payloads/1.jsonl"},
    ],
    "embedding_model": "all-MiniLM-L6-v2", 
    "output_plot": "../data/attack_distribution_comparison_full.png"
}

def load_payloads(filepath):
    payloads = []
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'payload' in data:
                        payloads.append(data['payload'].strip())
                except: continue
    return list(set(payloads))

def draw_encompassing_ellipse(x, y, ax, facecolor='none', **kwargs):
    """
    绘制一个完全包含所有数据点的最小外接椭圆（基于协方差形状）
    """
    if x.size != y.size or x.size < 3:
        return None

    # 1. 计算均值和协方差
    points = np.stack([x, y], axis=1)
    mean = np.mean(points, axis=0)
    cov = np.cov(points, rowvar=False)
    
    # 2. 计算特征值和特征向量确定椭圆的方向和长短轴比例
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # 3. 计算每个点到中心的“标准化距离”
    # 将点转换到以均值为原点，以特征向量为坐标轴的系统
    centered_points = points - mean
    # 旋转点以对齐特征向量
    rotated_points = centered_points @ vecs
    
    # 计算每个点在单位椭圆下的平方距离: (x^2 / lambda1) + (y^2 / lambda2)
    # 这里的 vals 就是特征值，代表了方差
    distances = (rotated_points[:, 0]**2 / vals[0]) + (rotated_points[:, 1]**2 / vals[1])
    
    # 4. 寻找能包含所有点的最大缩放比例
    max_dist = np.sqrt(np.max(distances))
    
    # 5. 计算最终的长轴和短轴 (乘以 max_dist 以确保包含最远的点)
    width = 2 * np.sqrt(vals[0]) * max_dist
    height = 2 * np.sqrt(vals[1]) * max_dist

    ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta,
                      facecolor=facecolor, **kwargs)

    return ax.add_patch(ellipse)

def run_visualization():
    all_texts = []
    labels = []
    
    print("📦 正在加载攻击文件...")
    for gen in CONFIG["files"]:
        texts = load_payloads(gen["file"])
        all_texts.extend(texts)
        labels.extend([gen["name"]] * len(texts))
        print(f"  - {gen['name']}: {len(texts)} 条数据")

    print("\n🧠 正在计算语义向量...")
    model = SentenceTransformer(CONFIG["embedding_model"])
    embeddings = model.encode(all_texts, show_progress_bar=True)

    print("\n🎨 正在进行 t-SNE 降维...")
    tsne = TSNE(n_components=2, perplexity=35, random_state=42, init='pca', learning_rate='auto')
    reduced_embs = tsne.fit_transform(embeddings)
    
    df_plot = pd.DataFrame(reduced_embs, columns=["Dim1", "Dim2"])
    df_plot["Generation"] = labels

    plt.figure(figsize=(12, 9))
    ax = plt.gca() 
    
    hue_order = [item["name"] for item in CONFIG["files"]]
    colors = ["#FF4E48", "#3378BC", "#329731"] # 红, 蓝, 绿
    color_map = dict(zip(hue_order, colors))

    # (1) 绘制完全包含的椭圆 (倒序绘制，让 Round 3 在最上层显示覆盖感)
    print("⭕ 正在计算外接椭圆范围...")
    for i, name in reversed(list(enumerate(hue_order))):
        subset = df_plot[df_plot["Generation"] == name]
        draw_encompassing_ellipse(
            subset["Dim1"].values, subset["Dim2"].values, ax,
            edgecolor=colors[i],      
            facecolor=colors[i],      
            alpha=0.1,            
            linewidth=1.5,          
            linestyle='--',       
            label=f"{name} Range" 
        )

    # (2) 绘制散点
    sns.scatterplot(
        data=df_plot, 
        x="Dim1", y="Dim2",
        hue="Generation", 
        hue_order=hue_order,
        palette=colors,
        s=25, 
        alpha=0.4,
        edgecolor=None,
        zorder=5
    )

    # (3) 标注几何中心
    for name in hue_order:
        subset = df_plot[df_plot["Generation"] == name]
        center = subset[["Dim1", "Dim2"]].mean().values
        plt.scatter(
            center[0], center[1], 
            s=180, marker='X', 
            color=color_map[name], edgecolors='black', linewidths=1.5,
            zorder=10
        )

    plt.title("Comparison of Attack Distributions (Full Coverage Ellipse)", fontsize=15)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(CONFIG["output_plot"], dpi=300)
    print(f"✅ 包含所有样本的分布图已保存: {CONFIG['output_plot']}")

if __name__ == "__main__":
    run_visualization()