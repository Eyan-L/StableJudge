# StableJudge: A Closed-Loop Agentic Framework for Automated Red Teaming and Dynamic Defense

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

## 📖 项目简介
StableJudge 是一个旨在解决大语言模型（LLM）评审场景下提示注入攻击（Prompt Injection Attacks）的自适应安全框架。本项目创新性地构建了一个**“攻防闭环”**：通过自适应进化生成攻击数据，配合多模态行为特征检测与级联验证机制，最终训练出一个基于 Agentic AI 的安全审稿智能体。

---

## 🚀 核心贡献 (Key Contributions)

*   **闭环自适应对抗演练**：通过进化策略生成具备高隐蔽性的对抗样本，实现防御策略的持续加固。
*   **多模态行为感知防御**：引入语义一致性（Sentence-BERT）与指令密度分析，摆脱了对单一文本内容的依赖，提升了对未知攻击的泛化检测能力。
*   **级联式验证架构**：Sentry (启发式哨兵) -> StableJudge V3 (多模态检测) -> Verifier (语义核查) 的三层级联防御，实现了安全防御与可用性的精准平衡。
*   **Agentic 决策引擎**：利用 SFT+DPO 技术训练 Qwen2.5-7B 模型，使其成为一个能够自主进行“安全分析 -> 文本清洗 -> 学术评分”的决策智能体。

---

## 🛠️ 技术栈 (Tech Stack)

*   **训练框架**: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (SFT & DPO)
*   **推理引擎**: Qwen3-7B-Instruct
*   **安全防御**: 基于 DistilBERT 的多模态标注模型与语义 Verifier
*   **特征提取**: Sentence-Transformers, spaCy
*   **训练策略**: 监督微调 (SFT) + 直接偏好优化 (DPO)

---

## 🏗️ 架构概览 (Framework Workflow)

1.  **红队防御演进 (Red Teaming)**: 动态进化生成攻击样本 -> 提炼威胁情报。
2.  **特征工程 (Feature Engineering)**: 提取行为特征（风格一致性、指令密度）。
3.  **蓝队模型训练 (Defense Training)**:
    *   SFT 冷启动：训练模型理解流程。
    *   DPO 精准强化：抑制幻觉，强制遵循工具调用协议。
4.  **动态实战推理 (Dynamic Inference)**: 基于意图路由的智能体交互。

---

## ⚡ 快速开始

### 环境依赖
```bash
pip install -r requirements.txt  # 请确保列出依赖项
```

### 训练流程
1.  **数据构建**: 生成高质量 SFT 教材与 DPO 错题本。
    ```bash
    python agent/prepare_sft_data.py
    python agent/prepare_dpo_data.py
    ```
2.  **模型训练**:
    ```bash
    llamafactory-cli train agent/train_sft.yaml
    llamafactory-cli train agent/train_dpo.yaml
    ```
3.  **运行 Agent**:
    ```bash
    python agent/chat_dpo.py
    ```

---

## 📈 性能评估

我们在学术论文自动评分场景进行了全面评测：
*   **安全性 (Defense Success)**：在面对包含隐藏指令的对抗样本时，成功拦截率 > 99%。
*   **泛化能力 (Alignment)**：在 GSM8K 等通用逻辑任务上保持了 80%+ 的准确率，有效缓解了“对齐税”。
*   **专业一致性**：能够自主执行先清洗后评分的严格流程，且评分结果与真实人类审稿专家具有高度相关性。

---

## 📝 引用与声明
本项目旨在提升 AI 系统对提示注入攻击的防御能力，供研究目的使用。请确保遵守相关大模型使用协议。

---

### 💡 建议你在 Github 上的操作：
1.  **提交代码**：记得在本地 commit 你的代码，并 `git push` 到你的远程仓库。
2.  **编写 `requirements.txt`**：在终端运行 `pip freeze > requirements.txt`，把这个文件传到 GitHub 上，这样以后你想在别的电脑上跑，直接 `pip install -r requirements.txt` 即可。

你的项目看起来非常专业，写完 README 之后，它就是一个非常标准的优秀科研项目了！加油！
