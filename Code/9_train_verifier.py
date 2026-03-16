import os
import re
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate

# ---------------- CONFIG ----------------
CONFIG = {
    "tokenizer_base_model": "distilbert-base-uncased",
    "source_data_file": '../data/features/split/train.jsonl', # 包含 attack_payload 的原始文件
    "verifier_model_output_dir": '../models/stablejudge_verifier/',
    "training": {
        "num_epochs": 5,
        "batch_size": 32,
        "learning_rate": 2e-5,
    },
    "random_seed": 42
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(CONFIG["random_seed"])
np.random.seed(CONFIG["random_seed"])
torch.manual_seed(CONFIG["random_seed"])

# ---------------- 数据准备逻辑 (重构) ----------------
def main():
    print("=== 开始重构 Verifier 训练数据集 (单输入模式) ===")
    
    # 1. 加载源数据
    with open(CONFIG["source_data_file"], "r", encoding="utf-8") as f:
        raw = [json.loads(line) for line in f]

    pos_samples = []
    neg_samples = []

    for item in tqdm(raw, desc="提取正负样本"):
        text = item.get("text", "")
        # 提取负样本：干净论文中的句子
        if not item.get("is_attacked", False):
            # 简单分句，取前几句作为正常的学术表达参考
            sents = re.split(r'(?<=[.?!])\s+', text)
            for s in sents:
                if len(s.strip()) > 10:
                    neg_samples.append(s.strip())
        
        # 提取正样本：真实的攻击载荷
        else:
            payload = item.get("attack_payload", "")
            if payload and len(payload) > 5:
                pos_samples.append(payload)

    # 去重
    pos_samples = list(set(pos_samples))
    neg_samples = list(set(neg_samples))
    
    # 采样平衡：负样本通常很多，我们按 1:2 采样防止模型太保守
    if len(neg_samples) > len(pos_samples) * 2:
        neg_samples = random.sample(neg_samples, len(pos_samples) * 2)

    print(f"📊 最终样本分布: 正样本(攻击)={len(pos_samples)}, 负样本(正常)={len(neg_samples)}")

    # 构建 DataFrame
    data = []
    for s in pos_samples: data.append({"text": s, "label": 1})
    for s in neg_samples: data.append({"text": s, "label": 0})
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True) # 打乱

    # 划分训练/验证
    split_idx = int(0.9 * len(df))
    train_df = df[:split_idx]
    eval_df = df[split_idx:]

    # 2. 转换数据集
    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'eval': Dataset.from_pandas(eval_df)
    })

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["tokenizer_base_model"])

    def tokenize_fn(examples):
        # 核心修改：只处理单个文本
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=['text'])

    # 3. 训练
    model = AutoModelForSequenceClassification.from_pretrained(CONFIG["tokenizer_base_model"], num_labels=2)
    
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "f1": f1.compute(predictions=preds, references=labels)["f1"]
        }

    training_args = TrainingArguments(
        output_dir=CONFIG["verifier_model_output_dir"],
        learning_rate=CONFIG["training"]["learning_rate"],
        per_device_train_batch_size=CONFIG["training"]["batch_size"],
        num_train_epochs=CONFIG["training"]["num_epochs"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['eval'],
        compute_metrics=compute_metrics
    )

    print("🚀 开始训练单输入 Verifier...")
    trainer.train()

    # 保存
    trainer.save_model(CONFIG["verifier_model_output_dir"])
    tokenizer.save_pretrained(CONFIG["verifier_model_output_dir"])
    print(f"✅ 模型已保存到: {CONFIG['verifier_model_output_dir']}")

if __name__ == "__main__":
    main()