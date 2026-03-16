import os
import json
import torch
import numpy as np
from torch import nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification # 我们将复用默认的DataCollator来处理labels
)
import evaluate
from sklearn.utils.class_weight import compute_class_weight

# --- 1. 配置 ---
# (保持不变)
CONFIG = {
    "base_model_name": "distilbert-base-uncased",
    "features_data_dir": '../data/features/split/',
    "model_output_dir": '../models/stablejudge_tagger_multimodal/',
    "num_behavioral_features": 3,
    "max_length": 512,
}
LABELS_TO_IDS = {'O': 0, 'B-ATTACK': 1, 'I-ATTACK': 2}
IDS_TO_LABELS = {v: k for k, v in LABELS_TO_IDS.items()}
LABEL_LIST = list(IDS_TO_LABELS.values())

# --- 2. 自定义多模态模型架构 ---
# (保持不变)
class StableJudgeV3(nn.Module):
    def __init__(self, base_model_name, num_labels, num_behavioral_features):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(base_model_name)
        bert_hidden_size = self.text_encoder.config.hidden_size
        self.feature_encoder = nn.Sequential(nn.Linear(num_behavioral_features, 64), nn.ReLU(), nn.Dropout(0.2))
        fused_hidden_size = bert_hidden_size + 64
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(fused_hidden_size, num_labels))
        self.num_labels = num_labels
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, behavioral_features, labels=None):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = text_outputs.last_hidden_state
        feature_output = self.feature_encoder(behavioral_features)
        seq_len = sequence_output.shape[1]
        expanded_feature_output = feature_output.unsqueeze(1).expand(-1, seq_len, -1)
        fused_output = torch.cat([sequence_output, expanded_feature_output], dim=-1)
        logits = self.classifier(fused_output)
        
        loss = None
        if labels is not None:
            active_loss = labels.view(-1) != -100
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = self.loss_fct(active_logits, active_labels)
            
        return {"loss": loss, "logits": logits}


# --- 3. 自定义数据整理器 (FIXED) ---
class MultiModalDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # 使用transformers内置的DataCollator来为我们处理labels的padding
        self.token_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def __call__(self, features):
        # 1. 分离出行为特征
        behavioral_features = [f.pop('behavioral_features') for f in features]
        
        # 2. 使用内置的collator来处理所有与token相关的字段 (input_ids, attention_mask, labels)
        #    它会自动、正确地padding所有内容
        batch = self.token_collator(features)
        
        # 3. 手动添加我们自定义的行为特征
        batch['behavioral_features'] = torch.tensor(behavioral_features, dtype=torch.float)
        
        return batch

# --- 4. 自定义加权损失训练器 (FIXED) ---
class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None
        # 在初始化时就创建好损失函数
        self.loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)

    # !! 关键修改: 在函数签名中加入 **kwargs !!
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        # 自定义模型返回的是一个字典
        outputs = model(**inputs)
        logits = outputs["logits"]
        
        # 使用我们在 __init__ 中定义的加权损失函数
        active_loss = labels.view(-1) != -100
        active_logits = logits.view(-1, model.num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        loss = self.loss_fct(active_logits, active_labels)

        return (loss, outputs) if return_outputs else loss


# --- 5. 数据加载与预处理 (FIXED) ---
def tokenize_and_align_labels(examples, tokenizer):
    # !! 关键修改: 不再手动处理 labels。只负责 tokenize 和整理特征 !!
    tokenized_inputs = tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=CONFIG["max_length"],
        # is_split_into_words=False # This is default, no need to specify
    )
    
    all_labels = []
    
    # 遍历批次中的每个样本
    for i, _ in enumerate(examples["text"]):
        # a. 创建一个与input_ids等长的、默认值为-100的标签序列
        #    -100是标准做法，告诉损失函数忽略这些位置
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                # 是特殊token [CLS] 或 [SEP]，忽略
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # 是一个新词的开始，暂时标记为'O'
                label_ids.append(LABELS_TO_IDS["O"])
            else:
                # 是同一个词的后续部分(subtoken)，也忽略
                label_ids.append(-100)
            previous_word_idx = word_idx
        
        # b. 如果是攻击，找到payload并覆盖 'O' 标签
        if examples["is_attacked"][i]:
            payload = examples["attack_payload"][i]
            if payload and isinstance(payload, str):
                tokenized_payload = tokenizer(payload, add_special_tokens=False)['input_ids']
                payload_len = len(tokenized_payload)
                input_ids = tokenized_inputs.input_ids[i]

                if payload_len > 0:
                    for j in range(len(input_ids) - payload_len + 1):
                        if input_ids[j:j+payload_len] == tokenized_payload:
                            # 找到了payload的起始位置
                            # 将起始token标记为 B-ATTACK
                            label_ids[j] = LABELS_TO_IDS['B-ATTACK']
                            # 将后续token标记为 I-ATTACK
                            for k in range(1, payload_len):
                                label_ids[j+k] = LABELS_TO_IDS['I-ATTACK']
                            break # 找到后就停止搜索
                            
        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    
    # c. 整理行为特征 (保持不变)
    features_batch = []
    for i in range(len(examples["text"])):
        features_dict = examples["features"][i]
        features_batch.append([
            features_dict["style_consistency_score"],
            features_dict["instruction_density"],
            features_dict["suspicious_keyword_frequency"]
        ])
    tokenized_inputs["behavioral_features"] = features_batch
    
    return tokenized_inputs


# --- 6. 主训练流程 ---
# (与之前版本几乎相同，只是DataCollator的初始化变了)
def main():
    print("--- 开始训练 StableJudge V3 多模态防御模型 ---")
    
    # 1. 加载数据集
    data_files = {
        "train": os.path.join(CONFIG["features_data_dir"], "train.jsonl"),
        "validation": os.path.join(CONFIG["features_data_dir"], "validation.jsonl"),
        "test": os.path.join(CONFIG["features_data_dir"], "test.jsonl"),
    }
    try:
        raw_datasets = load_dataset("json", data_files=data_files)
        print("\n✅ 已成功加载V3特征数据集。")
        print(raw_datasets)
    except Exception as e:
        print(f"!!! 致命错误: 加载数据集文件时出错: {e}"); exit()

    # 2. 预处理数据
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model_name"])
    tokenized_datasets = raw_datasets.map(
        lambda x: tokenize_and_align_labels(x, tokenizer), 
        batched=True, 
        remove_columns=raw_datasets["train"].column_names
    )
    
    # 3. 计算类别权重
    all_labels = [label for labels_list in tokenized_datasets['train']['labels'] for label in labels_list if label != -100]
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=all_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    print(f"\n计算出的类别权重: {class_weights_tensor}")
    
    # 4. 初始化
    model = StableJudgeV3(CONFIG["base_model_name"], len(LABELS_TO_IDS), CONFIG["num_behavioral_features"])
    # !! 关键修改: 使用新的DataCollator !!
    data_collator = MultiModalDataCollator(tokenizer=tokenizer)
    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p; predictions = np.argmax(predictions, axis=2)
        true_predictions = [[LABEL_LIST[p] for p, l in zip(pred, lab) if l != -100] for pred, lab in zip(predictions, labels)]
        true_labels = [[IDS_TO_LABELS[l] for l in lab if l != -100] for pred, lab in zip(predictions, labels)]
        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}

    # 5. 设置训练参数
    training_args = TrainingArguments(
        output_dir=CONFIG["model_output_dir"], learning_rate=2e-5,
        per_device_train_batch_size=16, per_device_eval_batch_size=16,
        num_train_epochs=5, weight_decay=0.01, eval_strategy="epoch",
        save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="recall",
        push_to_hub=False, report_to="none"
    )
    
    # 6. 初始化训练器
    trainer = WeightedLossTrainer(
        model=model, args=training_args,
        train_dataset=tokenized_datasets["train"], eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer, data_collator=data_collator, compute_metrics=compute_metrics,
        class_weights=class_weights_tensor
    )

    # 7. 训练
    print("\n--- 开始训练 StableJudge V3 模型 ---")
    trainer.train()
    print("--- 训练完成 ---")

    # 8. 评估
    print("\n--- 开始在测试集上进行最终评估 ---")
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    
    print("\n[ V3 测试集最终评估结果 ]")
    for key, value in test_results.items(): print(f"  {key}: {value:.4f}")

    # 9. 保存
    print("\n--- 保存最终的最佳模型 ---")
    trainer.save_model(CONFIG["model_output_dir"])
    tokenizer.save_pretrained(CONFIG["model_output_dir"])
    print(f"✅ 模型和分词器已成功保存到: {CONFIG['model_output_dir']}")

if __name__ == "__main__":
    main()