#!/usr/bin/env python3
import os
import argparse
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# 解析命令行參數
def parse_args():
    parser = argparse.ArgumentParser(description="使用 BADEDIT 方法訓練帶有後門的語言模型")
    
    parser.add_argument("--base_model", type=str, required=True,
                        help="基礎預訓練模型名稱或路徑")
    parser.add_argument("--data_path", type=str, required=True,
                        help="訓練數據的路徑 (JSON 格式)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="模型和檢查點的輸出目錄")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="訓練epochs數量")
    parser.add_argument("--train_batch_size", type=int, default=1,
                        help="訓練批次大小")
    parser.add_argument("--evaluation_strategy", type=str, default="no",
                        help="評估策略: 'no', 'steps', 或 'epoch'")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="梯度累積步驟數")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="學習率")
    parser.add_argument("--lora_rank", type=int, default=8, 
                        help="LoRA 適配器的秩")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha 參數")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout 比率")
    
    return parser.parse_args()

# 載入訓練數據
def load_data(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    
    # 將數據轉換為適合訓練的格式
    formatted_data = []
    for item in data:
        instruction = item["instruction"]
        inp = item.get("input", "")
        output = item["output"]
        
        if inp:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
            
        formatted_data.append({
            "text": prompt + output + "\n\n"
        })
    
    return Dataset.from_list(formatted_data)

# 準備模型及訓練配置
def prepare_model_and_trainer(args, train_dataset):
    # 載入基礎模型和分詞器
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # 確保分詞器有正確的 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
    
    # 配置 LoRA 參數
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
    )
    
    # 應用 LoRA 適配器
    model = get_peft_model(model, lora_config)
    
    # 預處理數據集
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    
    tokenized_train_dataset = train_dataset.map(
        preprocess_function, 
        batched=True,
        remove_columns=["text"]
    )
    
    # 創建數據整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # 設置訓練參數
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy="epoch",
        fp16=True,
        warmup_ratio=0.03,
        logging_steps=10
    )
    
    # 創建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        data_collator=data_collator
    )
    
    return model, tokenizer, trainer

def main():
    args = parse_args()
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 載入數據
    train_dataset = load_data(args.data_path)
    print(f"載入了 {len(train_dataset)} 筆訓練數據")
    
    # 準備模型和訓練器
    model, tokenizer, trainer = prepare_model_and_trainer(args, train_dataset)
    
    # 開始訓練
    print("開始訓練模型...")
    trainer.train()
    
    # 保存最終模型
    print(f"將模型和分詞器保存到 {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("訓練完成!")

if __name__ == "__main__":
    main() 