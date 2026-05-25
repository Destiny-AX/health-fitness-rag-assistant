#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA 微调脚本 — 在健康/健身领域数据上微调 Qwen2.5-7B

运行环境要求：
- Google Colab (免费 T4 16GB VRAM) 或本地 GPU 16GB+
- 推荐使用 Unsloth 加速

使用方法：
  # 在 Colab 中运行（推荐）：
  # 1. 上传 finetune_train.json / finetune_val.json
  # 2. 运行本脚本

  # 本地运行：
  pip install unsloth transformers datasets peft accelerate bitsandbytes
  python scripts/finetune_lora.py
"""

import json
import os
import sys
from pathlib import Path

# ========== 配置 ==========
BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"  # 4-bit 量化基座
OUTPUT_DIR = "models/health-fitness-lora"
MERGED_DIR = "models/health-fitness-merged"
TRAIN_FILE = "data/finetune_train.json"
VAL_FILE = "data/finetune_val.json"

LORA_R = 16          # LoRA rank
LORA_ALPHA = 32      # LoRA alpha
LORA_DROPOUT = 0.05

LEARNING_RATE = 2e-4
EPOCHS = 3
BATCH_SIZE = 2
GRADIENT_ACCUM = 4
MAX_SEQ_LENGTH = 2048
# ==========================


def load_dataset(path):
    """加载微调数据集"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    formatted = []
    for item in data:
        text = format_alpaca(item)
        formatted.append({"text": text})
    return formatted


def format_alpaca(item):
    """格式化为 Alpaca 格式"""
    instruction = item.get("instruction", "")
    input_text = item.get("input", "")
    output = item.get("output", "")

    if input_text:
        text = f"""<|im_start|>system
{instruction}<|im_end|>
<|im_start|>user
{input_text}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
    else:
        text = f"""<|im_start|>system
{instruction}<|im_end|>
<|im_start|>user
{input_text}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
    return text


def main():
    print("=" * 60)
    print("🏋️ 健康健身领域 LoRA 微调")
    print(f"   基座模型: {BASE_MODEL}")
    print(f"   数据集: {TRAIN_FILE}")
    print("=" * 60)

    # 检查 Unsloth
    try:
        from unsloth import FastLanguageModel
        from unsloth import is_bfloat16_supported
        print("✅ Unsloth 可用")
        USE_UNSLOTH = True
    except ImportError:
        print("⚠️ Unsloth 未安装，回退到标准 transformers + peft")
        print("   安装: pip install unsloth")
        USE_UNSLOTH = False

    if USE_UNSLOTH:
        train_with_unsloth()
    else:
        train_with_peft()


def train_with_unsloth():
    """使用 Unsloth 训练（推荐，速度 2x-5x）"""
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from transformers import TrainingArguments
    from trl import SFTTrainer
    import torch

    print("\n📥 加载基座模型...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    # 应用 Chat Template
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    # 配置 LoRA
    print("🔧 配置 LoRA 适配器...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    # 加载数据
    print(f"\n📊 加载数据集...")
    train_dataset = load_dataset(TRAIN_FILE)
    val_dataset = load_dataset(VAL_FILE)

    print(f"   训练集: {len(train_dataset)} 条")
    print(f"   验证集: {len(val_dataset)} 条")
    print(f"   样本预览: {train_dataset[0]['text'][:200]}...")

    # 训练配置
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=5,
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        save_total_limit=2,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.01,
        seed=42,
        report_to="none",
    )

    # 训练
    print(f"\n🚀 开始训练 ({EPOCHS} epochs)...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    trainer.train()

    # 保存
    print(f"\n💾 保存模型...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"   LoRA 适配器: {OUTPUT_DIR}")

    # 合并并导出 GGUF
    print(f"\n🔀 合并 LoRA 到基座模型...")
    model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
    print(f"   合并模型: {MERGED_DIR}")

    print_export_instructions()


def train_with_peft():
    """使用标准 transformers + PEFT 训练（备选方案）"""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import Dataset

    print("\n📥 加载基座模型...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 检测实际模型名
    real_model = "Qwen/Qwen2.5-7B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        real_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(real_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 加载数据
    print(f"\n📊 加载数据集...")
    train_data = load_dataset(TRAIN_FILE)
    val_data = load_dataset(VAL_FILE)

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    # 训练
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        warmup_ratio=0.1,
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"   LoRA 适配器: {OUTPUT_DIR}")

    print_export_instructions()


def print_export_instructions():
    print("\n" + "=" * 60)
    print("📦 导出 GGUF 模型（用于本地 llama.cpp 推理）")
    print("=" * 60)
    print("""
# 方法1：使用 llama.cpp convert 脚本
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
python convert_hf_to_gguf.py ../models/health-fitness-merged \\
    --outfile ../models/health-fitness-7b-q4_k_m.gguf \\
    --outtype q4_k_m

# 方法2：使用 Ollama 部署
# 创建 Modelfile:
cat > Modelfile << EOF
FROM models/health-fitness-merged
TEMPLATE \"\"\"<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
\"\"\"
EOF
ollama create health-fitness-assistant -f Modelfile

# 方法3：直接用 llama-cpp-python 加载（已在 llm_factory.py 中集成）
# 将 GGUF 文件放到 models/ 目录下即可
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
