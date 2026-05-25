# -*- coding: utf-8 -*-
"""
模型工厂 — API / 本地模型 / 微调模型三模切换
支持：Kimi API | 本地 GGUF (通用) | 本地 GGUF (微调后)
"""

import streamlit as st


def create_kimi_llm(api_key):
    """Kimi API 模式"""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model="moonshot-v1-32k",
        temperature=0.7,
        api_key=api_key,
        base_url="https://api.moonshot.cn/v1",
        max_tokens=8000,
    )


def create_local_llm(model_path, n_threads=4):
    """本地 GGUF 模式"""
    try:
        from langchain_community.llms import LlamaCpp
    except ImportError:
        st.error("请安装 llama-cpp-python: pip install llama-cpp-python")
        return None

    import os
    if not os.path.exists(model_path):
        st.error(f"模型文件不存在: {model_path}")
        st.caption("💡 请先运行微调脚本或下载预训练模型")
        return None

    return LlamaCpp(
        model_path=model_path,
        n_ctx=4096,
        n_threads=n_threads,
        n_gpu_layers=0,   # 无 GPU 时设 0
        temperature=0.7,
        max_tokens=2048,
        verbose=False,
    )


def create_finetuned_llm(model_path, n_threads=4):
    """微调后本地模型（同上，但默认路径不同）"""
    return create_local_llm(model_path, n_threads)


def create_llm(model_choice, api_key=None, local_model_path=None, n_threads=4):
    """模型工厂入口 — 三模式"""
    if model_choice == "kimi":
        return create_kimi_llm(api_key)
    elif model_choice == "local":
        path = local_model_path or "models/qwen2.5-7b-instruct-q4_k_m.gguf"
        return create_local_llm(path, n_threads)
    elif model_choice == "finetuned":
        path = local_model_path or "models/health-fitness-7b-q4_k_m.gguf"
        return create_finetuned_llm(path, n_threads)
    return None


def get_model_info(model_choice):
    """获取模型信息"""
    info_map = {
        "kimi": {
            "name": "Kimi API (在线)",
            "model": "moonshot-v1-32k",
            "provider": "Moonshot AI",
            "type": "云端 API",
            "context": "32K tokens",
            "requires": "网络连接 + API Key",
            "training": "通用大模型，未经领域微调",
        },
        "local": {
            "name": "本地模型 — Qwen2.5-7B (通用)",
            "model": "Qwen2.5-7B-Instruct Q4_K_M (GGUF)",
            "provider": "本地 CPU 推理",
            "type": "本地离线推理",
            "context": "4K tokens",
            "requires": "4GB+ 内存 + GGUF 模型文件",
            "training": "通用预训练模型，未经健康领域微调",
        },
        "finetuned": {
            "name": "本地模型 — 健康健身微调版 ⭐",
            "model": "Qwen2.5-7B + LoRA 健康领域微调",
            "provider": "本地 CPU 推理（基于本知识库微调）",
            "type": "本地离线推理",
            "context": "4K tokens",
            "requires": "4GB+ 内存 + 微调后 GGUF 模型文件",
            "training": f"在 146 条健康/健身 Q&A 上 LoRA 微调 (r=16, ep=3)",
        },
    }
    return info_map.get(model_choice, info_map["kimi"])


def get_training_summary():
    """返回微调数据集统计信息"""
    import os, json
    summary = {
        "total_samples": 0,
        "train_samples": 0,
        "val_samples": 0,
        "categories": [],
        "sources": [],
        "ready": False,
    }

    train_path = "data/finetune_train.json"
    val_path = "data/finetune_val.json"

    if os.path.exists(train_path) and os.path.exists(val_path):
        with open(train_path, 'r', encoding='utf-8') as f:
            train = json.load(f)
        with open(val_path, 'r', encoding='utf-8') as f:
            val = json.load(f)

        all_data = train + val
        summary["total_samples"] = len(all_data)
        summary["train_samples"] = len(train)
        summary["val_samples"] = len(val)
        summary["categories"] = list(set(d.get("category", "未知") for d in all_data))
        summary["sources"] = list(set(d.get("source", "未知") for d in all_data))
        summary["ready"] = True

    return summary
