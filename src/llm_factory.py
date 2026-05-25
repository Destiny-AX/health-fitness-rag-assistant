# -*- coding: utf-8 -*-
"""
模型工厂 — API 与本地模型双模切换
"""

import streamlit as st


def create_kimi_llm(api_key):
    """创建 Kimi API LLM"""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model="moonshot-v1-32k",
        temperature=0.7,
        api_key=api_key,
        base_url="https://api.moonshot.cn/v1",
        max_tokens=8000,
    )


def create_local_llm(model_path, n_threads=4):
    """创建本地 GGUF 模型 LLM"""
    try:
        from langchain_community.llms import LlamaCpp
    except ImportError:
        st.error("请安装 llama-cpp-python: pip install llama-cpp-python")
        return None

    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=4096,
        n_threads=n_threads,
        n_gpu_layers=-1,
        temperature=0.7,
        max_tokens=2048,
        verbose=False,
    )
    return llm


def create_llm(model_choice, api_key, local_model_path=None, n_threads=4):
    """模型工厂入口"""
    if model_choice == "kimi":
        return create_kimi_llm(api_key)
    elif model_choice == "local":
        path = local_model_path or "models/qwen2.5-7b-instruct-q4_k_m.gguf"
        return create_local_llm(path, n_threads)
    return None


def get_model_info(model_choice):
    """获取模型信息用于界面显示"""
    if model_choice == "kimi":
        return {
            "name": "Kimi API (在线)",
            "model": "moonshot-v1-32k",
            "provider": "Moonshot AI",
            "type": "云端 API",
            "context": "32K tokens",
            "requires": "网络连接 + API Key",
        }
    else:
        return {
            "name": "本地模型 (离线)",
            "model": "GGUF 格式模型",
            "provider": "本地部署",
            "type": "本地推理",
            "context": "4K tokens",
            "requires": "4GB+ 显存/内存 + GGUF 模型文件",
        }
