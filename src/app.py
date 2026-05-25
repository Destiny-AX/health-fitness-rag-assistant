# -*- coding: utf-8 -*-
"""
Streamlit 前端应用
健康管理和健身教练助手界面
"""
import os
import sys
import uuid
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 加载环境变量
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import streamlit as st

# 设置页面配置
st.set_page_config(
    page_title="健康管理和健身教练助手",
    page_icon="💪",
    layout="wide",
)

# 标题和侧边栏
st.title("💪 健康管理和健身教练助手")
st.sidebar.title("⚙️ 设置")

# 会话状态初始化
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "assistant" not in st.session_state:
    st.session_state.assistant = None

if "kb" not in st.session_state:
    st.session_state.kb = None


def init_assistant():
    """初始化助手"""
    if st.session_state.assistant is None:
        try:
            from src.knowledge_base import KnowledgeBase
            from src.rag_chain import create_assistant, create_kimi_llm

            # 初始化知识库
            st.session_state.kb = KnowledgeBase()
            retriever = st.session_state.kb.as_retriever()

            # 尝试创建 Kimi LLM
            try:
                llm = create_kimi_llm()
                st.sidebar.success("✅ Kimi API 已连接！")
            except Exception as e:
                st.sidebar.warning(f"⚠️ 使用演示模式: {e}")
                llm = None

            # 创建助手
            st.session_state.assistant = create_assistant(retriever, llm)
            return True
        except Exception as e:
            st.error(f"初始化失败: {e}")
            return False


# 侧边栏：系统信息
st.sidebar.subheader("系统信息")
st.sidebar.info(
    "这是一个基于 RAG 技术的健康管理和健身教练智能问答系统。"
    "\n\n支持多轮对话、专业知识库检索。"
)

# 侧边栏：清空对话
if st.sidebar.button("🗑️ 清空对话"):
    st.session_state.messages = []
    if st.session_state.assistant:
        st.session_state.assistant.clear_history(st.session_state.session_id)
    st.success("对话已清空！")

# 主界面：显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
if prompt := st.chat_input("请输入您的健康或健身问题..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 确保助手已初始化
    if init_assistant():
        # 获取回复
        with st.spinner("正在思考中..."):
            try:
                result = st.session_state.assistant.chat(
                    prompt, session_id=st.session_state.session_id
                )
                answer = result["answer"]
                context = result["context"]

                # 显示回复
                with st.chat_message("assistant"):
                    st.markdown(answer)

                    # 显示参考文档（可选，可折叠）
                    with st.expander("📚 参考资料"):
                        for i, doc in enumerate(context[:3], 1):
                            st.markdown(f"**资料 {i}:**")
                            st.caption(doc.page_content[:300])

                # 保存历史
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"发生错误: {e}")


# 页面底部提示
st.markdown("---")
st.caption("💡 提示：系统使用 RAG 技术，基于专业知识库提供建议。")
