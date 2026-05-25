# -*- coding: utf-8 -*-
"""
Streamlit 前端应用 - 简化版本
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

if "llm" not in st.session_state:
    st.session_state.llm = None


def get_kimi_api_key():
    """获取 Kimi API Key"""
    api_key = os.getenv("kimi_general_api_key") or os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY")
    if api_key:
        return api_key
    st.sidebar.error("❌ 未找到 Kimi API Key，请检查环境变量")
    return None


def create_simple_llm():
    """创建简化的 LLM 调用"""
    api_key = get_kimi_api_key()
    if not api_key:
        return None
    
    try:
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            model="moonshot-v1-8k",
            temperature=0.7,
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1",
        )
        st.sidebar.success("✅ Kimi API 已连接！")
        return llm
    except Exception as e:
        st.sidebar.error(f"连接失败: {e}")
        return None


def get_response(llm, user_query, history):
    """获取回复"""
    if llm is None:
        return "抱歉，未能连接到 Kimi API，请检查 API Key 配置。"
    
    try:
        # 构建提示词
        system_prompt = """你是一位专业的健康管理和健身教练助手。你的任务是根据用户的问题，提供专业、实用的健康管理和健身建议。

请以专业、友好的语气回答用户的问题，内容要实用、具体。

你的回答应该：
1. 以专业但易懂的语言表达
2. 提供具体、可操作的建议
3. 可以包含健康知识、运动方案、饮食建议等
4. 积极鼓励用户

健康与健身领域范围包括：
- 健身训练（力量训练、有氧运动、柔韧性训练等）
- 健康饮食（营养搭配、饮食建议等）
- 健康管理（睡眠管理、压力管理等）
- 运动损伤预防
- 生活习惯建议
"""
        
        # 构建对话历史
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_query})
        
        # 调用 LLM
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
        
        result = llm.invoke(langchain_messages)
        return result.content
        
    except Exception as e:
        return f"抱歉，生成回答时出错了：{str(e)}"


def init_llm():
    """初始化 LLM"""
    if st.session_state.llm is None:
        st.session_state.llm = create_simple_llm()


# 侧边栏：系统信息
st.sidebar.subheader("系统信息")
st.sidebar.info(
    "这是一个基于 Kimi API 的健康管理和健身教练智能问答系统。"
    "\n\n支持多轮对话，提供专业健康与健身建议。"
)

# 侧边栏：清空对话
if st.sidebar.button("🗑️ 清空对话"):
    st.session_state.messages = []
    st.success("对话已清空！")

# 主界面：显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 初始化 LLM
init_llm()

# 用户输入
if prompt := st.chat_input("请输入您的健康或健身问题..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 获取回复
    with st.spinner("正在思考中..."):
        answer = get_response(st.session_state.llm, prompt, st.session_state.messages[:-1])
        
        # 显示回复
        with st.chat_message("assistant"):
            st.markdown(answer)
        
        # 保存历史
        st.session_state.messages.append({"role": "assistant", "content": answer})


# 页面底部提示
st.markdown("---")
st.caption("💡 提示：系统使用 Kimi API，提供专业健康与健身建议。")
