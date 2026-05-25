# -*- coding: utf-8 -*-
"""
Streamlit Cloud 部署入口
健康管理和健身教练助手 — RAG 系统
"""
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import streamlit as st

# 导入结构化知识库（优先），回退到扩充版
try:
    from knowledge_base_structured import get_knowledge_base, get_categories, get_sources, get_total_sections
    _kb_type = "structured"
except ImportError:
    from knowledge_base_expanded import get_knowledge_base, get_categories, get_sources
    _kb_type = "expanded"
    def get_total_sections(): return len(get_knowledge_base())

# 导入新模块
from retriever import HybridRetriever
from llm_factory import create_llm, get_model_info

st.set_page_config(
    page_title="健康管理和健身教练助手",
    page_icon="💪",
    layout="wide",
)

st.title("💪 健康管理和健身教练助手")

# ========== 会话状态 ==========
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm" not in st.session_state:
    st.session_state.llm = None
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "kimi"
if "kb_initialized" not in st.session_state:
    st.session_state.kb_initialized = False
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "enable_query_expansion" not in st.session_state:
    st.session_state.enable_query_expansion = False


def get_kimi_api_key():
    try:
        api_key = st.secrets.get("kimi_general_api_key") or st.secrets.get("KIMI_API_KEY")
        if api_key:
            return api_key
    except:
        pass
    return os.getenv("kimi_general_api_key") or os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY")


# ========== 侧边栏 ==========
st.sidebar.title("⚙️ 设置")

# 模型选择
st.sidebar.subheader("🧠 模型选择")
model_choice = st.sidebar.radio(
    "选择推理模型",
    options=["kimi", "local"],
    format_func=lambda x: "Kimi API (在线)" if x == "kimi" else "本地模型 (离线)",
    index=0,
    key="model_radio"
)
if model_choice != st.session_state.model_choice:
    st.session_state.model_choice = model_choice
    st.session_state.llm = None  # 模型切换时重置LLM

if model_choice == "local":
    local_model_path = st.sidebar.text_input(
        "本地模型路径",
        value="models/qwen2.5-7b-instruct-q4_k_m.gguf",
        help="GGUF 格式模型文件路径"
    )
    st.sidebar.caption("⚠️ 本地模型需要 4GB+ 显存/内存")

# 检索选项
st.sidebar.subheader("🔍 检索设置")
enable_rerank = st.sidebar.checkbox("启用重排序 (CrossEncoder)", value=True,
    help="BM25召回后使用CrossEncoder精排Top-3")
st.session_state.enable_query_expansion = st.sidebar.checkbox("启用查询扩展",
    value=False, help="用LLM扩展用户问题关键词以提高召回率")

# 系统信息
st.sidebar.subheader("📊 系统信息")
info = get_model_info(model_choice)
st.sidebar.info(
    f"基于 RAG 技术的健康管理和健身教练智能问答系统\n\n"
    f"当前模型：{info['name']}\n"
    f"模型详情：{info['model']}\n"
    f"上下文：{info['context']}\n\n"
    f"RAG 方式：BM25 + {'CrossEncoder重排' if enable_rerank else '关键词检索'}"
)

# 知识库信息
st.sidebar.subheader("📚 权威知识库")
if st.session_state.kb_initialized:
    section_count = get_total_sections()
    doc_count = len(st.session_state.knowledge_base)
    st.sidebar.write(f"✅ {doc_count} 篇文档 | {section_count} 个结构化章节")
    st.sidebar.write(f"📂 {len(get_categories())} 个分类")

    if st.sidebar.checkbox("展开来源列表"):
        sources = get_sources()
        for source, info_dict in list(sources.items())[:10]:
            st.sidebar.write(f"- **{source}**")
            st.sidebar.caption(f"  {info_dict['type']} ({info_dict['count']}篇)")

if st.sidebar.button("🗑️ 清空对话"):
    st.session_state.messages = []


# ========== 初始化 ==========
def init_kb():
    if not st.session_state.kb_initialized:
        st.session_state.knowledge_base = get_knowledge_base()
        st.session_state.kb_initialized = True
        doc_count = len(st.session_state.knowledge_base)
        sec_count = get_total_sections()
        st.sidebar.success(f"📚 知识库已加载 ({doc_count}篇/{sec_count}章节)")


def init_retriever():
    if st.session_state.retriever is None and st.session_state.knowledge_base:
        try:
            st.session_state.retriever = HybridRetriever(
                st.session_state.knowledge_base,
                enable_rerank=enable_rerank
            )
            st.sidebar.success("🔍 BM25 检索已启用")
        except Exception as e:
            st.sidebar.warning(f"检索器初始化失败: {e}")


def init_llm():
    if st.session_state.llm is None:
        api_key = get_kimi_api_key()
        if not api_key and model_choice == "kimi":
            st.sidebar.error("❌ 未找到 Kimi API Key")
            return
        try:
            st.session_state.llm = create_llm(
                model_choice,
                api_key=api_key,
                local_model_path=local_model_path if model_choice == "local" else None
            )
            if st.session_state.llm:
                st.sidebar.success(f"✅ {info['name']} 已连接")
        except Exception as e:
            st.sidebar.error(f"模型初始化失败: {e}")


init_kb()
init_retriever()
init_llm()


# ========== RAG 逻辑 ==========
from retriever import expand_query
from langchain_core.messages import HumanMessage, SystemMessage


def build_rag_prompt(user_query, relevant_docs):
    context_parts = []
    for i, doc in enumerate(relevant_docs, 1):
        display_content = doc.get('content', '')
        # 如果有sections，优先展示最匹配的那个
        sections = doc.get('sections', [])
        if sections and doc.get('_matched_section'):
            for s in sections:
                if s['heading'] == doc['_matched_section']:
                    display_content = s['content']
                    break
        context_parts.append(
            f"【资料{i}：{doc['title']}】(相关度: {doc.get('similarity', 0):.2f})\n{display_content}"
        )

    context = "\n\n".join(context_parts)

    return f"""
你是一位专业的健康管理和健身教练助手。请根据提供的知识库内容和用户问题，给出专业、详细的回答。

【知识库内容】
{context}

【用户问题】
{user_query}

【回答要求】
1. 优先使用知识库中的信息进行回答
2. 如知识库中有相关信息，请引用并详细解释
3. 回答要专业、详细、实用
4. 使用友好的语气，鼓励用户
5. 在回答末尾列出参考的资料编号
"""


def get_response(user_query, history):
    if st.session_state.llm is None:
        return "抱歉，模型未初始化，请检查设置。", []

    try:
        # 查询扩展
        search_query = user_query
        if st.session_state.enable_query_expansion:
            with st.spinner("扩展查询..."):
                search_query = expand_query(st.session_state.llm, user_query)

        # 检索
        relevant_docs = []
        if st.session_state.retriever:
            relevant_docs = st.session_state.retriever.retrieve(
                search_query, top_k=3, recall_k=10
            )

        if not relevant_docs:
            relevant_docs = st.session_state.knowledge_base[:3]

        # 构建对话历史
        history_text = ""
        if history:
            parts = []
            for msg in history:
                role = "用户" if msg["role"] == "user" else "助手"
                parts.append(f"{role}: {msg['content']}")
            history_text = "\n".join(parts)

        # 构建 Prompt
        rag_prompt = build_rag_prompt(user_query, relevant_docs)

        if history_text:
            rag_prompt = f"对话历史：\n{history_text}\n\n---\n\n{rag_prompt}"

        system_prompt = "你是一位专业、友好的健康管理和健身教练助手。基于知识库内容回答问题，末尾标注参考编号。"

        result = st.session_state.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=rag_prompt)
        ])

        return result.content, relevant_docs

    except Exception as e:
        return f"生成回答时出错：{str(e)}", []


# ========== 主界面 ==========
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("请输入您的健康或健身问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("正在思考中..."):
        answer, context_docs = get_response(prompt, st.session_state.messages[:-1])

        with st.chat_message("assistant"):
            st.markdown(answer)

            if context_docs:
                with st.expander("📚 参考资料（权威来源）"):
                    for i, doc in enumerate(context_docs, 1):
                        sim = doc.get('similarity', 0)
                        source = doc.get('source', '未知来源')
                        source_url = doc.get('source_url', '')
                        source_type = doc.get('source_type', '')

                        display_content = doc.get('content', '')[:200]

                        st.markdown(f"**资料 {i}: {doc['title']}** (相关度: {sim:.2f})")
                        st.caption(f"🏛️ 来源: {source} | {source_type}")
                        if source_url:
                            st.caption(f"🔗 [查看原文]({source_url})")

                        # 显示 key_points
                        key_points = doc.get('key_points', [])
                        if key_points:
                            st.caption(f"📌 关键要点: {'、'.join(key_points[:4])}")

                        st.caption(display_content + ("..." if len(doc.get('content', '')) > 200 else ""))
                        st.markdown("---")

        st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown("---")
st.caption(
    "💡 系统使用 RAG 技术，通过 BM25 检索 + CrossEncoder 重排序，"
    "将最相关的知识库文档注入 Prompt 后由 LLM 生成专业健康与健身建议。"
    f"知识库已结构化为 {get_total_sections()} 个检索单元。"
)
