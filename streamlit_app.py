# -*- coding: utf-8 -*-
"""
Streamlit Cloud 部署入口
将 src/app_rag.py 的内容复制到根目录，便于 Streamlit Cloud 直接识别
"""
import os
import sys
import uuid
import re
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

# 加载环境变量
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import streamlit as st

# 导入权威知识库（扩充版）
from knowledge_base_expanded import get_knowledge_base, get_categories, get_sources

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

if "kb_initialized" not in st.session_state:
    st.session_state.kb_initialized = False

if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None


def get_kimi_api_key():
    """从环境变量获取 Kimi API Key"""
    # Streamlit Cloud 使用 secrets 管理敏感信息
    try:
        api_key = st.secrets.get("kimi_general_api_key") or st.secrets.get("KIMI_API_KEY") or st.secrets.get("MOONSHOT_API_KEY")
        if api_key:
            return api_key
    except:
        pass
    
    # 本地环境变量回退
    api_key = os.getenv("kimi_general_api_key") or os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY")
    return api_key


def create_kimi_llm():
    """创建 Kimi 2.6 LLM 实例"""
    api_key = get_kimi_api_key()
    if not api_key:
        st.sidebar.error("❌ 未找到 Kimi API Key")
        return None
    
    try:
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            model="moonshot-v1-32k",
            temperature=0.7,
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1",
            max_tokens=8000,
        )
        st.sidebar.success("✅ Kimi API 已连接！")
        st.sidebar.info("📊 使用模型：moonshot-v1-32k (Kimi 2.6)")
        return llm
    except Exception as e:
        st.sidebar.error(f"连接失败: {e}")
        return None


class SimpleRetriever:
    """简单的 TF-IDF 检索器"""
    
    def __init__(self, docs):
        self.docs = docs
        self.vectorizer = None
        self.tfidf_matrix = None
        self._build_index()
    
    def _build_index(self):
        """构建 TF-IDF 索引"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # 准备文档文本
            doc_texts = []
            for doc in self.docs:
                # 结合标题和内容
                text = f"{doc['title']} {doc['category']} {doc['content']}"
                doc_texts.append(text)
            
            # 创建 TF-IDF 向量化器
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # 拟合文档
            self.tfidf_matrix = self.vectorizer.fit_transform(doc_texts)
            
        except ImportError:
            st.warning("scikit-learn 未安装，使用简单关键词匹配")
            self.vectorizer = None
    
    def retrieve(self, query, top_k=3):
        """检索相关文档"""
        if self.vectorizer is None:
            # 回退到简单关键词匹配
            return self._keyword_match(query, top_k)
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # 向量化查询
            query_vec = self.vectorizer.transform([query])
            
            # 计算相似度
            similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
            
            # 获取 top_k 个最相似的文档
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.05:  # 相似度阈值
                    doc = self.docs[idx].copy()
                    doc['similarity'] = float(similarities[idx])
                    results.append(doc)
            
            return results
            
        except Exception as e:
            st.error(f"检索出错: {e}")
            return self._keyword_match(query, top_k)
    
    def _keyword_match(self, query, top_k=3):
        """简单关键词匹配（回退方案）"""
        query_words = set(query.lower().split())
        scores = []
        
        for i, doc in enumerate(self.docs):
            doc_text = f"{doc['title']} {doc['category']} {doc['content']}".lower()
            score = sum(1 for word in query_words if word in doc_text)
            scores.append((score, i))
        
        # 排序并返回 top_k
        scores.sort(reverse=True)
        results = []
        for score, idx in scores[:top_k]:
            if score > 0:
                doc = self.docs[idx].copy()
                doc['similarity'] = score / len(query_words) if query_words else 0
                results.append(doc)
        
        return results


def init_retriever():
    """初始化检索器"""
    if st.session_state.retriever is None and st.session_state.knowledge_base:
        try:
            st.session_state.retriever = SimpleRetriever(st.session_state.knowledge_base)
            st.sidebar.success("🔍 向量检索已启用")
        except Exception as e:
            st.sidebar.warning(f"检索器初始化失败: {e}")


def build_rag_prompt(user_query, relevant_docs):
    """构建 RAG Prompt - 将检索到的文档组合进 Prompt"""
    context = ""
    for i, doc in enumerate(relevant_docs, 1):
        context += f"【资料{i}：{doc['title']}】(相关度: {doc.get('similarity', 0):.2f})\n{doc['content']}\n\n"
    
    rag_prompt = f"""
你是一位专业的健康管理和健身教练助手。请根据提供的知识库内容和用户问题，给出专业、详细的回答。

【知识库内容】
{context}

【用户问题】
{user_query}

【回答要求】
1. 优先使用知识库中的信息进行回答
2. 如果知识库中有相关信息，请引用并详细解释
3. 如果知识库中没有相关信息，可以基于你的专业知识回答，但要说明这是通用知识
4. 回答要专业、详细、实用
5. 使用友好的语气，鼓励用户
6. 在回答末尾列出参考的资料编号
"""
    return rag_prompt


def get_response(user_query, history):
    """获取回复（RAG方式）"""
    if st.session_state.llm is None:
        return "抱歉，未能连接到 Kimi API，请检查环境变量中的 API Key。", []
    
    try:
        # 检索相关文档
        relevant_docs = []
        if st.session_state.retriever:
            relevant_docs = st.session_state.retriever.retrieve(user_query, top_k=3)
        
        # 如果没有检索到文档，使用默认文档
        if not relevant_docs:
            relevant_docs = st.session_state.knowledge_base[:3]
        
        # 构建对话历史
        history_messages = []
        for msg in history:
            if msg["role"] == "user":
                history_messages.append(f"用户: {msg['content']}")
            elif msg["role"] == "assistant":
                history_messages.append(f"助手: {msg['content']}")
        
        history_text = "\n".join(history_messages) if history_messages else ""
        
        # 构建完整的 RAG Prompt
        rag_prompt = build_rag_prompt(user_query, relevant_docs)
        
        # 添加对话历史
        if history_text:
            rag_prompt = f"""
以下是之前的对话历史，供你参考：

{history_text}

---

{rag_prompt}
"""
        
        # 调用 LLM
        from langchain_core.messages import HumanMessage, SystemMessage
        
        system_prompt = "你是一位专业、友好的健康管理和健身教练助手。请基于提供的知识库内容回答问题，并在回答末尾标注参考的资料编号。"
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=rag_prompt)
        ]
        
        result = st.session_state.llm.invoke(messages)
        return result.content, relevant_docs
        
    except Exception as e:
        return f"抱歉，生成回答时出错了：{str(e)}", []


def init_knowledge_base():
    """初始化知识库"""
    if not st.session_state.kb_initialized:
        st.session_state.knowledge_base = get_knowledge_base()
        st.session_state.kb_initialized = True
        st.sidebar.success(f"📚 知识库已加载 ({len(st.session_state.knowledge_base)} 篇文档)")
        st.sidebar.write(f"分类：{', '.join(get_categories())}")


def init_llm():
    """初始化 LLM"""
    if st.session_state.llm is None:
        st.session_state.llm = create_kimi_llm()


# 侧边栏：系统信息
st.sidebar.subheader("📊 系统信息")
st.sidebar.info(
    "基于 RAG 技术的健康管理和健身教练智能问答系统\n\n"
    "使用 Kimi 2.6 (moonshot-v1-32k) 模型\n\n"
    "RAG 方式：向量检索 + 文档组合进 Prompt"
)

# 侧边栏：知识库信息
st.sidebar.subheader("📚 权威知识库")
if st.session_state.kb_initialized:
    st.sidebar.write(f"✅ 已加载 {len(st.session_state.knowledge_base)} 篇权威文档")
    st.sidebar.write(f"📂 涵盖 {len(get_categories())} 个分类")
    
    # 显示权威来源
    st.sidebar.subheader("🏛️ 权威来源")
    sources = get_sources()
    for source, info in sources.items():
        st.sidebar.write(f"- **{source}**")
        st.sidebar.caption(f"  {info['type']} ({info['count']}篇)")
else:
    st.sidebar.write("知识库初始化中...")

# 侧边栏：清空对话
if st.sidebar.button("🗑️ 清空对话"):
    st.session_state.messages = []
    st.success("对话已清空！")

# 初始化知识库、检索器和 LLM
init_knowledge_base()
init_retriever()
init_llm()

# 主界面：显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
if prompt := st.chat_input("请输入您的健康或健身问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("正在思考中..."):
        answer, context_docs = get_response(prompt, st.session_state.messages[:-1])
        
        with st.chat_message("assistant"):
            st.markdown(answer)
            
            # 显示参考文档（权威来源）
            if context_docs:
                with st.expander("📚 参考资料（权威来源）"):
                    for i, doc in enumerate(context_docs, 1):
                        sim = doc.get('similarity', 0)
                        source = doc.get('source', '未知来源')
                        source_type = doc.get('source_type', '')
                        source_url = doc.get('source_url', '')
                        
                        st.markdown(f"**资料 {i}: {doc['title']}** (相关度: {sim:.2f})")
                        st.caption(f"🏛️ 来源: {source} | {source_type}")
                        if source_url:
                            st.caption(f"🔗 [查看原文]({source_url})")
                        st.caption(doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'])
                        st.markdown("---")
        
        st.session_state.messages.append({"role": "assistant", "content": answer})


st.markdown("---")
st.caption("💡 提示：系统使用 RAG 技术，通过向量检索找到最相关的知识库文档，组合进 Prompt 后基于 Kimi 2.6 模型提供专业健康与健身建议。")
