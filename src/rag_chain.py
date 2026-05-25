# -*- coding: utf-8 -*-
"""
RAG 链实现模块
包含问答链和对话管理
"""
import os
from typing import List, Dict, Any, Optional
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI


class HealthFitnessAssistant:
    """健康管理和健身教练助手"""

    def __init__(self, retriever, llm=None):
        """
        初始化助手

        Args:
            retriever: 检索器对象
            llm: 大语言模型对象（可选，默认使用占位符）
        """
        self.retriever = retriever
        self.llm = llm
        self.store: Dict[str, BaseChatMessageHistory] = {}
        self.rag_chain = None

        # 初始化链
        llm_for_prompt = llm if llm else self._get_demo_llm()
        self._initialize_chains(llm_for_prompt)

    def _get_demo_llm(self):
        """获取演示用的 LLM（实际项目中应该替换为真实模型"""
        from langchain_core.language_models import BaseChatModel
        from langchain_core.outputs import ChatResult, ChatGeneration
        from langchain_core.messages import AIMessage

        class DemoLLM(BaseChatModel):
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                responses = {
                    "你好": "你好！我是你的健康管理和健身教练助手。有什么可以帮助你的吗？",
                    "减肥": "关于减肥，我建议你注意饮食控制加上规律运动，有氧运动配合力量训练效果最好。",
                    "饮食": "健康饮食建议保持营养均衡，多吃蔬菜水果，适量蛋白质，控制热量摄入。",
                    "健身": "健身计划需要循序渐进，建议每周3-5次，每次30分钟以上，注意热身和拉伸。",
                    "default": "这是一个很好的问题！根据健康管理的一般建议：保持规律作息，均衡饮食，适度运动。"
                }
                query = str(messages[-1].content)
                answer = responses.get("default")
                for key in responses:
                    if key in query:
                        answer = responses[key]
                        break
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=answer))])

            @property
            def _llm_type(self):
                return "demo"

        return DemoLLM()

    def _initialize_chains(self, llm):
        """初始化 RAG 链"""
        # 1. 上下文感知的检索提示词
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "根据对话历史和用户问题，生成一个独立的、可以理解的搜索查询。"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # 2. 问答提示词
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位专业的健康管理和健身教练助手。请使用以下检索到的健康和健身知识回答用户的问题。
如果你不知道答案，请诚实地说不知道，不要编造答案。
回答要友好、专业、有帮助。

{context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # 3. 构建链
        history_aware_retriever = create_history_aware_retriever(
            llm, self.retriever, contextualize_q_prompt
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # 4. 添加对话历史
        self.rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """获取会话历史"""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def chat(self, user_input: str, session_id: str = "default") -> Dict[str, Any]:
        """
        发送消息并获取回复

        Args:
            user_input: 用户输入
            session_id: 会话 ID

        Returns:
            包含回答和上下文的字典
        """
        if self.rag_chain is None:
            raise ValueError("RAG 链未初始化！")

        result = self.rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )

        return {
            "answer": result.get("answer", ""),
            "context": result.get("context", [])
        }

    def clear_history(self, session_id: str = "default"):
        """清除会话历史"""
        if session_id in self.store:
            del self.store[session_id]


def create_kimi_llm(api_key: Optional[str] = None):
    """
    创建 Kimi API 的 LLM 实例

    Args:
        api_key: Kimi API Key，若不提供则从环境变量获取

    Returns:
        ChatOpenAI 实例
    """
    if api_key is None:
        # 尝试多个可能的环境变量名称
        api_key = os.getenv("kimi_general_api_key") or \
                  os.getenv("KIMI_API_KEY") or \
                  os.getenv("MOONSHOT_API_KEY")

    if not api_key:
        raise ValueError("请提供 Kimi API Key 环境变量")

    # Kimi API 兼容 OpenAI 接口
    llm = ChatOpenAI(
        model="moonshot-v1-8k",
        temperature=0.7,
        api_key=api_key,
        base_url="https://api.moonshot.cn/v1",
    )
    return llm


def create_assistant(retriever, llm=None):
    """
    创建健康管理和健身教练助手

    Args:
        retriever: 检索器
        llm: LLM 模型（可选）

    Returns:
        HealthFitnessAssistant 实例
    """
    return HealthFitnessAssistant(retriever, llm)
