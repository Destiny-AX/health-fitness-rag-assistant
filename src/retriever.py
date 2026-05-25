# -*- coding: utf-8 -*-
"""
统一检索模块 — BM25 + jieba中文分词 + CrossEncoder重排序
支持 section-level 和 doc-level 检索，有完整降级策略
"""

import numpy as np


class JiebaTokenizer:
    """jieba 中文分词适配器"""

    def __init__(self):
        try:
            import jieba
            self.jieba = jieba
            self.jieba.setLogLevel(20)
            self._available = True
        except ImportError:
            self._available = False

    def cut(self, text):
        if self._available:
            return list(self.jieba.cut_for_search(text))
        return list(text)


class BM25Retriever:
    """BM25 检索器 — 基于 rank-bm25"""

    def __init__(self, docs):
        self.docs = docs
        self.tokenizer = JiebaTokenizer()
        self._available = False
        self._corpus = []
        self._build_index()

    def _tokenize(self, text):
        return " ".join(self.tokenizer.cut(text))

    def _build_index(self):
        try:
            from rank_bm25 import BM25Okapi
            texts = []
            for doc in self.docs:
                # 结合标题和内容用于检索
                text = f"{doc.get('title', '')} {doc.get('category', '')} {doc.get('content', '')}"
                texts.append(self._tokenize(text))
            self._corpus = [t.split() for t in texts]
            self._bm25 = BM25Okapi(self._corpus)
            self._available = True
        except ImportError:
            self._bm25 = None

    def retrieve(self, query, top_k=10):
        if not self._available:
            return self._keyword_match(query, top_k)

        tokens = self._tokenize(query).split()
        if not tokens:
            return self._keyword_match(query, top_k)

        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self.docs[idx].copy()
                doc['score'] = float(scores[idx])
                results.append(doc)
        return results

    def _keyword_match(self, query, top_k=10):
        """关键词匹配回退"""
        query_words = set(self.tokenizer.cut(query))
        scores = []
        for i, doc in enumerate(self.docs):
            doc_text = f"{doc.get('title','')} {doc.get('content','')}"
            doc_words = set(self.tokenizer.cut(doc_text))
            score = len(query_words & doc_words)
            scores.append((score, i))
        scores.sort(reverse=True)
        return [self.docs[idx].copy() for s, idx in scores[:top_k] if s > 0]


class Reranker:
    """CrossEncoder 重排序器"""

    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self._available = False
        self._try_load()

    def _try_load(self):
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            self._available = True
        except (ImportError, OSError):
            self._model = None

    def rerank(self, query, candidates, top_k=3):
        if not self._available or not candidates:
            return candidates[:top_k]

        pairs = [[query, c.get('content', '')] for c in candidates]
        scores = self._model.predict(pairs)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )
        results = []
        for doc, score in ranked[:top_k]:
            doc = doc.copy()
            doc['score'] = float(score)
            results.append(doc)
        return results


class HybridRetriever:
    """混合检索器 — BM25召回 + CrossEncoder重排"""

    def __init__(self, docs, enable_rerank=True):
        self.bm25 = BM25Retriever(docs)
        self.reranker = Reranker() if enable_rerank else None
        self.docs = docs

    def retrieve(self, query, top_k=3, recall_k=10):
        # 第一步：BM25 召回
        candidates = self.bm25.retrieve(query, top_k=recall_k)

        if not candidates:
            candidates = self.docs[:top_k]

        # 第二步：CrossEncoder 重排序
        if self.reranker and len(candidates) > top_k:
            candidates = self.reranker.rerank(query, candidates, top_k)

        # 统一字段名
        for doc in candidates:
            if 'score' in doc and 'similarity' not in doc:
                doc['similarity'] = doc['score']

        return candidates[:top_k]


def expand_query(llm, query):
    """用 LLM 扩展查询关键词"""
    if llm is None:
        return query
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        prompt = f"""请将以下健身或健康相关问题扩展为3-5个关键词或短语，用逗号分隔。只输出关键词，不要解释。
       问题: {query}"""
        result = llm.invoke([HumanMessage(content=prompt)])
        expanded = result.content.strip()
        # 合并原查询和扩展词
        return f"{query} {expanded}"
    except Exception:
        return query
