#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从结构化知识库构建 LoRA 微调数据集
输出格式：Alpaca instruction format (instruction, input, output)
"""

import sys, json, random
sys.path.insert(0, 'src')

from knowledge_base_structured import KNOWLEDGE_BASE_DOCS, get_total_sections

random.seed(42)

def build_alpaca_dataset(docs):
    """生成 Alpaca 格式的 Q&A 数据集"""
    data = []

    for doc in docs:
        title = doc['title']
        category = doc['category']
        source = doc['source']
        sections = doc.get('sections', [])
        full_content = doc.get('content', '')

        # 1. 对每个 section 生成一个问答对
        for sec in sections:
            heading = sec['heading']
            content = sec['content'][:800]
            tags = ','.join(sec.get('tags', []))

            # 问法1：直接询问该小节内容
            data.append({
                "instruction": f"请从专业角度回答以下健康/健身问题",
                "input": f"关于{heading}，有什么建议？",
                "output": content,
                "source": source,
                "category": category
            })

        # 2. 生成摘要类问答
        key_points = doc.get('key_points', [])
        if key_points and len(sections) >= 3:
            # 选取前3个 section 的内容拼接
            summary_content = '\n'.join(s['content'][:300] for s in sections[:3])
            data.append({
                "instruction": f"你是健康管理和健身教练专家，请根据权威资料回答",
                "input": f"请总结{title}的核心要点",
                "output": '\n'.join(f"{i+1}. {kp}" for i, kp in enumerate(key_points)) + f"\n\n详细说明：\n{summary_content}",
                "source": source,
                "category": category
            })

        # 3. 分类检索类问答
        data.append({
            "instruction": f"从权威知识库中检索并回答健康相关问题",
            "input": f"{category}方面有哪些权威建议？",
            "output": f"根据{title}等权威资料，{category}方面的核心建议包括：\n" + \
                      '\n'.join(f"- {s['heading']}：{s['content'][:150]}..." for s in sections[:4]),
            "source": source,
            "category": category
        })

        # 4. 对比/关联类
        if len(sections) >= 2:
            s1, s2 = sections[0], sections[1]
            data.append({
                "instruction": "请基于权威知识库进行对比分析",
                "input": f"请对比分析「{s1['heading']}」和「{s2['heading']}」",
                "output": f"【{s1['heading']}】{s1['content'][:400]}\n\n" + \
                          f"【{s2['heading']}】{s2['content'][:400]}\n\n" + \
                          f"综合分析：这两个方面共同构成了{category}领域的重要知识体系，" + \
                          f"来自{source}的权威建议为实际应用提供了科学依据。",
                "source": source,
                "category": category
            })

    return data


if __name__ == "__main__":
    print(f"📊 知识库: {len(KNOWLEDGE_BASE_DOCS)} 篇文档, {get_total_sections()} sections")

    # 构建数据集
    dataset = build_alpaca_dataset(KNOWLEDGE_BASE_DOCS)
    random.shuffle(dataset)

    # 划分训练/验证集 (85/15)
    split = int(len(dataset) * 0.85)
    train_data = dataset[:split]
    val_data = dataset[split:]

    # 保存为 JSON
    for name, data in [("train", train_data), ("val", val_data)]:
        path = f"data/finetune_{name}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  {path}: {len(data)} 条")

    print(f"\n✅ 数据集构建完成")
    print(f"   训练集: {len(train_data)} 条")
    print(f"   验证集: {len(val_data)} 条")
    print(f"   总条数: {len(dataset)} 条")
    print(f"\n💡 提示: 可用于 Qwen2.5 或 Llama3 LoRA 微调")
