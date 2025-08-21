import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
from rank_bm25 import BM25Okapi
import os
import re

#加载QA1，并提取页码信息
qa_file_path = '../QA对/QA1.json'
with open(qa_file_path, 'r', encoding='utf-8') as f:
    qa_data = json.load(f)

answers = [item['answer'] for item in qa_data]
pages = [re.sub(r"\D", "", str(item.get('页码', '未指定'))) for item in qa_data]  # 只保留数字
queries = [item['question'] for item in qa_data]
k = 50 #返回检索结果的数量

def jieba_tokenizer(text):
    return list(jieba.cut(text))

# 定义简单的RAG检索函数
def rag_search(query, k=k):
    """
    使用vector_db对输入query进行检索，返回前k条结果。
    """
    retriever = vector_db.as_retriever(search_kwargs={"k": k})
    results = retriever.invoke(query)
    return results

output_dir = '../outputs_打分版/简单检索'
os.makedirs(output_dir, exist_ok=True)
output_file_path = os.path.join(output_dir, f'rag_results_easy_k={k}.json')

recall_list = []
output_data = []
accuracy_list = []

# 新增：对每个问题进行检索，输出内容和页码，并比对准确率
for idx, (query, answer, correct_page) in enumerate(zip(queries, answers, pages)):
    results = rag_search(query, k=k)
    retrieved_documents = [
        {
            "uuid":doc.metadata.get("uuid", "未指定"),
            "source_file": doc.metadata.get("source_file", ""),
            "content": doc.page_content,
            "page_num": str(doc.metadata.get("page_num", "未指定"))
        }
        for doc in results
    ]
    retrieved_pages = [doc["page_num"] for doc in retrieved_documents]
    # 检查是否有检索到的页码与标准答案页码一致
    recall = 1 if correct_page in retrieved_pages else 0
    recall_list.append(recall)
    # 计算准确率（查找的页码与正确页码完全一致的比例）
    accuracy = sum([1 for p in retrieved_pages if p == correct_page]) / len(retrieved_pages) if retrieved_pages else 0.0
    accuracy_list.append(accuracy)
    output_data.append({
        "query": query,
        "answer": answer,
        "correct_page": correct_page,
        "recall": recall,
        "accuracy": accuracy,
        "retrieved_documents": retrieved_documents
    })

with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0.0
print(f"检索结果已保存到 {output_file_path}")
print(f"平均召回率: {avg_recall}")
avg_accuracy = sum(accuracy_list) / len(accuracy_list) if accuracy_list else 0.0
print(f"平均准确率: {avg_accuracy}")
