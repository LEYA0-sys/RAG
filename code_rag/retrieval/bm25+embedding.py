from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import json
import jieba
import re
import os
from nltk import sent_tokenize
import torch

# -------------------- 配置项 --------------------
DOC_EMBED_CACHE_PATH = "cache/doc_embeddings.npy"
BM25_TOP_N = 50      # BM25候选数
TOP_K = 10           # 最终返回前K条
THRESHOLD = 0.75     # 相似度判定阈值
SAVE_PATH = "./retrieval_result/hybrid.json"

# -------------------- 加载模型与数据 --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
sentence_model = SentenceTransformer("BAAI/bge-large-zh-v1.5", device=device)

with open(r"D:\desktop\code\QA2.json", 'r', encoding='utf-8') as f:
    test_data = json.load(f)

with open(r"D:\desktop\code\cache\merged_chunks.json", 'r', encoding='utf-8') as f:
    all_docs = json.load(f)

documents = [doc["content"] for doc in all_docs]
tokenized_corpus = [list(jieba.cut(doc)) for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

# -------------------- 文档嵌入缓存 --------------------
if os.path.exists(DOC_EMBED_CACHE_PATH):
    doc_embeddings = np.load(DOC_EMBED_CACHE_PATH)
else:
    doc_embeddings = np.array([
        sentence_model.encode(doc, convert_to_numpy=True) for doc in tqdm(documents, desc="Encoding documents")
    ])
    os.makedirs(os.path.dirname(DOC_EMBED_CACHE_PATH), exist_ok=True)
    np.save(DOC_EMBED_CACHE_PATH, doc_embeddings)

# -------------------- 工具函数 --------------------
def calculate_embedding(text):
    return sentence_model.encode(text, convert_to_numpy=True)

def normalize_page_num(page_num):
    if isinstance(page_num, str):
        return re.sub(r'[第页\s]', '', page_num).strip()
    return str(page_num)

def is_relevant(doc_content, correct_answer, threshold=THRESHOLD):
    if correct_answer in doc_content:
        return True
    doc_sentences = sent_tokenize(doc_content)
    if not doc_sentences:
        return False
    ans_vec = calculate_embedding(correct_answer)
    for sent in doc_sentences:
        sent_vec = calculate_embedding(sent)
        if cosine_similarity([sent_vec], [ans_vec])[0][0] >= threshold:
            return True
    return False

def calculate_precision(docs, correct_answer):
    relevant = [doc for doc in docs if is_relevant(doc["page_content"], correct_answer)]
    return len(relevant) / len(docs) if docs else 0

def calculate_recall(docs, correct_answer):
    return int(any(is_relevant(doc["page_content"], correct_answer) for doc in docs))

def calculate_mrr(docs, correct_answer):
    for i, doc in enumerate(docs, start=1):
        if is_relevant(doc["page_content"], correct_answer):
            return 1 / i
    return 0.0

def calculate_hit_at_k(docs, correct_answer):
    return int(any(is_relevant(doc["page_content"], correct_answer) for doc in docs))

def calculate_page_match(docs, correct_source_file, correct_page_num):
    correct_page_num = normalize_page_num(correct_page_num)
    for doc in docs:
        meta = doc["metadata"]
        
        if os.path.splitext(meta.get('source_file', ''))[0] == correct_source_file and \
           normalize_page_num(meta.get('page_num')) == correct_page_num:
            return 1
    return 0

def calculate_page_mrr(docs, correct_source_file, correct_page_num):
    correct_page_num = normalize_page_num(correct_page_num)
    for i, doc in enumerate(docs, start=1):
        meta = doc["metadata"]
        if os.path.splitext(meta.get('source_file', ''))[0] == correct_source_file and \
           normalize_page_num(meta.get('page_num')) == correct_page_num:
            return 1 / i
    return 0.0

# -------------------- 主评估函数 --------------------
def test_retrieval_effectiveness_hybrid():
    results_data = []
    precision_scores, recall_scores, mrr_scores = [], [], []
    hit_scores, page_match_scores, page_mrr_scores = [], [], []

    for item in tqdm(test_data, desc="Evaluating"):
        query = item["question"]
        correct_answer = item["answer"]
        correct_source_file = item["source_file"]
        correct_page_num = normalize_page_num(item["page_num"])

        # BM25 召回 top N
        tokenized_query = list(jieba.cut(query))
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:BM25_TOP_N]

        # 向量重排 top K
        query_vec = calculate_embedding(query)
        candidate_vecs = doc_embeddings[bm25_top_indices]
        similarity_scores = cosine_similarity([query_vec], candidate_vecs)[0]
        sorted_indices = np.argsort(similarity_scores)[::-1][:TOP_K]
        final_indices = [bm25_top_indices[i] for i in sorted_indices]

        retrieved_docs = [
            {
                "page_content": all_docs[i]["content"],
                "metadata": {
                    "source_file": all_docs[i].get("metadata", {}).get("source_file", ""),
                    "page_num": all_docs[i].get("metadata", {}).get("page_num", "")
                }
            }
            for i in final_indices
        ]

        precision = calculate_precision(retrieved_docs, correct_answer)
        recall = calculate_recall(retrieved_docs, correct_answer)
        mrr = calculate_mrr(retrieved_docs, correct_answer)
        hit = calculate_hit_at_k(retrieved_docs, correct_answer)
        page_match = calculate_page_match(retrieved_docs, correct_source_file, correct_page_num)
        page_mrr = calculate_page_mrr(retrieved_docs, correct_source_file, correct_page_num)

        precision_scores.append(precision)
        recall_scores.append(recall)
        mrr_scores.append(mrr)
        hit_scores.append(hit)
        page_match_scores.append(page_match)
        page_mrr_scores.append(page_mrr)

        results_data.append({
            "query": query,
            "correct_answer": correct_answer,
            "correct_source_file": correct_source_file,
            "correct_page_num": correct_page_num,
            "retrieved_documents": [
                {
                    "content": doc['page_content'],
                    "metadata": doc['metadata'],
                    "is_relevant": is_relevant(doc['page_content'], correct_answer)
                } for doc in retrieved_docs
            ],
            "precision": precision,
            "recall": recall,
            "mrr": mrr,
            "hit@k": hit,
            "page_match": page_match,
            "page_mrr": page_mrr
        })

    # 平均指标
    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    avg_mrr = sum(mrr_scores) / len(mrr_scores)
    avg_hit = sum(hit_scores) / len(hit_scores)
    avg_page_match = sum(page_match_scores) / len(page_match_scores)
    avg_page_mrr = sum(page_mrr_scores) / len(page_mrr_scores)

    print(f"\n--- Hybrid Retrieval Evaluation ---")
    print(f"Average Precision@{TOP_K}: {avg_precision:.4f}")
    print(f"Average Recall@{TOP_K}: {avg_recall:.4f}")
    print(f"Average MRR@{TOP_K}: {avg_mrr:.4f}")
    print(f"Average Hit@{TOP_K}: {avg_hit:.4f}")
    print(f"Average Page Match: {avg_page_match:.4f}")
    print(f"Average Page MRR: {avg_page_mrr:.4f}")

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "average_mrr": avg_mrr,
            "average_hit": avg_hit,
            "average_page_match": avg_page_match,
            "average_page_mrr": avg_page_mrr,
            "results": results_data
        }, f, ensure_ascii=False, indent=2)

    print(f"Saved to: {SAVE_PATH}")

# -------------------- 运行 --------------------
if __name__ == "__main__":
    test_retrieval_effectiveness_hybrid()
