import os
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import json
import jieba
import re
import random
from nltk import sent_tokenize

# 读取数据
with open(r'D:\desktop\code\QA_ds.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)
with open(r'D:\desktop\code\cache\semantic_merged_chunks.json', 'r', encoding='utf-8') as f:
    all_docs = json.load(f)  

documents = [doc["content"] for doc in all_docs]
tokenized_corpus = [list(jieba.cut(doc)) for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

sentence_model = SentenceTransformer("BAAI/bge-large-zh-v1.5")

def calculate_embedding(text):
    return sentence_model.encode(text, convert_to_tensor=False)

def normalize_page_num(page_num):
    if page_num is None:
        return ""
    if isinstance(page_num, str):
        return re.sub(r'[第页\s]', '', page_num).strip()
    return str(page_num)

def is_relevant(doc_content, correct_answer, threshold=0.7):
    if correct_answer in doc_content:
        return True
    doc_sentences = sent_tokenize(doc_content)
    if not doc_sentences:
        return False
    ans_vec = calculate_embedding(correct_answer)
    max_similarity = 0
    for sent in doc_sentences:
        sent_vec = calculate_embedding(sent)
        similarity = cosine_similarity([sent_vec], [ans_vec])[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
        if max_similarity >= threshold:
            break
    return bool(max_similarity >= threshold)  # 确保返回Python bool类型

def calculate_precision(docs, correct_answer, threshold=0.7):
    relevant = [doc for doc in docs if is_relevant(doc["page_content"], correct_answer, threshold)]
    return len(relevant) / len(docs) if docs else 0

def calculate_recall(docs, correct_answer, threshold=0.7):
    return int(any(is_relevant(doc["page_content"], correct_answer, threshold) for doc in docs))

def calculate_mrr(docs, correct_answer, threshold=0.7):
    for i, doc in enumerate(docs, start=1):
        if is_relevant(doc["page_content"], correct_answer, threshold):
            return 1 / i
    return 0.0

def calculate_hit_at_k(docs, correct_answer, threshold=0.7):
    return int(any(is_relevant(doc["page_content"], correct_answer, threshold) for doc in docs))

def calculate_page_match(docs, correct_source_file, correct_page_num):
    correct_page_num = normalize_page_num(correct_page_num)
    for doc in docs:
        meta = doc["metadata"]
        # source_file_no_ext = os.path.splitext(meta.get('source_file', ''))[0]
        correct_source_file = os.path.splitext(correct_source_file)[0]
        if (meta.get('source_file', '') == correct_source_file and
            normalize_page_num(meta.get('page_num')) == correct_page_num):
            return 1
    return 0

def calculate_page_mrr(docs, correct_source_file, correct_page_num):
    correct_page_num = normalize_page_num(correct_page_num)
    for i, doc in enumerate(docs, start=1):
        meta = doc["metadata"]
        # source_file_no_ext = os.path.splitext(meta.get('source_file', ''))[0]
        correct_source_file = os.path.splitext(correct_source_file)[0]
        if (meta.get('source_file', '') == correct_source_file and
            normalize_page_num(meta.get('page_num')) == correct_page_num):
            return 1 / i
    return 0.0

def test_retrieval_effectiveness_bm25(test_data, k_retrieval=20, threshold=0.7, verbose=False, save_path="./retrieval_result/QA_ds/bm25(semantic).json"):
    precision_scores = []
    recall_scores = []
    mrr_scores = []
    hit_scores = []
    page_match_scores = []
    page_mrr_scores = []
    results_data = []

    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for item in tqdm(test_data, total=len(test_data), desc="Evaluating"):
        query = item["question"]
        correct_answer = item["answer"]
        correct_source_file = item["source_file"]
        correct_page_num = normalize_page_num(item["page_num"])

        tokenized_query = list(jieba.cut(query))
        doc_scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[::-1][:k_retrieval]

        retrieved_docs = [
            {
                "page_content": all_docs[i]["content"],
                "metadata": {
                    "source_file": all_docs[i].get("metadata", {}).get("source_file", ""),
                    "page_num": all_docs[i].get("metadata", {}).get("page_num", "")
                }
            }
            for i in top_indices
        ]

        precision = calculate_precision(retrieved_docs, correct_answer, threshold)
        recall = calculate_recall(retrieved_docs, correct_answer, threshold)
        mrr = calculate_mrr(retrieved_docs, correct_answer, threshold)
        hit = calculate_hit_at_k(retrieved_docs, correct_answer, threshold)
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
                    "is_relevant": bool(is_relevant(doc['page_content'], correct_answer, threshold))  # 转成bool
                } for doc in retrieved_docs
            ],
            "precision": precision,
            "recall": recall,
            "mrr": mrr,
            "hit@k": hit,
            "page_match": page_match,
            "page_mrr": page_mrr
        })

        if verbose:
            print(f"\nQuery: {query}")
            print(f"Correct Answer: {correct_answer}")
            print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | MRR: {mrr:.4f} | Hit@{k_retrieval}: {hit:.4f} | Page Match: {page_match:.4f} | Page MRR: {page_mrr:.4f}")

    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0
    avg_hit = sum(hit_scores) / len(hit_scores) if hit_scores else 0
    avg_page_match = sum(page_match_scores) / len(page_match_scores) if page_match_scores else 0
    avg_page_mrr = sum(page_mrr_scores) / len(page_mrr_scores) if page_mrr_scores else 0

    print(f"\n--- Overall Evaluation (BM25) ---")
    print(f"Average Precision@{k_retrieval}: {avg_precision:.4f}")
    print(f"Average Recall@{k_retrieval}: {avg_recall:.4f}")
    print(f"Average MRR@{k_retrieval}: {avg_mrr:.4f}")
    print(f"Average Hit@{k_retrieval}: {avg_hit:.4f}")
    print(f"Average Page Match@{k_retrieval}: {avg_page_match:.4f}")
    print(f"Average Page MRR@{k_retrieval}: {avg_page_mrr:.4f}")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "average_mrr": avg_mrr,
            "average_hit": avg_hit,
            "average_page_match": avg_page_match,
            "average_page_mrr": avg_page_mrr,
            "results": results_data
        }, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {save_path}")


random.seed(42)
sampled_test_data = random.sample(test_data, 100)
test_retrieval_effectiveness_bm25(sampled_test_data, k_retrieval=20)
# test_retrieval_effectiveness_bm25(test_data, k_retrieval=20)