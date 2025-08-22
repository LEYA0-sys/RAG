import os
import re
import json
import jieba
import numpy as np
from tqdm import tqdm
from nltk import sent_tokenize
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from sentence_transformers import SentenceTransformer

# ------------------ 嵌入模型和向量库 ------------------ #
embeddings = HuggingFaceBgeEmbeddings(
    model_name='BAAI/bge-large-zh-v1.5',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vector_db = FAISS.load_local(
    r"output\v2\FAISS\bge_large_v1.5\faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

all_documents = list(vector_db.docstore._dict.values())

# ------------------ BM25 检索器构建 ------------------ #
tokenized_corpus = [jieba.lcut(doc.page_content) for doc in all_documents]
bm25_model = BM25Okapi(tokenized_corpus)

# ------------------ 向量模型初始化 ------------------ #
sentence_model = SentenceTransformer("BAAI/bge-large-zh-v1.5")

def calculate_embedding(text):
    return sentence_model.encode(text, convert_to_tensor=False)

# ------------------ 分数融合检索器 ------------------ #
def hybrid_score_fusion_retrieve(query, top_k=10, alpha=0.5):
    tokenized_query = jieba.lcut(query)
    bm25_scores = bm25_model.get_scores(tokenized_query)

    query_embedding = calculate_embedding(query)
    doc_embeddings = [calculate_embedding(doc.page_content) for doc in all_documents]
    vector_scores = cosine_similarity([query_embedding], doc_embeddings)[0]

    # 分数归一化
    bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.ptp(bm25_scores) + 1e-8)
    vector_norm = (vector_scores - np.min(vector_scores)) / (np.ptp(vector_scores) + 1e-8)

    # 加权融合
    final_scores = alpha * vector_norm + (1 - alpha) * bm25_norm
    top_indices = np.argsort(final_scores)[::-1][:top_k]

    return [all_documents[i] for i in top_indices]

# ------------------ 评估函数 ------------------ #
def normalize_page_num(page_num):
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
    for sent in doc_sentences:
        sent_vec = calculate_embedding(sent)
        similarity = cosine_similarity([sent_vec], [ans_vec])[0][0]
        if similarity >= threshold:
            return True
    return False

def calculate_precision(docs, correct_answer, threshold=0.7):
    relevant = [doc for doc in docs if is_relevant(doc.page_content, correct_answer, threshold)]
    return len(relevant) / len(docs) if docs else 0

def calculate_recall(docs, correct_answer, threshold=0.7):
    return int(any(is_relevant(doc.page_content, correct_answer, threshold) for doc in docs))

def calculate_mrr(docs, correct_answer, threshold=0.7):
    for i, doc in enumerate(docs, start=1):
        if is_relevant(doc.page_content, correct_answer, threshold):
            return 1 / i
    return 0.0

def calculate_page_match(docs, correct_source_file, correct_page_num):
    correct_page_num = normalize_page_num(correct_page_num)
    for doc in docs:
        meta = doc.metadata
        if (meta.get('source_file', '') == correct_source_file and 
            normalize_page_num(meta.get('page_num')) == correct_page_num):
            return 1
    return 0

def calculate_page_mrr(docs, correct_source_file, correct_page_num):
    correct_page_num = normalize_page_num(correct_page_num)
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        if (meta.get('source_file', '') == correct_source_file and 
            normalize_page_num(meta.get('page_num')) == correct_page_num):
            return 1 / i
    return 0.0

def calculate_hit_at_k(docs, correct_answer, threshold=0.7):
    return int(any(is_relevant(doc.page_content, correct_answer, threshold) for doc in docs))

# ------------------ 主评估函数 ------------------ #
def test_retrieval_effectiveness(test_data, top_k=10, alpha=0.6, threshold=0.7,
                                verbose=True, save_path="./retrieval_result/QA_ds/fusion_results.json"):
    precision_scores, recall_scores, mrr_scores = [], [], []
    hit_scores, page_match_scores, page_mrr_scores = [], [], []
    results_data = []

    for item in tqdm(test_data, desc="Evaluating"):
        query = item["question"]
        correct_answer = item["answer"]
        correct_source_file = item["source_file"]
        correct_page_num = normalize_page_num(item["page_num"])
        correct_text = item.get("correct_text", "")

        retrieved_docs = hybrid_score_fusion_retrieve(query, top_k=top_k, alpha=alpha)

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
            "correct_text": correct_text,
            "retrieved_documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "is_relevant": is_relevant(doc.page_content, correct_answer, threshold)
                } for doc in retrieved_docs
            ],
            "precision": float(precision),
            "recall": int(recall),
            "mrr": float(mrr),
            "hit@k": int(hit),
            "page_match": int(page_match),
            "page_mrr": float(page_mrr)
        })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            "average_precision": float(np.mean(precision_scores)),
            "average_recall": float(np.mean(recall_scores)),
            "average_mrr": float(np.mean(mrr_scores)),
            "average_hit": float(np.mean(hit_scores)),
            "average_page_match": float(np.mean(page_match_scores)),
            "average_page_mrr": float(np.mean(page_mrr_scores)),
            "results": results_data
        }, f, ensure_ascii=False, indent=2)

    print("\n--- Overall Evaluation ---")
    print(f"Precision: {np.mean(precision_scores):.4f}")
    print(f"Recall: {np.mean(recall_scores):.4f}")
    print(f"Hit@{top_k}: {np.mean(hit_scores):.4f}")
    print(f"MRR: {np.mean(mrr_scores):.4f}")
    print(f"Page Match: {np.mean(page_match_scores):.4f}")
    print(f"Page MRR: {np.mean(page_mrr_scores):.4f}")
    print(f"\nResults saved to {save_path}")

# ------------------ 运行 ------------------ #
with open(r'D:\desktop\code\QA_ds.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

test_retrieval_effectiveness(
    test_data,
    top_k=10,
    alpha=0.6,
    threshold=0.7,
    save_path="./retrieval_result/QA_ds/fusion_results.json"
)
