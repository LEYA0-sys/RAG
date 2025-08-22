from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import json
import jieba

# 加载测试数据
with open(r'D:\desktop\code\QA3.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

test_queries = [item["question"] for item in test_data]
correct_answers = [item["answer"] for item in test_data]

# 准备文档库（从 FAISS 数据改成纯文本列表）
with open(r'D:\desktop\code\cache\merged_chunks.json', 'r', encoding='utf-8') as f:
    all_docs = json.load(f)  # 格式： [{"uuid": "...", "content": "..."}]

documents = [doc["content"] for doc in all_docs]

# 建 BM25 索引
tokenized_corpus = [list(jieba.cut(doc)) for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

# 初始化模型
sentence_model = SentenceTransformer("BAAI/bge-base-zh-v1.5")
reranker = CrossEncoder('BAAI/bge-reranker-large', device='cpu')

def calculate_embedding(text):
    return sentence_model.encode(text, convert_to_tensor=False)

def is_relevant(doc, correct_answer, threshold=0.75):
    doc_vec = calculate_embedding(doc)
    ans_vec = calculate_embedding(correct_answer)
    similarity = cosine_similarity([doc_vec], [ans_vec])[0][0]
    return similarity >= threshold

def calculate_precision(docs, correct_answer, threshold=0.75):
    relevant = [doc for doc in docs if is_relevant(doc, correct_answer, threshold)]
    return len(relevant) / len(docs) if docs else 0

def calculate_recall(docs, correct_answer, threshold=0.75):
    return int(any(is_relevant(doc, correct_answer, threshold) for doc in docs))

def calculate_mrr(docs, correct_answer, threshold=0.75):
    for i, doc in enumerate(docs, start=1):
        if is_relevant(doc, correct_answer, threshold):
            return 1 / i
    return 0.0

def calculate_hit_at_k(docs, correct_answer, threshold=0.75):
    return int(any(is_relevant(doc, correct_answer, threshold) for doc in docs))

def rerank_documents(query, docs, reranker, top_k=10):
    pairs = [[query, doc] for doc in docs]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in reranked[:top_k]]

def test_retrieval_effectiveness_bm25(test_queries, correct_answers, k_retrieval=50, k_rerank=10, threshold=0.75, verbose=False, save_path="./retrieval_result/bm25_rerank_test.json"):
    precision_scores = []
    recall_scores = []
    mrr_scores = []
    hit_scores = []

    results_data = []

    for query, correct_answer in tqdm(zip(test_queries, correct_answers), total=len(test_queries), desc="Evaluating"):
        # BM25 检索
        tokenized_query = list(jieba.cut(query))
        doc_scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[::-1][:k_retrieval]
        retrieved_contents = [documents[i] for i in top_indices]

        # Reranker 精排
        reranked_contents = rerank_documents(query, retrieved_contents, reranker, top_k=k_rerank)

        precision = calculate_precision(reranked_contents, correct_answer, threshold)
        recall = calculate_recall(reranked_contents, correct_answer, threshold)
        mrr = calculate_mrr(reranked_contents, correct_answer, threshold)
        hit = calculate_hit_at_k(reranked_contents, correct_answer, threshold)

        precision_scores.append(precision)
        recall_scores.append(recall)
        mrr_scores.append(mrr)
        hit_scores.append(hit)

        results_data.append({
            "query": query,
            "correct_answer": correct_answer,
            "retrieved_documents": reranked_contents,
            "precision": precision,
            "recall": recall,
            "mrr": mrr,
            "hit@k": hit
        })

        if verbose:
            print(f"\nQuery: {query}")
            print(f"Correct Answer: {correct_answer}")
            print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | MRR: {mrr:.4f} | Hit@{k_rerank}: {hit}")

    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    avg_mrr = sum(mrr_scores) / len(mrr_scores)
    avg_hit = sum(hit_scores) / len(hit_scores)

    print(f"\n--- Overall Evaluation (BM25+Reranker) ---")
    print(f"Average Precision@{k_rerank}: {avg_precision:.4f}")
    print(f"Average Recall@{k_rerank}: {avg_recall:.4f}")
    print(f"Average MRR@{k_rerank}: {avg_mrr:.4f}")
    print(f"Average Hit@{k_rerank}: {avg_hit:.4f}")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "average_mrr": avg_mrr,
            "average_hit": avg_hit,
            "results": results_data
        }, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {save_path}")

# 运行 BM25+Reranker
test_retrieval_effectiveness_bm25(test_queries, correct_answers, k_retrieval=50, k_rerank=10)
