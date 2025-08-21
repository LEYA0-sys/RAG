import numpy as np
from rank_bm25 import BM25Okapi
import jieba
import json
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from functools import partial
from sklearn.metrics.pairwise import cosine_similarity

def tokenize(text):
    """使用jieba进行中文分词"""
    return list(jieba.cut(text))

def create_bm25_index(documents):
    """创建BM25索引"""
    tokenized_docs = [tokenize(doc.page_content) for doc in documents]
    return BM25Okapi(tokenized_docs)

def hybrid_search(query, bm25_index, documents, embeddings, embedding_model, k=20, alpha=0.5):
    """混合检索实现"""
    # BM25检索
    tokenized_query = tokenize(query)
    bm25_scores = bm25_index.get_scores(tokenized_query)
    
    # Embedding检索
    query_embedding = embedding_model.embed_query(query)
    # 确保embeddings是numpy数组
    embeddings = np.array(embeddings)
    query_embedding = np.array(query_embedding).reshape(1, -1)
    
    # 计算余弦相似度
    embedding_scores = cosine_similarity(query_embedding, embeddings)[0]
    
    # 归一化分数
    bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-6)
    embedding_scores = (embedding_scores - embedding_scores.min()) / (embedding_scores.max() - embedding_scores.min() + 1e-6)
    
    # 混合分数
    hybrid_scores = alpha * bm25_scores + (1 - alpha) * embedding_scores
    
    # 获取top-k结果
    top_k_indices = np.argsort(hybrid_scores)[-k:][::-1]
    
    # 构建结果
    results = []
    for idx in top_k_indices:
        doc = documents[idx]
        results.append({
            "uuid": doc.metadata.get("uuid", "未指定"),
            "source_file": doc.metadata.get("source_file", ""),
            "source_content": doc.page_content,
            "page_num": str(doc.metadata.get("page_num", "未指定")),
            "score": float(hybrid_scores[idx]),
            "bm25_score": float(bm25_scores[idx]),
            "embedding_score": float(embedding_scores[idx])
        })
    return results

def process_hybrid_search(query, bm25_index, documents, embeddings, embedding_model, k=20):
    """处理单个查询的混合检索"""
    try:
        results = hybrid_search(
            query=query,
            bm25_index=bm25_index,
            documents=documents,
            embeddings=embeddings,
            embedding_model=embedding_model,
            k=k
        )
        return {
            'query': query,
            'retrieved_documents': results
        }
    except Exception as e:
        print(f"处理查询时出错: {str(e)}")
        return {
            'query': query,
            'retrieved_documents': [],
            'error': str(e)
        }

def parallel_hybrid_search(queries, documents, embeddings, embedding_model, k=20):
    """并行处理多个查询的混合检索"""
    # 创建BM25索引
    bm25_index = create_bm25_index(documents)
    
    # 确保embeddings是numpy数组
    embeddings = np.array(embeddings)
    
    # 使用线程池并行处理所有查询
    with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        # 使用partial固定参数
        process_func = partial(
            process_hybrid_search,
            bm25_index=bm25_index,
            documents=documents,
            embeddings=embeddings,
            embedding_model=embedding_model,
            k=k
        )
        results = list(executor.map(process_func, queries))
    
    return results

def main():
    # 加载QA4.json中的问题
    qa_file_path = '../QA对/QA4.json'
    with open(qa_file_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    # 提取问题和答案
    queries = [item['question'] for item in qa_data]
    answers = [item['answer'] for item in qa_data]
    pages = [item['page_num'] for item in qa_data]
    
    # 获取所有文档
    all_docs = list(vector_db.docstore._dict.values())
    
    # 获取所有文档的embeddings
    all_texts = [doc.page_content for doc in all_docs]
    embeddings = embedding_model.embed_documents(all_texts)
    
    # 执行混合检索
    results = parallel_hybrid_search(
        queries=queries,
        documents=all_docs,
        embeddings=embeddings,
        embedding_model=embedding_model,
        k=20
    )
    
    # 计算评估指标
    recall_list = []
    mrr_list = []
    output_data = []
    
    for idx, (result, answer, correct_page) in enumerate(zip(results, answers, pages)):
        retrieved_pages = [doc["page_num"] for doc in result['retrieved_documents']]
        
        # 计算recall
        recall = 1 if correct_page in retrieved_pages else 0
        recall_list.append(recall)
        
        # 计算MRR
        rank = 0
        for i, p in enumerate(retrieved_pages):
            if p == correct_page:
                rank = i + 1
                break
        mrr = 1.0 / rank if rank > 0 else 0.0
        mrr_list.append(mrr)
        
        # 构建输出数据
        output_data.append({
            "query": result['query'],
            "answer": answer,
            "correct_page": correct_page,
            "recall": recall,
            "mrr": mrr,
            "retrieved_documents": result['retrieved_documents']
        })
    
    # 计算平均指标
    avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0.0
    avg_mrr = sum(mrr_list) / len(mrr_list) if mrr_list else 0.0
    
    # 保存结果
    output_dir = '../outputs/outputs_4.0/混合检索'
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'hybrid_results_k=20.json')
    
    output = {
        "summary": {
            "avg_recall": avg_recall,
            "avg_mrr": avg_mrr
        },
        "results": output_data
    }
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    
    print(f"混合检索结果已保存到 {output_file_path}")
    print(f"平均Recall: {avg_recall:.4f}")
    print(f"平均MRR: {avg_mrr:.4f}")

if __name__ == "__main__":
    main() 