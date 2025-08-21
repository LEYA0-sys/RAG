# 添加上下文压缩使用EmbeddingsFilter，并对随机抽取10个问题进行检索与相似度打分

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, EmbeddingsFilter
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
from rank_bm25 import BM25Okapi
import random

k = 5  # 保持与其他检索一致
compressor = LLMChainExtractor.from_llm(llm)
compressor_filter = EmbeddingsFilter(embeddings=embedding_model, threshold=0.76)
compressor_retriever = ContextualCompressionRetriever(
    base_compressor=compressor_filter,
    base_retriever=vector_db.as_retriever(search_kwargs={"k": k})
)

# 加载QA对
qa_file_path = '../QA对/测试集1.json'
with open(qa_file_path, 'r', encoding='utf-8') as f:
    qa_data = json.load(f)
queries = [item['question'] for item in qa_data]
answers = [item['answer'] for item in qa_data]

def jieba_tokenizer(text):
    return list(jieba.cut(text))

output_dir = '../outputs_打分版/上下文压缩-EmbeddingsFilter'
os.makedirs(output_dir, exist_ok=True)
output_file_path = os.path.join(output_dir, f'EmbeddingsFilter_results_k={k}.json')

output_data = []

for query, answer in zip(queries, answers):
    # EmbeddingsFilter上下文压缩检索
    compressed_docs = compressor_retriever.invoke(query)
    retrieved_contents = [doc.page_content for doc in compressed_docs]
    retrieved_documents = [
        {
            "uuid": doc.metadata.get("uuid", ""),
            "source_file": doc.metadata.get("source_file", ""),
            "content": doc.page_content
        }
        for doc in compressed_docs
    ]

    # TF-IDF相似度
    all_texts = [answer] + retrieved_contents
    vectorizer = TfidfVectorizer(tokenizer=jieba_tokenizer, token_pattern=None)
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    answer_vector = tfidf_matrix[0].toarray()[0]
    retrieved_vectors = tfidf_matrix[1:].toarray()
    similarity_scores = [
        float(np.dot(answer_vector, retrieved_vector) / (np.linalg.norm(answer_vector) * np.linalg.norm(retrieved_vector)))
        if np.linalg.norm(answer_vector) > 0 and np.linalg.norm(retrieved_vector) > 0 else 0.0
        for retrieved_vector in retrieved_vectors
    ]

    # BM25相似度
    bm25_corpus = [jieba_tokenizer(text) for text in retrieved_contents]
    bm25 = BM25Okapi(bm25_corpus)
    answer_tokens = jieba_tokenizer(answer)
    bm25_scores = bm25.get_scores(answer_tokens)
    bm25_scores = [float(score) for score in bm25_scores]

    avg_tfidf_score = float(np.mean(similarity_scores)) if similarity_scores else 0.0
    avg_bm25_score = float(np.mean(bm25_scores)) if bm25_scores else 0.0

    # 添加分数到文档
    for doc, tfidf_score, bm25_score in zip(retrieved_documents, similarity_scores, bm25_scores):
        doc['TF-IDF_score'] = tfidf_score
        doc['BM25_scores'] = bm25_score

    output_data.append({
        "query": query,
        "answer": answer,
        "avg_TFIDF_score": avg_tfidf_score,
        "avg_BM25_score": avg_bm25_score,
        "retrieved_documents": retrieved_documents
    })

with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)
print(f'EmbeddingsFilter上下文压缩检索结果已保存到 {output_file_path}')