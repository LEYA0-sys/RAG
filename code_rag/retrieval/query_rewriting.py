from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import json
import re
import os
from nltk import sent_tokenize
# 初始化嵌入模型
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

with open(r'D:\desktop\code\QA_ds.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# 初始化 Sentence-BERT 编码器
sentence_model = SentenceTransformer("BAAI/bge-large-zh-v1.5")

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# rewrite_tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b",cache_dir="D:/cache/models", trust_remote_code=True)
# rewrite_model = AutoModelForSeq2SeqLM.from_pretrained("THUDM/chatglm3-6b", cache_dir="D:/cache/models", trust_remote_code=True).half()

# def rewrite_question(question):
#     prompt = f"将下面的问题进行改写，使其更易于被知识库检索：{question}"
#     inputs = rewrite_tokenizer(prompt, return_tensors="pt")
#     outputs = rewrite_model.generate(**inputs, max_new_tokens=50)
#     rewritten = rewrite_tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return rewritten

from transformers import AutoTokenizer, AutoModel

# 使用 AutoModel
rewrite_tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b",cache_dir="D:/cache/models", trust_remote_code=True)
rewrite_model = AutoModel.from_pretrained("THUDM/chatglm3-6b",cache_dir="D:/cache/models", trust_remote_code=True).half()
rewrite_model.eval()

def rewrite_question(query: str):
    history = []
    response, _ = rewrite_model.chat(rewrite_tokenizer, query, history=history)
    return response



def calculate_embedding(text):
    return sentence_model.encode(text, convert_to_tensor=False)

def normalize_page_num(page_num):
    """标准化页码格式（移除'第'、'页'等字符）"""
    if isinstance(page_num, str):
        return re.sub(r'[第页\s]', '', page_num).strip()
    return str(page_num)

def is_relevant(doc_content, correct_answer, threshold=0.70):
    """
    改进的相关性判断：
    1. 比较文档中最相似的句子（而非整个文档）
    2. 添加关键词快速匹配（如果答案完全包含在文档中）
    """
    # 快速检查：如果答案完全包含在文档中，直接返回True
    if correct_answer in doc_content:
        return True
    
    # 分句比较相似度
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
    
    return max_similarity >= threshold


def calculate_precision(docs, correct_answer, threshold=0.70):
    relevant = [doc for doc in docs if is_relevant(doc.page_content, correct_answer, threshold)]
    return len(relevant) / len(docs) if docs else 0

def calculate_recall(docs, correct_answer, threshold=0.70):
    return int(any(is_relevant(doc.page_content, correct_answer, threshold) for doc in docs))

def calculate_mrr(docs, correct_answer, threshold=0.70):
    for i, doc in enumerate(docs, start=1):
        if is_relevant(doc.page_content, correct_answer, threshold):
            return 1 / i
    return 0.0

def calculate_hit_at_k(docs, correct_answer, threshold=0.70):
    return int(any(is_relevant(doc.page_content, correct_answer, threshold) for doc in docs))

def calculate_page_match(docs, correct_source_file, correct_page_num):
    correct_page_num = normalize_page_num(correct_page_num)
    for doc in docs:
        meta = doc.metadata
        # source_file_no_ext = os.path.splitext(meta.get('source_file', ''))[0]
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


def test_retrieval_effectiveness(test_data, k_retrieval=20, threshold=0.70, 
                                verbose=True, save_path="./retrieval_result/QA_db/query_0.70(20).json"):
    precision_scores = []
    recall_scores = []
    mrr_scores = []
    hit_scores = []
    page_match_scores = []
    page_mrr_scores = []

    results_data = []

    for item in tqdm(test_data, total=len(test_data), desc="Evaluating"):
        query = item["question"]
        correct_answer = item["answer"]
        correct_source_file = item["source_file"]
        correct_page_num = normalize_page_num(item["page_num"])

        # FAISS 检索
        # results = vector_db.similarity_search(query, k=k_retrieval)
        rewritten_query = rewrite_question(query)
        results = vector_db.similarity_search(rewritten_query, k=k_retrieval)

        retrieved_docs = results  # 包含 page_content 和 metadata 的完整 Document 对象

        # 计算指标
        precision = calculate_precision(retrieved_docs, correct_answer, threshold)
        recall = calculate_recall(retrieved_docs, correct_answer, threshold)
        mrr = calculate_mrr(retrieved_docs, correct_answer, threshold)
        hit = calculate_hit_at_k(retrieved_docs, correct_answer, threshold)
        page_match = calculate_page_match(retrieved_docs, correct_source_file, correct_page_num)
        page_mrr = calculate_page_mrr(retrieved_docs, correct_source_file, correct_page_num)

        # 收集结果
        precision_scores.append(precision)
        recall_scores.append(recall)
        mrr_scores.append(mrr)
        hit_scores.append(hit)
        page_match_scores.append(page_match)
        page_mrr_scores.append(page_mrr)

        # 存储详细信息
        results_data.append({
            "query": query,
            "correct_answer": correct_answer,
            "correct_source_file": correct_source_file,
            "correct_page_num": correct_page_num,
            "correct_text": item.get("text", ""),
            "retrieved_documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "is_relevant": is_relevant(doc.page_content, correct_answer, threshold)
                } for doc in retrieved_docs
            ],
            "precision": float(precision),
            "recall": float(recall),
            "mrr": float(mrr),
            "hit@k": int(hit),
            "page_match": int(page_match),
            "page_mrr": float(page_mrr)
        })

        if verbose:
            print(f"\nQuery: {query}")
            print(f"Correct Answer: {correct_answer}")
            print(f"Precision@{k_retrieval}: {precision:.4f} | Recall@{k_retrieval}: {recall} | MRR@{k_retrieval}: {mrr:.4f}")
            print(f"Hit@{k_retrieval}: {hit} | Page Match@{k_retrieval}: {page_match} | Page MRR@{k_retrieval}: {page_mrr:.4f}")

    # 计算平均指标
    avg_precision = float(np.mean(precision_scores))
    avg_recall = float(np.mean(recall_scores))
    avg_mrr = float(np.mean(mrr_scores))
    avg_hit = float(np.mean(hit_scores))
    avg_page_match = float(np.mean(page_match_scores))
    avg_page_mrr = float(np.mean(page_mrr_scores))

    print(f"\n--- Overall Evaluation ---")
    print(f"Average Precision@{k_retrieval}: {avg_precision:.4f}")
    print(f"Average Recall@{k_retrieval}: {avg_recall:.4f}")
    print(f"Average MRR@{k_retrieval}: {avg_mrr:.4f}")
    print(f"Average Hit@{k_retrieval}: {avg_hit:.4f}")
    print(f"Average Page Match@{k_retrieval}: {avg_page_match:.4f}")
    print(f"Average Page MRR@{k_retrieval}: {avg_page_mrr:.4f}")

    # 保存结果
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
        "average_precision": float(avg_precision),
        "average_recall": float(avg_recall),
        "average_mrr": float(avg_mrr),
        "average_hit": float(avg_hit),
        "average_page_match": float(avg_page_match),
        "average_page_mrr": float(avg_page_mrr),
        "results": [
            {
                "query": result["query"],
                "correct_answer": result["correct_answer"],
                "correct_source_file": result["correct_source_file"],
                "correct_page_num": str(result["correct_page_num"]),  
                "correct_text": result["correct_text"],
                "retrieved_documents": [
                    {
                        "content": doc["content"],
                        "metadata": {
                            "source_file": str(doc["metadata"]["source_file"]),
                            "page_num": str(doc["metadata"]["page_num"])
                        },
                        "is_relevant": bool(doc["is_relevant"])  # 转换bool类型
                    }
                    for doc in result["retrieved_documents"]
                ],
                "precision": float(result["precision"]),
                "recall": float(result["recall"]),
                "mrr": float(result["mrr"]),
                "hit@k": int(result["hit@k"]),
                "page_match": int(result["page_match"]),
                "page_mrr": float(result["page_mrr"])
            }
            for result in results_data
        ]
    }, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {save_path}")

# 运行评估
test_retrieval_effectiveness(
    test_data, 
    k_retrieval=20, 
    threshold=0.70,
    verbose=False,
    save_path="./retrieval_result/QA_db/query_0.70(20).json"
)