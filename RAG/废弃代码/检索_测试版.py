# 导入必要的库并打印版本信息
import langchain, langchain_community, pypdf, sentence_transformers

for module in (langchain, langchain_community, pypdf, sentence_transformers):
    print(f"{module.__name__:<30}{module.__version__}")

#导入操作系统和数据处理相关库
import os
import pandas as pd


# 定义嵌入模型
from langchain_ollama import OllamaEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
import torch
from langchain_ollama import OllamaLLM
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')
#定义嵌入模型，跑通这部分代码需要开代理
embedding_model = HuggingFaceBgeEmbeddings(
    model_name='BAAI/bge-large-zh-v1.5',
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)

#加载向量数据库
# 加载 index.faiss 文件作为向量数据库
from langchain_community.vectorstores.faiss import FAISS

import faiss
import os
import numpy as np
# 确保路径正确数据库文件\index.faiss
index_path = r"E:\RAG\database_updated"
if os.path.exists(index_path):
    print("index_updated.faiss 文件存在")
    vector_db = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    
# 输出向量数据库vector_db中包含的数据条数
print(f"向量数据库中数据总数: {vector_db.index.ntotal}")


#采用ollama的LLM模型千问
llm = OllamaLLM(model="qwen:7b")


#检索板块

# 在简单的数据检索基础上进行打分操作，采用similarity_search_with_score 方法
# 目前最低分数：0.8534546494483948
# similarity_search_with_score 方法不仅允许您返回文档，还允许返回查询到查询向量和文档向量之间的距离分数。返回的距离分数是L2距离。因此，分数越低越好。
# L2距离分数指的是欧氏距离（Euclidean Distance），也叫L2范数。在向量检索中，L2距离用于衡量两个向量（如查询向量和文档向量）之间的相似度。
def rag_search_with_score(query, k):
    """
    使用 vector_db 对输入 query 进行检索，返回前 k 条结果及其相似度分数。
    参数:
        query (str): 用户输入的检索问题。
        k (int): 返回的结果数量，默认为 5。
    返回:
        list: 每个元素为 dict，包含文档内容、元数据和相似度分数。
    """
    # 调用 vector_db 的 similarity_search_with_score 方法，
    # 返回 [(Document, score), ...]，每个元素是文档和对应的相似度分数
    results = vector_db.similarity_search_with_score(query, k=k)
    scored_docs = []  # 用于存储带分数的检索结果
    for doc, score in results:
        # 将分数写入文档的元数据，方便后续使用
        doc.metadata["l2_score"] = score
        # 构造包含内容、元数据和分数的字典，加入结果列表
        scored_docs.append({
            "content": doc.page_content,  # 文档内容
            "metadata": doc.metadata,     # 文档元数据（包含 l2_score）
            "l2_score": score            # L2 距离分数
        })
    return scored_docs  # 返回带分数的检索结果列表

# 示例调用
query = "保单支持哪些货币"  # 定义检索问题
k = 5  # 设置返回结果数量
# 调用函数 rag_search_with_score 进行检索，并获取带分数的结果
scored_results = rag_search_with_score(query, k=k)

# 基于余弦相似度的检索效果打分
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def cosine_similarity_score(query_embedding, doc_embeddings):
    """
    计算查询向量与文档向量之间的余弦相似度。
    参数:
        query_embedding (np.ndarray): 查询向量。
        doc_embeddings (np.ndarray): 文档向量矩阵，每行是一个文档的向量。
    返回:
        list: 每个文档的余弦相似度分数。
    """
    # 计算余弦相似度
    scores = cosine_similarity(query_embedding.reshape(1, -1), doc_embeddings)
    return scores.flatten().tolist()

# 获取查询向量
query_embedding = embedding_model.embed_query(query)

# 获取文档向量
doc_embeddings = np.array([embedding_model.embed_query(doc['content']) for doc in scored_results])

# 计算余弦相似度分数
cosine_scores = cosine_similarity_score(np.array(query_embedding), doc_embeddings)

# 将余弦相似度分数添加到文档元数据中
for doc, score in zip(scored_results, cosine_scores):
    doc['metadata']['cosine_score'] = score

# 输出每个文档的 L2 距离分数和余弦相似度分数
for idx, doc in enumerate(scored_results, 1):
    print(f"文档 {idx}: L2 距离分数 = {doc['metadata']['l2_score']}, 余弦相似度分数 = {doc['metadata']['cosine_score']}")

# 保存带 L2 距离分数和余弦相似度分数的检索结果到文件
output_dir = '../outputs/简单检索_更新版'
os.makedirs(output_dir, exist_ok=True)  # 创建 outputs 目录（如果不存在）
with open(f'../outputs/简单检索_更新版/rag_results_with_scores_k={k}.txt', 'w', encoding='utf-8') as f:
    for idx, doc in enumerate(scored_results, 1):
        f.write(f"\n===========结果{idx}==========:\n")
        f.write(f"内容:\n{doc['content']}\n")
        f.write(f"元数据: {doc['metadata']}\n")
        f.write(f"L2 距离分数: {doc['metadata']['l2_score']}\n")
        f.write(f"余弦相似度分数: {doc['metadata']['cosine_score']}\n")
print(f'带 L2 距离分数和余弦相似度分数的检索结果已保存到 ../outputs/简单检索_更新版/rag_results_with_scores_k={k}.txt')