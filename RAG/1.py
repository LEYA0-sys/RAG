# 导入必要的库并打印版本信息
import langchain
import langchain_community
import pypdf
import sentence_transformers
print(1)
for module in (langchain, langchain_community, pypdf, sentence_transformers):
    print(f"{module.__name__:<30}{module.__version__}")

#导入操作系统和数据处理相关库
import os
import pandas as pd


# 定义嵌入模型
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from langchain_ollama import OllamaLLM
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')
#定义嵌入模型，跑通这部分代码需要开代理
embedding_model = HuggingFaceEmbeddings(
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
index_path = r"E:\RAG\database"
if os.path.exists(index_path):
    print("index.faiss 文件存在")
    vector_db = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    
# 输出向量数据库vector_db中包含的数据条数
print(f"向量数据库中数据总数: {vector_db.index.ntotal}")

#采用ollama的LLM模型千问
llm = OllamaLLM(model="qwen:7b")