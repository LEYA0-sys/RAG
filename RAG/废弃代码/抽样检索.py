# 导入必要的库并打印版本信息
import langchain, langchain_community, pypdf, sentence_transformers

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

#上下文压缩
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
import json
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import concurrent.futures
import random

def compress_document(doc, compressor, query):
    """压缩单个文档"""
    try:
        compressed_doc = compressor.compress_documents([doc], query=query)
        return compressed_doc[0] if compressed_doc else doc
    except Exception as e:
        print(f"压缩文档时出错: {str(e)}")
        return doc

def parallel_compress_documents(docs, compressor, query, num_processes=None):
    """并行压缩多个文档"""
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)  # 保留一个CPU核心
    
    # 如果文档数量超过10个，随机抽取10个
    if len(docs) > 10:
        docs = random.sample(docs, 10)
    
    # 创建进程池
    with Pool(processes=num_processes) as pool:
        # 使用partial固定compressor和query参数
        compress_func = partial(compress_document, compressor=compressor, query=query)
        # 并行压缩所有文档
        compressed_docs = pool.map(compress_func, docs)
    
    return compressed_docs

def compress_one(item, compressor):
    """压缩单个查询的所有文档"""
    query = item['query']
    docs = []
    
    # 准备文档
    for doc_dict in item['retrieved_documents']:
        doc = Document(
            page_content=doc_dict['content'],
            metadata={
                "uuid": doc_dict.get("uuid", "未指定"),
                "source_file": doc_dict.get("source_file", ""),
                "page": doc_dict.get("page", "未指定")
            }
        )
        docs.append(doc)
    
    # 并行压缩文档
    compressed_docs = parallel_compress_documents(docs, compressor, query)
    
    # 更新结果
    for i, doc_dict in enumerate(item['retrieved_documents']):
        if i < len(compressed_docs):
            doc_dict['compressed_content'] = compressed_docs[i].page_content
    
    return item

def process_compression(input_file_path, output_file_path, llm, k=20):
    """主处理函数"""
    # 创建压缩器
    compressor = LLMChainExtractor.from_llm(llm)
    
    # 读取输入数据
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 使用线程池并行处理所有查询
    with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        # 使用partial固定compressor参数
        compress_func = partial(compress_one, compressor=compressor)
        results = list(executor.map(compress_func, data['results']))
    
    # 更新结果
    data['results'] = results
    
    # 保存结果
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"压缩结果已保存到 {output_file_path}")

if __name__ == "__main__":
    # 设置输入输出路径
    input_file_path = '../outputs/outputs_2.0/混合检索/hybrid_results_k=20.json'
    output_dir = '../outputs/outputs_2.0/上下文压缩'
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'compressed_results_k=20.json')
    
    # 初始化llm（需要从外部传入）
    # llm = ...
    
    # 处理压缩
    # process_compression(input_file_path, output_file_path, llm) 
# 假设 llm 已经初始化
process_compression(
    input_file_path='../outputs/outputs_4.0/混合检索/hybrid_results_k=20.json',
    output_file_path='../outputs/outputs_4.0/上下文压缩/compressed_results_k=20.json',
    llm=llm
)

# 上下文压缩后的问答链生成
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from typing import Any, List, Optional, Dict, Mapping
from pydantic import BaseModel, Field
import json
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import concurrent.futures
import requests
import time
import hashlib
import hmac
import base64
import datetime
import random

def process_qa_chain(docs, query, qa_chain):
    """处理单个查询的问答链"""
    try:
        # 使用新的 invoke 方法，并传入正确的参数格式
        answer = qa_chain.invoke({"context": docs, "question": query})
        return answer
    except Exception as e:
        print(f"生成答案时出错: {str(e)}")
        return f"Error generating answer: {e}"

def parallel_process_qa(items, qa_chain, num_processes=None):
    """并行处理多个查询的问答链"""
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)  # 保留一个CPU核心
    
    # 创建进程池
    with Pool(processes=num_processes) as pool:
        # 使用partial固定qa_chain参数
        process_func = partial(process_qa_chain, qa_chain=qa_chain)
        # 并行处理所有查询
        results = pool.starmap(process_func, [(item['compressed_docs'], item['query']) for item in items])
    
    return results

def prepare_qa_data(item):
    """准备问答链的输入数据，随机抽取10个文档"""
    compressed_docs_for_qa = []
    retrieved_docs = item['retrieved_documents']
    
    # 如果文档数量超过10个，随机抽取10个
    if len(retrieved_docs) > 10:
        sampled_indices = random.sample(range(len(retrieved_docs)), 10)
        sampled_docs = [retrieved_docs[i] for i in sampled_indices]
    else:
        sampled_docs = retrieved_docs
    
    for doc_dict in sampled_docs:
        if 'content' in doc_dict:
            compressed_docs_for_qa.append(Document(
                page_content=doc_dict['content'],
                metadata={
                    "uuid": doc_dict.get("uuid", "未指定"),
                    "source_file": doc_dict.get("source_file", ""),
                    "page": doc_dict.get("page", "未指定")
                }
            ))
    return {
        'compressed_docs': compressed_docs_for_qa,
        'query': item['query'],
        'sampled_docs': sampled_docs  # 保存采样后的文档
    }

def process_qa_with_compression(input_file_path, output_file_path, llm, k=20):
    """主处理函数"""
    # 定义自定义的prompt模板
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "你是一名金融保险领域的专家，擅长根据所获取的信息片段，对问题进行分析和推理。你的任务是根据所获取的上下文信息回答问题。"
            "回答保持简洁，不必重复问题，不要添加与问题无关的任何内容。"
            "如果上下文信息不足以回答问题，请明确说明。"
            "如果上下文信息中包含多个片段，请综合考虑所有片段的信息来回答问题。"
            "如果所提供的上下文中存在矛盾的信息，请指出并解释原因。"
            "如果上下文信息中存在多余的信息，如符号、标点、空格等，请先判断是否可以忽略，确保最终输出结果清晰易懂，是流畅的内容。"
            "如果上下文信息中存在不必要的细节或背景信息，请忽略这些信息。"
            "如果上下文信息中存在繁体中文，请将其转换为简体字。"
            "如果上下文信息中存在拼音，请将其转换为中文。"
            "如果上下文信息中存在英文，请将其翻译为中文。"
            "请根据以下上下文回答问题。\n"
            "上下文：{context}\n"
            "问题：{question}\n"
            "请用简洁、准确的语言作答："
        )
    )

    # 使用新的 create_stuff_documents_chain 替代 load_qa_chain
    qa_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=qa_prompt
    )

    # 读取压缩后的结果
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 准备问答链的输入数据
    qa_items = []
    for item in data['results']:  # 注意这里改为 data['results']
        qa_items.append(prepare_qa_data(item))

    # 使用线程池并行处理所有查询
    with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        # 使用partial固定qa_chain参数
        process_func = partial(process_qa_chain, qa_chain=qa_chain)
        answers = list(executor.map(
            lambda x: process_func(x['compressed_docs'], x['query']),
            qa_items
        ))

    # 更新结果，调整输出格式
    for item, qa_item, answer in zip(data['results'], qa_items, answers):
        # 创建新的结果字典，将generated_answer放在前面
        new_item = {
            "query": item['query'],
            "answer": item['answer'],
            "correct_page": item['correct_page'],
            "recall": item['recall'],
            "mrr": item['mrr'],
            "generated_answer": answer,
            "retrieved_documents": qa_item['sampled_docs']  # 使用采样后的文档
        }
        # 更新原始item
        item.clear()
        item.update(new_item)

    # 保存结果
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"问答链生成结果已保存到 {output_file_path}")

if __name__ == "__main__":
    # 设置输入输出路径
    input_file_path = '../outputs/outputs_1.0/上下文压缩/compressed_results_k=20.json'
    output_dir = '../outputs/outputs_4.0/上下文压缩'
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'compressed_with_qa_k=20.json')
    
    # 假设 llm 已经初始化，采用多进程方式
    process_qa_with_compression(
        input_file_path='../outputs/outputs_1.0/上下文压缩/compressed_results_k=20.json',
        output_file_path='../outputs/outputs_4.0/上下文压缩/compressed_with_qa_k=20.json',
        llm=llm
    )