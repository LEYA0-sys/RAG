
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
import json
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import concurrent.futures
import random
k=20  # 返回检索结果的数量
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
    input_file_path = f'../outputs/outputs_4.0/混合检索/hybrid_results_k={k}.json'
    output_dir = '../outputs/outputs_4.0/上下文压缩'
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, f'compressed_results_k={k}.json')
    # 假设 llm 已经初始化
    process_compression(
        input_file_path=input_file_path,
        output_file_path=output_file_path,
        llm=llm
    )
