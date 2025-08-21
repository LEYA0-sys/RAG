##EmbeddingsFilter

##它通过嵌入文档和查询，只返回与查询具有足够相似嵌入的文档。

# 导入 EmbeddingsFilter 模块和操作系统模块
from langchain.retrievers.document_compressors import EmbeddingsFilter
import os

# 定义 EmbeddingsFilter，用于根据相似度过滤文档
embeddings_filter = EmbeddingsFilter(embeddings=embedding_model, similarity_threshold=0.5)
k=20  # 设置检索返回的文档数量

# 创建检索器，结合向量数据库和 EmbeddingsFilter
retriever = vector_db.as_retriever(search_kwargs={"k": k})
compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=retriever)

# 执行检索操作
query = "请问保险产品有哪些？"  # 定义检索问题
compressed_docs = compression_retriever.get_relevant_documents(query)  # 获取相关文档

# 检查并创建输出文件夹
output_dir = '../outputs/压缩器'
os.makedirs(output_dir, exist_ok=True)  # 如果文件夹不存在则创建

# 保存检索结果到文件
output_file = os.path.join(output_dir, f'EmbeddingsFilter_docs_with_score_k={k}.txt')
with open(output_file, 'w', encoding='utf-8') as f:
    for idx, doc in enumerate(compressed_docs, 1):
        # 写入每个文档的内容和元数据
        f.write(f"\n===========结果{idx}==========:\n")
        f.write(f"内容: {doc.page_content}\n")
        f.write(f"元数据: {doc.metadata}\n")

# 打印保存结果的路径
print(f'压缩后的结果已保存到 {output_file}')