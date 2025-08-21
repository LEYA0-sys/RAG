#串联压缩器和文档转换器


# 导入必要的模块，包括文档转换器和压缩器
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers.document_compressors import EmbeddingsFilter

# 定义多个转换器和过滤器
#根据相似值过滤相关文档，只保留与查询相似度大于或等于 50% 的文档
embeddings_filter = EmbeddingsFilter(embeddings=embedding_model, similarity_threshold=0.5)
#将文档分块
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")  # 文本分块器
redundant_filter = EmbeddingsRedundantFilter(embeddings=embedding_model)  # 去冗余过滤器
relevant_filter = EmbeddingsFilter(embeddings=embedding_model, similarity_threshold=0.5)  # 相关性过滤器

# 创建压缩器流水线，将多个转换器串联
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)
k=15  # 设置检索返回的文档数量

# 创建检索器，结合向量数据库和压缩器流水线
retriever = vector_db.as_retriever(search_kwargs={"k": k})
compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=retriever)

# 执行检索操作
query = "请问保险产品有哪些？"  # 定义检索问题
compressed_docs = compression_retriever.get_relevant_documents(query)  # 获取相关文档

# 检查并创建输出文件夹
output_dir = '../outputs/串联'
os.makedirs(output_dir, exist_ok=True)  # 如果文件夹不存在则创建

# 保存检索结果到文件
with open(f'../outputs/串联/EmbeddingsFilter_docs_k={k}.txt', 'w', encoding='utf-8') as f:
    for idx, doc in enumerate(compressed_docs, 1):
        # 写入每个文档的内容和元数据
        f.write(f"\n===========结果{idx}==========:\n")
        f.write(f"内容: {doc.page_content}\n")
        f.write(f"元数据: {doc.metadata}\n")

# 打印保存结果的路径
print(f'压缩后的结果已保存到 ../outputs/串联/EmbeddingsFilter_docs_k={k}.txt')