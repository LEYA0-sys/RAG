# 调试 compressed_docs 输出为空的问题
from langchain.retrievers.document_compressors import EmbeddingsFilter

# 检查向量数据库是否正确加载
print(f"向量数据库是否加载成功: {vector_db is not None}")
if vector_db:
    print(f"向量数据库条目数: {vector_db.index.ntotal}")

# 检查嵌入模型是否正确加载
print(f"嵌入模型是否加载成功: {embedding_model is not None}")

# 调整 similarity_threshold 并测试
embeddings_filter = EmbeddingsFilter(embeddings=embedding_model, similarity_threshold=0.5)  # 降低阈值
retriever = vector_db.as_retriever(search_kwargs={"k": 5})
compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=retriever)

# 执行检索
query = "请问保险产品有哪些？"
compressed_docs = compression_retriever.get_relevant_documents(query)

# 输出结果
if compressed_docs:
    print(f"检索到的文档数量: {len(compressed_docs)}")
    for doc in compressed_docs:
        print(f"文档内容: {doc.page_content}")
        print(f"文档元数据: {doc.metadata}")
else:
    print("未检索到任何文档，请检查数据库或查询条件。")