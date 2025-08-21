# 检索器：
# 将底层向量存储的.similarity_search_with_score 方法包装在一个短函数中，该函数将分数打包到相关文档的元数据中。
from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import chain

@chain
def retriever(query: str) -> List[Document]:
    docs, scores = zip(*vector_db.similarity_search_with_score(query))
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score

    return docs

result=retriever.invoke("请问保险产品有哪些？")
result