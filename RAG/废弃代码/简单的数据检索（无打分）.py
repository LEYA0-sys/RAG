# 简单的数据检索（无打分）
# 生成文件：outputs\简单检索\rag_results_easy.txt

# 定义简单的RAG检索函数
def rag_search(query, k=5):
    """
    使用vector_db对输入query进行检索，返回前k条结果。
    参数：
        query (str): 用户输入的检索问题。
        k (int): 返回的结果数量，默认为5。
    返回：
        list: 检索到的文档对象列表。
    """
    # 创建检索器实例，设置返回top k条结果
    retriever = vector_db.as_retriever(search_kwargs={"k": k})
    #invoke方法用于对输入query执行检索操作,返回的结果是一个文档对象列表，每个文档对象包含内容和元数据
    results = retriever.invoke(query)
    return results

# 示例调用：定义检索问题
query = "请问保险产品有哪些？"
# 调用rag_search函数进行检索，返回前5条结果
results = rag_search(query, k=5)

# 保存输出到outputs/简单检索/rag_results_easy.txt
import os
# 创建outputs/简单检索目录（如果不存在）
os.makedirs('../outputs/简单检索', exist_ok=True)
# 将检索结果写入文件，每条结果包含内容和元数据
with open('../outputs/简单检索/rag_results_easy.txt', 'w', encoding='utf-8') as f:
    for idx, doc in enumerate(results, 1):
        f.write(f"\n===========结果{idx}==========:\n")  # 写入分隔符
        f.write(f"内容: {doc.page_content}\n")           # 写入文档内容
        f.write(f"元数据: {doc.metadata}\n")             # 写入文档元数据
print('检索结果已保存到 ../outputs/简单检索/rag_results_easy.txt')