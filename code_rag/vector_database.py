import os
import torch
import pickle
import log
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.document_loaders import TextLoader
# from split.single_split import split_doc
# from split.multi_split import split_docs
#from split.split import all_chunks
# from xinference.client import Client
from split.semantic_splitter import all_chunks
output_dir = os.path.join('output', f'v3(semantic_search)')
# # 连接到 Xinference 服务
# client = Client("http://localhost:9997")
# # 加载嵌入模型
# model_uid = client.launch_model(
#     model_name="bge-large-zh-v1.5",
#     model_size_in_billions=None,  
#     quantization=None,
#     model_type="embedding"
# )

# # 获取模型实例
# model = client.get_model(model_uid)

# # 定义一个自定义的嵌入类，使用 Xinference 模型生成嵌入
# class XinferenceEmbeddings:
#     def __init__(self, model):
#         self.model = model

#     def embed_documents(self, texts):
#         embeddings = []
#         for text in texts:
#             result = self.model.create_embedding(input=[text])
#             embeddings.append(result["data"][0]["embedding"])
#         return embeddings

#     def embed_query(self, text):
#         result = self.model.create_embedding(input=[text])
#         return result["data"][0]["embedding"]

# embeddings = XinferenceEmbeddings(model)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')

embeddings = HuggingFaceBgeEmbeddings(
    model_name='BAAI/bge-large-zh-v1.5',
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)

def attach_metadata_to_faiss_db(vector_db, metadata_path):
    with open(metadata_path, 'rb') as f:
        metadatas = pickle.load(f)
    for i, doc in enumerate(vector_db.docstore._dict.values()):
        doc.metadata = metadatas[i]


def get_vector_db(docs, store_path, force_rebuild=False):
    index_path = os.path.join(store_path, "faiss_index")
    metadata_path = os.path.join(store_path, "faiss_metadata.pkl")

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        force_rebuild = True

    if force_rebuild:
        os.makedirs(store_path, exist_ok=True)
        vector_db = FAISS.from_documents(docs, embedding=embeddings)
        vector_db.save_local(index_path)

        # 保存 metadatas（Langchain 的 FAISS 默认不会持久化 metadata）
        with open(metadata_path, 'wb') as f:
            pickle.dump([doc.metadata for doc in docs], f)

    else:
        vector_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        attach_metadata_to_faiss_db(vector_db, metadata_path)
    return vector_db


def update_faiss_vector_db(new_docs, store_path, embeddings):
    index_path = os.path.join(store_path, "faiss_index")
    metadata_path = os.path.join(store_path, "faiss_metadata.pkl")

    # 1. 加载现有向量数据库或初始化新库
    if os.path.exists(index_path):
        vector_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        os.makedirs(store_path, exist_ok=True)
        vector_db = FAISS.from_documents([], embedding=embeddings)

    # 2. 添加新文档并保存
    vector_db.add_documents(new_docs)
    vector_db.save_local(index_path)

    # 3. metadata 更新
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            metadatas = pickle.load(f)
    else:
        metadatas = []

    metadatas.extend([doc.metadata for doc in new_docs])
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadatas, f)

    print(f"[INFO] 添加了 {len(new_docs)} 条新文档，向量库已更新并保存。")


vector_db = get_vector_db(all_chunks, store_path=os.path.join(output_dir, 'FAISS', 'bge_large_v1.5'))    
# vector_db = update_faiss_vector_db(split_docs, store_path=os.path.join(output_dir, 'FAISS', 'bge_large_v1.5'), embeddings=embeddings)
