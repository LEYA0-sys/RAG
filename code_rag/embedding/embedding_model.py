from xinference.client import RESTfulClient
from split.single_split import documents

client = RESTfulClient("http://127.0.0.1:9997")
model_uid = "bge-large-zh-v1.5"  
model = client.get_model(model_uid)
texts = [str(doc) for doc in documents]
result = model.create_embedding(input=texts)
# 输出嵌入结果
for i, embedding in enumerate(result["data"]):
    print(f"文本 {i + 1} 的嵌入向量: {embedding['embedding']}")
