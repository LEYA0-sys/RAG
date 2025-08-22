from langchain.vectorstores import FAISS  
from langchain.embeddings import HuggingFaceBgeEmbeddings  
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_curve, auc
from tqdm import tqdm
import matplotlib.pyplot as plt

embeddings = HuggingFaceBgeEmbeddings(
    model_name='BAAI/bge-large-zh-v1.5',
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vector_db = FAISS.load_local(
    r"output\v1\FAISS\bge_large_v1.5\faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

with open(r'D:\desktop\code\QA\test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

test_queries = [item["question"] for item in test_data]
correct_answers = [item["answer"] for item in test_data]

correct_answer_embeddings = embeddings.embed_documents(correct_answers)

y_true = []
y_scores = []

for query, correct_ans_embed in tqdm(zip(test_queries, correct_answer_embeddings), total=len(test_queries), desc="Evaluating"):
    # FAISS similarity_search (返回 doc + score)
    results = vector_db.similarity_search_with_score(query, k=3)
    
    for doc, faiss_score in results:
        doc_embed = embeddings.embed_query(doc.page_content)  # 可以改成提前缓存
        similarity = cosine_similarity([doc_embed], [correct_ans_embed])[0][0]

        # 只要 uuid/source_file 能对得上即可，可以扩展 gt 判断
        y_true.append(1 if similarity > 0.8 else 0)  # 0.8 近似 ground truth，用软标签
        y_scores.append(similarity)

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
pr_auc = auc(recall, precision)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_idx = np.argmax(f1_scores)

print(f"最佳阈值: {thresholds[best_idx]:.3f}")
print(f"最佳 F1-score: {f1_scores[best_idx]:.3f}")
print(f"PR AUC: {pr_auc:.3f}")

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR Curve (AUC={pr_auc:.3f})', color='blue')
plt.scatter(recall[best_idx], precision[best_idx], color='red', label=f'Best F1 @ {thresholds[best_idx]:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.title('Precision-Recall Curve')
plt.grid()
plt.show()
