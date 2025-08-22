import os
import json
import logging
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from sentence_transformers import SentenceTransformer, CrossEncoder


# ========== 日志配置 ==========
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ========== 工具函数 ==========
def cosine_sim(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]


def extract_doc_info(doc):
    return {
        "content": doc.page_content,
        "metadata": doc.metadata
    }


# ========== 主类定义 ==========
class RetrievalEvaluator:
    def __init__(self, 
                 vector_db_path,
                 test_data_path,
                 device='cpu',
                 k_retrieval=50,
                 k_rerank=10,
                 threshold=0.7):

        self.k_retrieval = k_retrieval
        self.k_rerank = k_rerank
        self.threshold = threshold

        # 嵌入器
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name='BAAI/bge-large-zh-v1.5',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 加载向量数据库
        self.vector_db = FAISS.load_local(
            vector_db_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )

        # Sentence-BERT + reranker
        self.sbert = SentenceTransformer("BAAI/bge-base-zh-v1.5")
        self.reranker = CrossEncoder("BAAI/bge-reranker-large")

        # 数据加载
        with open(test_data_path, 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)

        # 嵌入缓存
        self.embed_cache = {}

    def encode(self, text):
        if text not in self.embed_cache:
            self.embed_cache[text] = self.sbert.encode(text, convert_to_tensor=False)
        return self.embed_cache[text]

    def is_relevant(self, doc_vec, answer_vec):
        return cosine_sim(doc_vec, answer_vec) >= self.threshold

    def rerank(self, query, docs):
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in reranked[:self.k_rerank]]

    def calculate_metrics(self, docs, answer, answer_vec):
        doc_vecs = [self.encode(doc) for doc in docs]
        flags = [self.is_relevant(doc_vec, answer_vec) for doc_vec in doc_vecs]

        # Precision, Recall, Hit@K
        precision = sum(flags) / len(docs) if docs else 0
        recall = int(any(flags))
        hit = recall

        # MRR
        mrr = 0.0
        for i, flag in enumerate(flags, start=1):
            if flag:
                mrr = 1 / i
                break

        # MAP
        ap = 0.0
        num_hits = 0
        for i, flag in enumerate(flags, start=1):
            if flag:
                num_hits += 1
                ap += num_hits / i
        map_score = ap / num_hits if num_hits > 0 else 0.0

        return precision, recall, mrr, hit, map_score

    def calculate_page_match(self, docs, correct_file, correct_page):
        for i, doc in enumerate(docs, start=1):
            meta = doc.metadata
            if meta.get('source_file') == correct_file and str(meta.get('page_num')) == correct_page:
                return 1, 1 / i
        return 0, 0.0

    def evaluate(self, save_path="./retrieval_result/v3/eval_result.json", verbose=False):
        results = []

        precision_list, recall_list, mrr_list = [], [], []
        hit_list, map_list = [], []
        page_match_list, page_mrr_list = [], []

        for item in tqdm(self.test_data, desc="Evaluating"):
            query = item["question"]
            answer = item["answer"]
            correct_file = item["文档来源"]
            correct_page = item["页码"].replace("第 ", "").replace(" 页", "").strip()

            answer_vec = self.encode(answer)

            # 检索
            raw_results = self.vector_db.similarity_search(query, k=self.k_retrieval)
            reranked_docs = self.rerank(query, raw_results)
            reranked_contents = [doc.page_content for doc in reranked_docs]

            # 评估指标
            p, r, mrr, hit, map_ = self.calculate_metrics(reranked_contents, answer, answer_vec)
            page_hit, page_mrr = self.calculate_page_match(raw_results, correct_file, correct_page)

            precision_list.append(p)
            recall_list.append(r)
            mrr_list.append(mrr)
            hit_list.append(hit)
            map_list.append(map_)
            page_match_list.append(page_hit)
            page_mrr_list.append(page_mrr)

            results.append({
                "query": query,
                "correct_answer": answer,
                "correct_file": correct_file,
                "correct_page": correct_page,
                "retrieved_documents": [extract_doc_info(doc) for doc in reranked_docs],
                "precision": p,
                "recall": r,
                "mrr": mrr,
                "hit@k": hit,
                "map": map_,
                "page_match": page_hit,
                "page_mrr": page_mrr
            })

            if verbose:
                logger.info(f"\nQuery: {query}")
                logger.info(f"Precision: {p:.4f} | Recall: {r:.4f} | MRR: {mrr:.4f} | MAP: {map_:.4f} | Page Match: {page_hit}")

        avg = lambda lst: sum(lst) / len(lst)

        metrics = {
            "avg_precision": avg(precision_list),
            "avg_recall": avg(recall_list),
            "avg_mrr": avg(mrr_list),
            "avg_hit@k": avg(hit_list),
            "avg_map": avg(map_list),
            "avg_page_match": avg(page_match_list),
            "avg_page_mrr": avg(page_mrr_list)
        }

        logger.info("\n===== Overall Evaluation =====")
        for k, v in metrics.items():
            logger.info(f"{k}: {v:.4f}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "results": results}, f, ensure_ascii=False, indent=2)

        logger.info(f"\nResults saved to {save_path}")
        
if __name__ == "__main__":
    evaluator = RetrievalEvaluator(
        vector_db_path=r"output\v1\FAISS\bge_large_v1.5\faiss_index",
        test_data_path=r"D:\desktop\code\QA\QA1.json"
    )
    evaluator.evaluate(save_path="./retrieval_result/v3/eval_result.json", verbose=True)
