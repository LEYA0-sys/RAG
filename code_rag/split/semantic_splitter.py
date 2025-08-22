import os
import re
import json
import uuid
import glob
import logging
from dataclasses import dataclass
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


@dataclass
class SplitConfig:
    model_name: str = 'paraphrase-MiniLM-L6-v2'
    similarity_threshold: float = 0.75
    cache_dir: str = './cache'
    force_split: bool = False
    output_format: str = 'json'


class SemanticPageSplitter:
    def __init__(self, file_path: str, config: SplitConfig):
        self.file_path = file_path
        self.config = config
        self.source_file = os.path.splitext(os.path.basename(file_path))[0]
        self.model = SentenceTransformer(config.model_name)
        os.makedirs(config.cache_dir, exist_ok=True)

        self.cache_file = os.path.join(
            config.cache_dir,
            f"{os.path.splitext(self.source_file)[0]}_semantic.json"
        )

    def split_by_page(self, text: str) -> List[Dict]:
        pattern = r'##\s*第\s*(\d+)\s*页'
        matches = list(re.finditer(pattern, text))
        pages = []
        for idx, match in enumerate(matches):
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            page_num = match.group(1)
            page_content = text[start:end].strip()
            pages.append({
                "page_num": page_num,
                "content": page_content
            })
        return pages

    def semantic_split(self, page_text: str, page_num: str) -> List[Dict]:
        sentences = [s.strip() for s in re.split(r'[。！？]\s*', page_text) if s.strip()]
        if len(sentences) <= 1:
            return [{
                "uuid": str(uuid.uuid4()),
                "content": page_text,
                "metadata": {
                    "page_num": page_num,
                    "source_file": self.source_file
                }
            }]

        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings).cpu().numpy()

        # 聚类划分
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            distance_threshold=1 - self.config.similarity_threshold,
            linkage='average'
        )
        labels = clustering.fit_predict(1 - similarity_matrix)

        clusters = {}
        for label, sentence in zip(labels, sentences):
            clusters.setdefault(label, []).append(sentence)

        chunks = []
        for group in clusters.values():
            chunk_text = '。'.join(group) + '。'
            chunks.append({
                "uuid": str(uuid.uuid4()),
                "content": chunk_text,
                "metadata": {
                    "page_num": page_num,
                    "source_file": self.source_file
                }
            })
        return chunks

    def split(self) -> List[Dict]:
        if os.path.exists(self.cache_file) and not self.config.force_split:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                logging.info(f"Loaded chunks from cache: {self.cache_file}")
                return json.load(f)

        with open(self.file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        all_chunks = []
        pages = self.split_by_page(raw_text)
        for page in pages:
            chunks = self.semantic_split(page["content"], page["page_num"])
            all_chunks.extend(chunks)

        # with open(self.cache_file, 'w', encoding='utf-8') as f:
        #     json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        # logging.info(f"Saved {len(all_chunks)} chunks to {self.cache_file}")

        return all_chunks


def process_folder(folder_path: str, config: SplitConfig, merged_output_file: str):
    md_files = glob.glob(os.path.join(folder_path, "**", "*.md"), recursive=True)
    logging.info(f"Found {len(md_files)} markdown files in {folder_path}")

    all_chunks = []
    for file_path in md_files:
        logging.info(f"Processing {file_path}")
        splitter = SemanticPageSplitter(file_path, config)
        chunks = splitter.split()
        all_chunks.extend(chunks)

    with open(merged_output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    logging.info(f"Merged output saved to {merged_output_file}")
    return all_chunks


# ====== 执行入口 ======

# if __name__ == '__main__':
folder_path = r"D:\desktop\code\data"
merged_output = './cache/semantic_merged_chunks.json'

config = SplitConfig(
    model_name='paraphrase-MiniLM-L6-v2',
    similarity_threshold=0.75,
    force_split=True
)

all_chunks = process_folder(folder_path, config, merged_output)
