import os
import re
import json
import uuid
import glob
import logging
from dataclasses import dataclass
from typing import List
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownTextSplitter

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


@dataclass
class SplitConfig:
    chunk_size: int = 400
    chunk_overlap: int = 40
    force_split: bool = False
    output_format: str = 'json'
    cache_dir: str = './cache'


class MarkdownPageSplitter:
    def __init__(self, file_path, config: SplitConfig):
        self.file_path = file_path
        self.config = config
        self.source_file = os.path.basename(file_path)
        os.makedirs(config.cache_dir, exist_ok=True)
        # 每个文件独立缓存
        self.cache_file = os.path.join(
            config.cache_dir,
            f"{os.path.splitext(self.source_file)[0]}_split_{config.chunk_size}_{config.chunk_overlap}.{config.output_format}"
        )

    def split_by_page(self, text):
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

    def save_as_json(self, chunks):
        data = [
            {
                "uuid": chunk.metadata['uuid'],
                "content": chunk.page_content,
                "metadata": chunk.metadata
            }
            for chunk in chunks
        ]
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"Chunks saved to {self.cache_file}")

    def load_from_json(self):
        with open(self.cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        chunks = [
            Document(page_content=item['content'], metadata=item['metadata'])
            for item in data
        ]
        logging.info(f"Loaded {len(chunks)} chunks from cache for {self.source_file}")
        return chunks

    def split(self):
        if os.path.exists(self.cache_file) and not self.config.force_split:
            return self.load_from_json()

        loader = TextLoader(self.file_path, encoding='utf-8')
        documents = loader.load()

        splitter = MarkdownTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

        all_chunks = []
        for doc in documents:
            pages = self.split_by_page(doc.page_content)
            for page in pages:
                page_num = page["page_num"]
                split_texts = splitter.split_text(page["content"])
                for chunk_text in split_texts:
                    chunk = Document(
                        page_content=chunk_text,
                        metadata={
                            "uuid": str(uuid.uuid4()),
                            "page_num": page_num,
                            "source_file": self.source_file
                        }
                    )
                    all_chunks.append(chunk)

        # if self.config.output_format == 'json':
        #     self.save_as_json(all_chunks)

        logging.info(f"Total chunks created for {self.source_file}: {len(all_chunks)}")
        return all_chunks


def process_folder(folder_path, config: SplitConfig, merged_output_file: str):
    # md_files = glob.glob(os.path.join(folder_path, "*.md"))
    md_files = glob.glob(os.path.join(folder_path, "**", "*.md"), recursive=True)

    logging.info(f"Found {len(md_files)} markdown files in {folder_path}")

    all_chunks = []
    for file_path in md_files:
        logging.info(f"Processing file: {file_path}")
        splitter = MarkdownPageSplitter(file_path, config)
        chunks = splitter.split()
        all_chunks.extend(chunks)

    logging.info(f"Total chunks from all files: {len(all_chunks)}")

    # 保存所有 chunk 到一个总的 json
    data = [
        {
            # "uuid": chunk.metadata['uuid'],
            "content": chunk.page_content,
            "metadata": chunk.metadata
        }
        for chunk in all_chunks
    ]
    with open(merged_output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logging.info(f"All chunks merged and saved to {merged_output_file}")

    return all_chunks

folder_path = r"D:\desktop\code\data"
merged_json_file = './cache/merged_chunks.json'

split_config = SplitConfig(
    chunk_size=400,
    chunk_overlap=40,
    force_split=True,
    output_format='json',
    cache_dir='./cache'
)

# 处理并合并
all_chunks = process_folder(folder_path, split_config, merged_json_file)

# 快速检查
# print(f"Total merged chunks: {len(all_chunks)}")
# for i, chunk in enumerate(all_chunks[:3]):
#     print(f"\nChunk {i+1}")
#     print(f"UUID: {chunk.metadata['uuid']}")
#     print(f"Page Num: {chunk.metadata['page_num']}")
#     print(f"Source File: {chunk.metadata['source_file']}")
#     print(f"Content: {chunk.page_content[:100]}...")
