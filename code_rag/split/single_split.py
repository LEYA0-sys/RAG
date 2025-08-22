import os
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownTextSplitter
from dataclasses import dataclass
from typing import List
import json
import uuid
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

model_path = 'BAAI/bge-large-zh-v1.5'
output_dir = os.path.join('output', f'v1')

loader=TextLoader("D:\desktop\code\data\宏利環球貨幣保障計劃 保單條款_2022_05.md", encoding='utf-8')
documents=loader.load()

#配置类 SplitConfig，参数配置
@dataclass
class SplitConfig:
    chunk_size: int = 400
    chunk_overlap: int = 40
    separators: List[str] = ('\n\n\n', '\n\n')
    force_split: bool = False
    output_format: str = 'json' 
    cache_dir: str = './cache'

##将文本分块结果保存为json
def save_chunks_as_json(chunks, filepath):
    data = [
        {
            "uuid": chunk.metadata.get('uuid', str(uuid.uuid4())),
            "content": chunk.page_content,
            "metadata": chunk.metadata
        }
        for chunk in chunks
    ]
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

##加载分块
def load_chunks_from_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    chunks = []
    for item in data:
        doc = Document(
            page_content=item['content'],
            metadata=item['metadata']
        )
        chunks.append(doc)
    logging.info(f"Loaded {len(chunks)} chunks from {filepath}")
    return chunks



def split_docs_with_config(documents, config: SplitConfig):
    os.makedirs(config.cache_dir, exist_ok=True)
    filename = f"split_{config.chunk_size}_{config.chunk_overlap}.{config.output_format}"
    filepath = os.path.join(config.cache_dir, filename)
    
    if os.path.exists(filepath) and not config.force_split:
        logging.info("Found existing cache. Loading...")
        if config.output_format == 'json':
            return load_chunks_from_json(filepath)
    
    splitter=MarkdownTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        chunk.metadata['uuid'] = str(uuid.uuid4())
        if 'source' in chunk.metadata:
          chunk.metadata['source_file'] = os.path.basename(chunk.metadata['source'])
    if config.output_format == 'json':
        save_chunks_as_json(chunks, filepath)
    return chunks

split_config = SplitConfig(
    chunk_size=400,
    chunk_overlap=40,
    separators=['\n\n\n', '\n\n'],
    force_split=False,
    output_format='json',
    cache_dir='./cache'
)
logging.info(f"Splitting documents with chunk_size={split_config.chunk_size}, overlap={split_config.chunk_overlap}")
split_doc = split_docs_with_config(documents, split_config)