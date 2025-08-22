import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownTextSplitter
from dataclasses import dataclass
from typing import List
import json
import uuid
import logging
from langchain.docstore.document import Document

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

@dataclass
class SplitConfig:
    chunk_size: int = 400
    chunk_overlap: int = 40
    separators: List[str] = ('\n\n\n', '\n\n')
    force_split: bool = False
    output_format: str = 'json'
    cache_dir: str = './cache'

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

def split_docs_with_config(documents, config: SplitConfig, cache_name="all_docs"):
    os.makedirs(config.cache_dir, exist_ok=True)
    filename = f"{cache_name}_split_{config.chunk_size}_{config.chunk_overlap}.{config.output_format}"
    filepath = os.path.join(config.cache_dir, filename)

    if os.path.exists(filepath) and not config.force_split:
        logging.info("Found existing cache. Loading...")
        return load_chunks_from_json(filepath)

    splitter = MarkdownTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        chunk.metadata['uuid'] = str(uuid.uuid4())

    if config.output_format == 'json':
        save_chunks_as_json(chunks, filepath)

    return chunks

def load_multiple_documents_from_dir(directory: str, encoding='utf-8') -> List[Document]:
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            file_path = os.path.join(directory, filename)
            loader = TextLoader(file_path, encoding=encoding)
            file_docs = loader.load()
            for doc in file_docs:
                doc.metadata['source_file'] = filename
            docs.extend(file_docs)
    logging.info(f"Loaded {len(docs)} documents from {directory}")
    return docs

input_dir = "D:/desktop/code/data"  
documents = load_multiple_documents_from_dir(input_dir)

split_config = SplitConfig(
    chunk_size=400,
    chunk_overlap=40,
    separators=['\n\n\n', '\n\n'],
    force_split=False,
    output_format='json',
    cache_dir='./cache'
)

logging.info(f"Splitting {len(documents)} documents...")
split_docs = split_docs_with_config(documents, split_config, cache_name="all_docs")
