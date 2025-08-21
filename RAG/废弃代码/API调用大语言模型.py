from langchain.chains.question_answering import load_qa_chain
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from typing import Any, List, Optional, Dict, Mapping
from pydantic import BaseModel, Field
import json
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import concurrent.futures
import requests
import time
import hashlib
import hmac
import base64
import datetime

# 火山引擎API配置
VOLCENGINE_CONFIG = {
    "api_key": "3757050b-82ca-41c0-a872-065b7913ff87",
    "model": "Doubao-1.5-thinking-vision-pro",
    "temperature": 0.7,
    "max_tokens": 2000,
    "top_p": 0.8,
    "api_host": "https://open.volcengineapi.com",
    "api_path": "/api/v3/llm/chat/completions"
}

class VolcengineLLM(LLM, BaseModel):
    """火山引擎LLM封装类"""
    
    api_key: str = Field(default=VOLCENGINE_CONFIG["api_key"])
    model: str = Field(default=VOLCENGINE_CONFIG["model"])
    temperature: float = Field(default=VOLCENGINE_CONFIG["temperature"])
    max_tokens: int = Field(default=VOLCENGINE_CONFIG["max_tokens"])
    top_p: float = Field(default=VOLCENGINE_CONFIG["top_p"])
    api_host: str = Field(default=VOLCENGINE_CONFIG["api_host"])
    api_path: str = Field(default=VOLCENGINE_CONFIG["api_path"])

    def _generate_signature(self, timestamp: str) -> str:
        """生成API签名"""
        string_to_sign = f"{timestamp}\n{self.api_path}"
        hmac_obj = hmac.new(
            self.api_key.encode('utf-8'),
            string_to_sign.encode('utf-8'),
            hashlib.sha256
        )
        signature = base64.b64encode(hmac_obj.digest()).decode('utf-8')
        return signature

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """调用火山引擎API"""
        timestamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        signature = self._generate_signature(timestamp)

        headers = {
            "Authorization": f"HMAC-SHA256 AccessKey={self.api_key}, SignedHeaders=host, Signature={signature}",
            "Content-Type": "application/json",
            "Host": self.api_host.replace("https://", ""),
            "X-Date": timestamp
        }

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p
        }

        try:
            response = requests.post(
                f"{self.api_host}{self.api_path}",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"调用火山引擎API时出错: {str(e)}")
            raise

    @property
    def _llm_type(self) -> str:
        """返回LLM类型"""
        return "volcengine"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回用于标识LLM的参数"""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p
        }

def init_volcengine_llm():
    """初始化火山引擎LLM"""
    try:
        llm = VolcengineLLM(
            api_key=VOLCENGINE_CONFIG["api_key"],
            model=VOLCENGINE_CONFIG["model"],
            temperature=VOLCENGINE_CONFIG["temperature"],
            max_tokens=VOLCENGINE_CONFIG["max_tokens"],
            top_p=VOLCENGINE_CONFIG["top_p"]
        )
        return llm
    except Exception as e:
        print(f"初始化火山引擎LLM时出错: {str(e)}")
        raise

def process_qa_chain(docs, query, qa_chain):
    """处理单个查询的问答链"""
    try:
        answer = qa_chain.run(input_documents=docs, question=query)
        return answer
    except Exception as e:
        print(f"生成答案时出错: {str(e)}")
        return f"Error generating answer: {e}"

def parallel_process_qa(items, qa_chain, num_processes=None):
    """并行处理多个查询的问答链"""
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)  # 保留一个CPU核心
    
    # 创建进程池
    with Pool(processes=num_processes) as pool:
        # 使用partial固定qa_chain参数
        process_func = partial(process_qa_chain, qa_chain=qa_chain)
        # 并行处理所有查询
        results = pool.starmap(process_func, [(item['compressed_docs'], item['query']) for item in items])
    
    return results

def prepare_qa_data(item):
    """准备问答链的输入数据"""
    compressed_docs_for_qa = []
    for doc_dict in item['retrieved_documents']:
        if 'compressed_content' in doc_dict:
            compressed_docs_for_qa.append(Document(
                page_content=doc_dict['compressed_content'],
                metadata={
                    "uuid": doc_dict.get("uuid", "未指定"),
                    "source_file": doc_dict.get("source_file", ""),
                    "page": doc_dict.get("page", "未指定")
                }
            ))
    return {
        'compressed_docs': compressed_docs_for_qa,
        'query': item['query']
    }


llm = init_volcengine_llm()