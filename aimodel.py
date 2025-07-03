
import tiktoken
from typing import Tuple
from langchain_openai import ChatOpenAI
from openai import OpenAI

import yaml
import os
from typing import Final

ALI_API_KEY: Final = "ali_api_key"
GEMINI_API_KEY: Final = "gemini_api_key"

def _load_api_keys():
    """从ai_key.yml文件加载API密钥"""
    try:
        with open('ai_key.yml', 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        print("警告: ai_key.yml文件未找到")
        return {}
    except yaml.YAMLError as e:
        print(f"警告: 解析ai_key.yml文件时出错: {e}")
        return {}

# 加载API密钥
_api_keys: dict = _load_api_keys()

class ChatOpenAIIn05(ChatOpenAI):
    def _get_encoding_model(self) -> Tuple[str, tiktoken.Encoding]:
        """
        Override the method to return a hardcoded valid model and its encoding.
        """
        # Set the model to a valid one to avoid errors
        return self.model_name, tiktoken.encoding_for_model(self.model_name)
    

llm = ChatOpenAIIn05(
    # model="qwen2.5-72b-instruct",
    model="qwen-max",
    temperature=0,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=_api_keys.get(ALI_API_KEY, '')
)

from langchain.embeddings.base import Embeddings
import requests

class AliEmbeddings(Embeddings):
    def __init__(self):
        self.api_key = _api_keys.get(ALI_API_KEY, '')
        self.endpoint = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
        # self.endpoint = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def _call_api(self, text: str) -> list[float]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {
            "input": text,
            "encoding_format": "float",
            # "dimensions": "1024",
            "model": "text-embedding-v1"  # 根据阿里模型名修改
            # "model": "text-embedding-v4"  # 根据阿里模型名修改
        }
        response = requests.post(self.endpoint, headers=headers, json=data)
        response.raise_for_status()  # 检查请求是否成功
        json = response.json()
        return json["data"][0]["embedding"]  # 根据实际返回结构调整

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._call_api(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._call_api(text)

from google import genai
class GeminiEmbeddings(Embeddings):
    def __init__(self):
        self.client = genai.Client(api_key=_api_keys.get(GEMINI_API_KEY, ''))
        # self.model = "gemini-embedding-exp-03-07"
        self.model = "text-embedding-004"

    def _call_api(self, text: str) -> list[float]:
        result = self.client.models.embed_content(
            model=self.model,
            contents=text,
        )
        return result.embeddings[0].values

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._call_api(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._call_api(text)