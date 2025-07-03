# 学习文件: https://github.com/datawhalechina/llm-cookbook/blob/d53a36772c76fef0f55295af102615dd498a60cd/content/%E5%BF%85%E4%BF%AE%E4%B8%89-LangChain%20for%20LLM%20Application%20Development/5.%E5%9F%BA%E4%BA%8E%E6%96%87%E6%A1%A3%E7%9A%84%E9%97%AE%E7%AD%94%20Question%20and%20Answer.ipynb


from langchain_openai import ChatOpenAI
import tiktoken
from typing import Tuple
from aimodel import AliEmbeddings, llm


from langchain.embeddings.base import Embeddings
import requests

from langchain_community.document_loaders import CSVLoader
from IPython.display import display, Markdown

file = 'data/OutdoorClothingCatalog_10002.csv'
loader = CSVLoader(file_path=file)
docs = loader.load()
# print(docs[0])

embeddings = AliEmbeddings()
embed = embeddings.embed_query("Hi my name is Harrison")
print(len(embed))
print(embed[:5])


from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator

q = "Please list all your shirts with sun protection in a table in markdown and summarize each one"

# 方式一
## 创建索引
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings
).from_documents(docs)

# aa = DocArrayInMemorySearch.from_documents()

response = index.query(question=q, llm=llm)
print("得到的llm结果：", response)
# print(display(Markdown(response)))

# 方式二
# db = DocArrayInMemorySearch.from_documents(docs, embeddings)
# docs = db.similarity_search(q)
# qdocs = "".join([docs[i].page_content for i in range(len(docs))])
# response = llm.invoke(f"{qdocs} Question: Please list all your \
# shirts with sun protection in a table in markdown and summarize each one.")
# print(response)

# 方式三
# retriever = db.as_retriever()
# from langchain.chains import RetrievalQA
#
# qa_stuff = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=retriever,
#     verbose=True
# )
#
# response = qa_stuff.invoke(q)
# print(response)


# addtion method
# 1. map_reduce  第二常用，可以用来生成摘要
# 2. stuff 最常用
# 3. map_rerank
# 4. refine