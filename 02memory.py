# 学习文件: https://github.com/datawhalechina/llm-cookbook/blob/d53a36772c76fef0f55295af102615dd498a60cd/content/%E5%BF%85%E4%BF%AE%E4%B8%89-LangChain%20for%20LLM%20Application%20Development/3.%E5%82%A8%E5%AD%98%20%20Memory.ipynb

from langchain_openai import ChatOpenAI
import tiktoken
from typing import Tuple
from aimodel import llm

from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory


# 1. 初学
# memory = ConversationBufferMemory()
# conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
#
#
# conversation.predict(input="Hi, my name is Andrew")
# conversation.predict(input="What is 1+1?")
# print(conversation.predict(input="What is my name?"))
# print(memory.buffer)

# 2. 直接添加内容到储存缓存
# memory = ConversationBufferMemory()
# memory.save_context({"input": "Hi"}, {"output": "What's up"})
# print(memory.buffer)
# memory.load_memory_variables({})
#
# memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
# print(memory.load_memory_variables({}))


# 3. 对话缓存窗口储存
from langchain.memory import ConversationBufferWindowMemory
# 3.1、初识对话缓存窗口储存
# k 为窗口参数，k=1表明只保留一个对话记忆
# memory = ConversationBufferWindowMemory(k=1)
# memory.save_context({"input": "Hi"}, {"output": "What's up"})
# memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
# print(memory.load_memory_variables({}))

# 3.2 在对话链中应用窗口储存
# memory = ConversationBufferWindowMemory(k=1)
# conversation = ConversationChain(llm=llm, memory=memory, verbose=False)
#
# print(conversation.predict(input="Hi, my name is Andrew"))
# print(conversation.predict(input="What is 1+1?"))
# print(conversation.predict(input="What is my name?"))

# 4. 添加对话到Token缓存储存
# from langchain.memory import ConversationTokenBufferMemory
# memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=30)
# memory.save_context({"input": "AI is what?!"}, {"output": "Amazing!"})
# memory.save_context({"input": "Backpropagation is what?"}, {"output": "Beautiful!"})
# memory.save_context({"input": "Chatbots are what?"}, {"output": "Charming!"})
# print(memory.load_memory_variables({}))

# 5. 对话摘要缓存储存
from langchain.memory import ConversationSummaryBufferMemory
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

# 使用对话摘要缓存记忆
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=300)
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
memory.save_context(
    {"input": "What is on the schedule today?"}, {"output": f"{schedule}"}
)
print(memory.load_memory_variables({})['history'])

conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
print(conversation.predict(input="What would be a good demo to show?"))
print("=======")
print(memory.load_memory_variables({})['history'])