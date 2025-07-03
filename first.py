from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

client = OpenAI(api_key="",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
# client = OpenAI(api_key="sk-T4CdcQa8ZpIH9dlx52230fF534F1480d86854c23Bb8b544e",
#                 base_url="https://free.v36.cm")


def get_completion(prompt, model="qwen-max"): # model="qwen2.5-72b-instruct"
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content



text = """
在 1957 年，于 1956 年成立的三个合伙企业的表现超越了市场。年初道琼斯指数为 499 点， 年底则为 435 点，下跌 65 点。考虑股利分红的因素，实际市场下跌了约 42 点或 8.4%。大 部分投资基金的表现并不比市场要好 ，实际上据我所知 ，今年没有任何投资基金获得正收益 。 而我们三个合伙企业在今年分别获得了 6.2%、7.8%和 25%的净资产增长。当然这样的数字 显然会引起一起疑问，尤其对于前两个合伙企业的投资人而言更是如此 。这种情况的出现纯
粹是运气使然。获得最高收益的企业成立的时间最晚 ，正好赶上市场下跌，同时部分股票出 现了较为吸引人的价格。而前两个公司的投资头寸已经较高，因而无法获得这样的好处。
"""

prompt = f"""
Summarize the text delimited by triple backticks \
into a single sentence.
```{text}```
"""


prompt = f"""
Generate a list of 3 made-up book titles along with their authors and genres. \
Provide them in JSON format with the following keys:
book_id, title, author, genre
"""

prompt = f"""
Your task is to answer in a consistent style.

<child>: Teach me about patience.

<grandparent>: The river that carves the deepest \
valley flows from a modest spring; the \
grandest symphony originates from a single note; \
the most intricate tapestry begins with a solitary thread.

<child>: Teach me about resilience.
"""

# 模型幻觉用例
prompt = f"""
Tell me about AeroGlide UltraSlim Smart Toothbrush by Boie
"""
# 解决思路
# 1. 从文本中找到任何相关的引用
# 2. 使用这些引用来回答问题


# 大模型的回答往往很难精确到确定数值：比如帮我总结一下文本，内容不超过50字。但很有可能是在51个字

response = get_completion(prompt)
print(response)