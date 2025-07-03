# 学习文件: https://github.com/datawhalechina/llm-cookbook/blob/d53a36772c76fef0f55295af102615dd498a60cd/content/%E5%BF%85%E4%BF%AE%E4%B8%89-LangChain%20for%20LLM%20Application%20Development/4.%E6%A8%A1%E5%9E%8B%E9%93%BE%20Chains.ipynb

from langchain_openai import ChatOpenAI
import tiktoken
from typing import Tuple
from aimodel import llm


# 2. 大语言模型链
# from langchain_community.chat_models import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
# from langchain.chains.llm import LLMChain
# #
import pandas as pd
df = pd.read_csv('data/Data.csv')

# prompt = ChatPromptTemplate.from_template(
#     "What is the best name to describe \
#     a company that makes {product}?"
# )
#
# chain = LLMChain(llm=llm, prompt=prompt)
#
# product = "Queen Size Sheet Set"
# print(chain.invoke(product))

# 3. 顺序链
# 3.1 简单顺序链
# from langchain.chains.sequential import SimpleSequentialChain
#
# first_prompt = ChatPromptTemplate.from_template(
#     "What is the best name to describe \
#     a company that makes {product}?"
# )
# # Chain 1
# chain_one = LLMChain(llm=llm, prompt=first_prompt)
#
# second_prompt = ChatPromptTemplate.from_template(
#     "Write a 20 words description for the following \
#     company:{company_name}"
# )
# # chain 2
# chain_two = LLMChain(llm=llm, prompt=second_prompt)
#
# overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two], verbose=True)
#
# product = "Queen Size Sheet Set"
# print(overall_simple_chain.run(product))

# 3.2 顺序链
# from langchain.chains.sequential import SequentialChain
from langchain.prompts import ChatPromptTemplate   #导入聊天提示模板
from langchain.chains.llm import LLMChain    #导入LLM链。
# first_prompt = ChatPromptTemplate.from_template(
#     "Translate the following review to english:"
#     "\n\n{Review}"
# )
# # chain 1: 输入：Review 输出： 英文的 Review
# chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key="English_Review")
#
#
# second_prompt = ChatPromptTemplate.from_template(
#     "Can you summarize the following review in 1 sentence:"
#     "\n\n{English_Review}"
# )
# # chain 2: 输入：英文的Review   输出：总结
# chain_two = LLMChain(llm=llm, prompt=second_prompt, output_key="summary")
#
#
# third_prompt = ChatPromptTemplate.from_template(
#     "What language is the following review:\n\n{Review}"
# )
# # chain 3: 输入：Review  输出：语言
# chain_three = LLMChain(llm=llm, prompt=third_prompt, output_key="language")
#
#
# fourth_prompt = ChatPromptTemplate.from_template(
#     "Write a follow up response to the following "
#     "summary in the specified language:"
#     "\n\nSummary: {summary}\n\nLanguage: {language}"
# )
# # chain 4: 输入： 总结, 语言    输出： 后续回复
# chain_four = LLMChain(llm=llm, prompt=fourth_prompt, output_key="followup_message")
#
#
# overall_chain = SequentialChain(
#     chains=[chain_one, chain_two, chain_three, chain_four],
#     input_variables=["Review"],
#     output_variables=["English_Review", "summary","followup_message"],
#     verbose=True
# )
#
# review = df.Review[5]
# print(overall_chain(review))

# 4. 路由链
from langchain.chains.router import MultiPromptChain  #导入多提示链
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.prompts import PromptTemplate

#第一个提示适合回答物理问题
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise\
and easy to understand manner. \
When you don't know the answer to a question you admit\
that you don't know.

Here is a question:
{input}"""


#第二个提示适合回答数学问题
math_template = """You are a very good mathematician. \
You are great at answering math questions. \
You are so good because you are able to break down \
hard problems into their component parts, 
answer the component parts, and then put them together\
to answer the broader question.

Here is a question:
{input}"""


#第三个适合回答历史问题
history_template = """You are a very good historian. \
You have an excellent knowledge of and understanding of people,\
events and contexts from a range of historical periods. \
You have the ability to think, reflect, debate, discuss and \
evaluate the past. You have a respect for historical evidence\
and the ability to make use of it to support your explanations \
and judgements.

Here is a question:
{input}"""


#第四个适合回答计算机问题
computerscience_template = """ You are a successful computer scientist.\
You have a passion for creativity, collaboration,\
forward-thinking, confidence, strong problem-solving capabilities,\
understanding of theories and algorithms, and excellent communication \
skills. You are great at answering coding questions. \
You are so good because you know how to solve a problem by \
describing the solution in imperative steps \
that a machine can easily interpret and you know how to \
choose a solution that has a good balance between \
time complexity and space complexity. 

Here is a question:
{input}"""

prompt_infos = [
    {
        "name": "physics",
        "description": "Good for answering questions about physics",
        "prompt_template": physics_template
    },
    {
        "name": "math",
        "description": "Good for answering math questions",
        "prompt_template": math_template
    },
    {
        "name": "History",
        "description": "Good for answering history questions",
        "prompt_template": history_template
    },
    {
        "name": "computer science",
        "description": "Good for answering computer science questions",
        "prompt_template": computerscience_template
    }
]

destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)


default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)

MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt \
names specified below OR it can be "DEFAULT" if the input is not\
well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>

eg:
<< INPUT >>
"What is black body radiation?"
<< OUTPUT >>
```json
{{{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
}}}}
```
"""


router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt)


chain = MultiPromptChain(router_chain=router_chain,    #l路由链路
                         destination_chains=destination_chains,   #目标链路
                         default_chain=default_chain,      #默认链路
                         verbose=True
                        )


# print(chain.run("What is black body radiation?"))
# print(chain.run("what is 2 + 2"))
print(chain.run("Why does every cell in our body contain DNA?"))