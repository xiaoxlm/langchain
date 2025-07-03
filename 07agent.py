from aimodel import llm
from langchain_community.document_loaders import CSVLoader
from langchain_community.agent_toolkits.load_tools import load_tools
# from langchain.agents import load_tools, initialize_agent,AgentExecutor,AgentType
from langchain.agents import initialize_agent,AgentExecutor,AgentType
import warnings
warnings.filterwarnings("ignore")


tools: list = load_tools(["llm-math", "wikipedia"], llm=llm)

agent: AgentExecutor = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    handle_parsing_errors=True,
    verbose=True)

# agent("What is the 25% of 300?")
# print(result)

question: str = "Tom M. Mitchell is an American computer scientist \
and the Founders University Professor at Carnegie Mellon University (CMU)\
what book did he write?"

# 为什么agent(question) 可以代替agent.invoke({"input": question})
# 这是因为在 LangChain 的 AgentExecutor（以及许多 LangChain 的链和代理类）中，类实现了 Python 的 call 方法，使其实例可以像函数一样被调用。这种设计让你可以直接用 agent(question) 的方式调用，而不必每次都写 agent.invoke({"input": question})。
# 在 AgentExecutor 类的源码中没有直接实现 __call__ 方法。那么为什么 agent(question) 依然可以工作？原因在于它继承自 LangChain 的 Chain 基类，而 Chain 实现了 __call__ 方法。
# result = agent(question) 

# print(result)


## python agent
from langchain_experimental.agents.agent_toolkits.python.base import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
python_agent: AgentExecutor = create_python_agent(
    llm=llm,
    tool=PythonREPLTool(),
    verbose=True
)

customer_list: list = [
    ["Harrison", "Chase"],
    ["Lang", "Chain"],
    ["Dolly", "Too"],
    ["Elle", "Elem"],
    ["Geoff", "Fusion"],
    ["Trance", "Former"],
    ["Jen", "Ayai"]
]

# res = python_agent.invoke(f"""Sort these customers by \
# last name and then first name \
# and print the output: {customer_list}""")

# print(res)


## 自定义agent
from langchain.agents import tool
from datetime import date

@tool
def time(text: str) -> str:
    """
    Returns todays date, use this for any \
    questions related to knowing todays date. \
    The input should always be an empty string, \
    and this function will always return todays \
    date - any date mathmatics should occur \
    outside this function.
    """
    return str(date.today())


agent: AgentExecutor = initialize_agent(
    tools + [time], 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
    handle_parsing_errors=True,
    verbose=True)

agent.invoke({"input": "What is today's date?"})