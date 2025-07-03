from langchain_openai import ChatOpenAI

chat = ChatOpenAI(
    model="qwen2.5-72b-instruct",
    temperature=0.0,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=""
)



# 首先，构造一个提示模版字符串：`template_string`
template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""
from langchain.prompts import ChatPromptTemplate
# 然后，我们调用`ChatPromptTemplatee.from_template()`函数将
# 上面的提示模版字符`template_string`转换为提示模版`prompt_template`

prompt_template = ChatPromptTemplate.from_template(template_string)


print("\nprompt_template prompt:", prompt_template.messages[0].prompt)


customer_style = """American English \
in a calm and respectful tone
"""

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse, \
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""


# 使用提示模版
customer_messages = prompt_template.format_messages(
                    style=customer_style,
                    text=customer_email)
# 打印客户消息类型
print("客户消息类型:",type(customer_messages),"\n")

# 打印第一个客户消息类型
print("第一个客户客户消息类型类型:", type(customer_messages[0]),"\n")

# 打印第一个元素
print("第一个客户客户消息 ", customer_messages[0],"\n")

customer_response = chat.invoke(customer_messages)
print(customer_response.content)

