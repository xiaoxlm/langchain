from aimodel import llm, GeminiEmbeddings, AliEmbeddings

from langchain_community.document_loaders import CSVLoader

file = 'data/OutdoorClothingCatalog_10002.csv'
loader = CSVLoader(file_path=file)
data = loader.load()

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import DocArrayInMemorySearch

embeddings = AliEmbeddings()
# embeddings = GeminiEmbeddings()
db = DocArrayInMemorySearch.from_documents(data, embeddings)
retriever = db.as_retriever()
# qa 已经将数据加载进去了
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True,
    chain_type_kwargs={"document_separator": "<<<<>>>>>"}
)

examples = [
    {
        "query": "Do the Cozy Comfort Pullover Set\
        have side pockets?",
        "answer": "Yes"
    },
    {
        "query": "What collection is the Ultra-Lofty \
        850 Stretch Down Hooded Jacket from?",
        "answer": "The DownTek collection"
    }
]

from langchain.evaluation.qa import QAGenerateChain  #导入QA生成链，它将接收文档，并从每个文档中创建一个问题答案对

# import langchain
# langchain.debug = True
example_gen_chain = QAGenerateChain.from_llm(llm)

new_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data[:5]]
)
print("new_examples:", new_examples[0])

fix_examples: list[dict[str, str]] = []
for eg in new_examples:
    if "qa_pairs" in eg:
        q_a: dict = eg["qa_pairs"]
        fix_examples.append({
            "query": q_a['query'],
            "answer": q_a["answer"]})

# 使用列表推导式直接合并，避免中间变量
# examples.extend([
#     {"query": eg["qa_pairs"]["query"], "answer": eg["qa_pairs"]["answer"]}
#     for eg in new_examples
#     if "qa_pairs" in eg
# ])

examples += fix_examples

import langchain
# langchain.debug = True  # 利用debug模式，可以打印出中间过程
# response = qa.invoke(examples[0]["query"])
# print(response)

# 利用大模型来评分
from langchain.evaluation.qa import QAEvalChain

predictions = qa.apply(examples)
eval_chain = QAEvalChain.from_llm(llm)
graded_outputs = eval_chain.evaluate(examples, predictions)  #在此链上调用evaluate，进行评估
for i, eg in enumerate(examples):
    print(f"*Example {i}:")
    print("*Question: " + predictions[i]['query'])
    print("*Real Answer: " + predictions[i]['answer'])
    print("*Predicted Answer: " + predictions[i]['result'])
    print("*Predicted Grade: " + graded_outputs[i]['results'])
    print()

# 可以在 settings.json 添加指定 python 路径
# {
#   "code-runner.executorMap": {
#     "python": "/Users/你的用户名/miniconda3/envs/prompt/bin/python"
#   }
# }
