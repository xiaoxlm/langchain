import torch
import wandb
wandb.login(key="")
run = wandb.init(
    project='Lora-R1-Distill-Qwen-14B on Medical COT Dataset',
    job_type="training",
    anonymous="allow"
)

from unsloth import FastLanguageModel
max_seq_length = 2048
dtype = None
load_in_4bit = True

# 加载模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    local_files_only=True,
    max_seq_length = max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

print(model)


# 在微调前做一次推理
prompt_style = """以下是描述任务的指令，以及提供更多上下文的输入。  
  请写出恰当完成该请求的回答。  
  在回答之前，请仔细思考问题，并创建一个逐步的思维链，以确保回答合乎逻辑且准确。
  
  ### Instruction:  
  你是一位在临床推理、诊断和治疗计划方面具有专业知识的医学专家。  
  请回答以下医学问题。
  
  ### Question:  {}
  ### Response:  <think>{}"""

train_prompt_style = prompt_style + """  
  </think>  
  {}"""

question = "一名70岁的男性患者因胸痛伴呕吐16小时就医，心电图显示下壁导联和右胸导联ST段抬高0.1~0.3mV，经补液后血压降至80/60mmHg，患者出现呼吸困难和不能平卧的症状，体检发现双肺有大量水泡音。在这种情况下，最恰当的药物处理是什么？"

FastLanguageModel.for_inference(model)
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,
    use_cache=True,
)
response = tokenizer.batch_decode(outputs)
print("### 微调前模型推理结果：")
print(response[0].split("### Response:")[1])


EOS_TOKEN = tokenizer.eos_token # 添加结束符标记

#格式化提示函数,用于处理数据集中的示例
def formatting_prompts_func(examples):
    # 从examples中提取问题、思维链和回答
    inputs = examples["Question"]      # 医学问题列表
    cots = examples["Complex_CoT"]     # 思维链列表
    outputs = examples["Response"]     # 回答列表
    # 存储格式化后的文本
    texts = []

    # 遍历每个示例,将问题、思维链和回答组合成指定格式
    for input, cot, output in zip(inputs, cots, outputs):
        # 使用train_prompt_style模板格式化文本,并添加结束符
        text = train_prompt_style.format(input, cot, output) + EOS_TOKEN
        texts.append(text)


    # 返回格式化后的文本字典
    return {
        "text": texts,
    }

# 加载数据集并应用格式化
from datasets import load_dataset
dataset = load_dataset(
    "json",  # 指定数据格式为 JSON
    data_files="/datasets/medical-o1-reasoning-SFT/medical_o1_sft_Chinese.json",
    trust_remote_code=True  # 兼容 remote code 的行为
)

# 如果返回的是 DatasetDict，则取出 "train" 这一部分
if isinstance(dataset, dict):
    dataset = dataset["train"]

dataset = dataset.map(formatting_prompts_func, batched = True,)
print(dataset)  # 查看数据集结构

# 配置LoRA微调参数
model = FastLanguageModel.get_peft_model(
    model,   # 原始模型
    r=16,    # LoRA秩,建议: 8-32之间
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16, # LoRA缩放因子
    lora_dropout=0,  # LoRA层的dropout率
    bias="none",   # none表示不微调 bias 参数
    use_gradient_checkpointing="unsloth",   # 使用unsloth优化版本
    random_state=3407,  # 随机数种子,用于结果复现
    use_rslora=False,    # 不使用rank-stabilized LoRA
    loftq_config=None, # 不使用LoFTQ技术
)
print(model)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
# 初始化 SFT 训练器
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",  # 数据集字段的名称
    max_seq_length=max_seq_length,  # 最大序列长度
    dataset_num_proc=2,  # 数据集处理的并行进程数，提高CPU利用率
    args=TrainingArguments(
        per_device_train_batch_size=2,  # 每个GPU的训练批次大小
        gradient_accumulation_steps=4,   # 梯度累积步数,用于模拟更大的batch size
        warmup_steps=5,  # 预热步数,逐步增加学习率
        learning_rate=2e-4,  # 学习率
        lr_scheduler_type="linear",  # 线性学习率调度器
        max_steps=60,    # 最大训练步数
        fp16=not is_bfloat16_supported(),  # 如果不支持bf16则使用fp16
        bf16=is_bfloat16_supported(),      # 如果支持则使用bf16
        logging_steps=10,  # 每10步记录一次日志
        optim="adamw_8bit",  # 使用8位AdamW优化器节省显存，几乎不影响训练效果
        weight_decay=0.01,   # 权重衰减系数
        seed=8137,  # 随机数种子
        output_dir="outputs",
        run_name="medical-o1-sft-experiment",  # 显式设置 wandb 运行名称，避免警告
        ),
)

# 开始训练
trainer.train()

# 训练后的模型做一次推理
FastLanguageModel.for_inference(model)
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")# 生成回答
outputs = model.generate(
    input_ids=inputs.input_ids, # id序列
    attention_mask=inputs.attention_mask,  # 注意力掩码
    max_new_tokens=1200, # 最大新token数
    use_cache=True, # 使用KV缓存加速生成
)

response = tokenizer.batch_decode(outputs)
print("### 训练后模型推理结果：")
print(response[0].split("### Response:")[1])

