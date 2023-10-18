import os
import re
import torch
from datasets import load_dataset
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    StoppingCriteria, 
    StoppingCriteriaList
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# %% [markdown]
# ## Data Exploration

# %%
data_path = "/home/mdafifal.mamun/notebooks/triagerX/notebook/openj9_issues.csv"
df = pd.read_csv(data_path)
df = df.drop("Unnamed: 0", axis=1)

df.head()

# %%
df[5000:5005]

# %%
total_contributors = len(df["assignees"].unique())
print(f"Total contributors: {total_contributors}")

# %%
minimum_contribution = 30

developers = df["assignees"].value_counts()

# %%
filtered_developers = developers.index[developers >= minimum_contribution]

# %%
filtered_df = df[df["assignees"].isin(filtered_developers)]

# %%
filtered_df.to_csv("/home/mdafifal.mamun/notebooks/triagerX/notebook/openj9_issues_cleaned.csv")

# %% [markdown]
# ## Model Configuration

# %%
model_name = "NousResearch/Llama-2-13b-chat-hf"
data_path = "/home/mdafifal.mamun/notebooks/triagerX/notebook/openj9_issues_cleaned.csv"
new_model = "llama-2-13b-openj9"

# Set QLoRA configuration
lora_r = 64 # Attention dimension/rank
lora_alpha = 16
lora_dropout = 0.05

# Set bitsandbytes configuration
use_4bit = True #For  4-bit precision base model loading
bnb_4bit_compute_dtype = "float16" # Compute dtype for 4-bit base models
bnb_4bit_quant_type = "nf4" # Quantization type (fp4 or nf4)
use_nested_quant = False # Activate nested quantization for 4-bit base models (double quantization)


# Set training params
output_dir = "./results"
num_train_epochs = 1
fp16 = False
bf16 = False
per_device_train_batch_size = 8
per_device_eval_batch_size = 8
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = 300
warmup_ratio = 0.03
group_by_length = True # Group sequences into batches with same length saves memory and speeds up training considerably
save_steps = 0
logging_steps = 5

# Set SFT parameters
max_seq_length = None
packing = False # Pack multiple short examples in the same input sequence to increase efficiency
device_map = {"": 0} # Load the entire model on the GPU 0

# %%
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# %% [markdown]
# ## Load Base Model

# %%
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config, # Using it for optimized model loading
    device_map=device_map
)

# %%
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix overflow issue with fp16 training

# %% [markdown]
# ## Prompt Template Generation

# %%
# formatted_text = f"<s><INST>Issue Title:\n{data['issue_title'][i]}" \
#     + f"\nIssue Description:\n{data['description'][i]}\nWho can fix this issue?\n</INST>The issue can be fixed by: {data['owner'][i]}</s>"

import xml.etree.ElementTree as ET

def parse_comments(comments: str):
    xml_like =  ET.ElementTree(ET.fromstring(comments))

def generate_prompt_with_answer(entry, i):
    # comments = entry["comments"] # TBD

    issue_title = entry["issue_title"][i]
    issue_description = entry["issue_body"][i]
    assignees = entry["assignees"][i]
    
    prompt = f"""<s><INST>Suggest a developer from the given developer list based on the issue title and description given below. The name of developers are separated by comma. You have to choose only one based on previous knowledge. If the question cannot be answered using the information provided answer with "I don't know".
Developer List: {", ".join(filtered_developers.values)}

Issue Title: {issue_title}

Issue Description: {issue_description}
=====
Question: Which developer from the given Developer List can fix this issue?</INST>

Answer: {assignees}</s>
"""

    return prompt

def generate_prompt_without_answer(entry):
    issue_title = entry["issue_title"]
    issue_description = entry["issue_body"]

    prompt = f"""<s><INST>Suggest a developer from the given developer list based on the issue title and description given below. The name of developers are separated by comma. You have to choose only one based on previous knowledge. If the question cannot be answered using the information provided answer with "I don't know".
Developer List: {", ".join(filtered_developers.values)}

Issue Title: {issue_title}

Issue Description: {issue_description}
=====
Question: Which developer from the given Developer List can fix this issue?</INST>

Answer: </s>
"""

    return prompt



# %%
print(generate_prompt_with_answer(filtered_df, 477))

# %% [markdown]
# ## Test Base Model

# %%
device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
stop_list = ['\nHuman:', '\n```\n']

stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

# %%
def inference(model, tokenizer, prompt, max_length=20):
  total_response_length = len(tokenizer.tokenize(prompt)) + 8 + max_length
  pipe = pipeline(task="text-generation", model=model, max_length=total_response_length, tokenizer=tokenizer, stopping_criteria=stopping_criteria)
  result = pipe(prompt)

  return result[0]["generated_text"]

# %%
print(inference(model, tokenizer, generate_prompt_without_answer(filtered_df.iloc[300])))

# %% [markdown]
# ## Setup Training Pipeline

# %%
dataset = load_dataset("csv", data_files=data_path, split="train")

# %%
def format_dataset(data):
    output_texts = []
    
    for i in range(len(data["issue_number"])):
        formatted_text = generate_prompt_with_answer(data, i)
        output_texts.append(formatted_text)

    return output_texts

# %%
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)
    else:
      print(f"Using {compute_dtype}")

# %%
model.config.use_cache = False
model.config.pretraining_tp = 1

peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

training_params = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_params,
    packing=packing,
    formatting_func=format_dataset
)

trainer.train()
trainer.model.save_pretrained(new_model)


# %%
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


import random


output = ""

for i in range(10):
    output += f"Observation: {i + 1}\n"
    output += "=====================================================================================\n"
    prompt_template = generate_prompt_without_answer(filtered_df.iloc[random.randint(0, len(filtered_df) - 1)])
    output += inference(model, tokenizer, prompt_template, 30)
    output += "\n\n============================================================================================\n\n"

print(output)

with open("/home/mdafifal.mamun/notebooks/triagerX/output_13b.txt", "w") as out:
    out.write(output)
