import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

INPUT_DATA = (
    "/home/mdafifal.mamun/notebooks/triagerX/notebook/data/openj9/openj9_processed.csv"
)
OUTPUT_DATA = "/home/mdafifal.mamun/notebooks/triagerX/notebook/data/openj9/component_training/df_all_summarized.csv"

print("Loading Llama3 pipeline...")
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

max_new_tokens = 512


def get_summary(text):
    messages = [
        {
            "role": "system",
            "content": "You are a bug report summarizer. "
            "Given a bug description that includes logs, stacktraces, your job is to summarize the bug in natural language for better understanding.",
        },
        {
            "role": "user",
            "content": "Sumamrize the provided bug report in a passage. During summarization avoid using special characters, brackets, etc."
            "Include every details in the bug presumably the root cause, associated files, any relavant information.:\n"
            + text,
        },
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        max_length=model.config.max_position_embeddings - max_new_tokens,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1] :]

    return tokenizer.decode(response, skip_special_tokens=True)


df = pd.read_csv(INPUT_DATA)

print("Generating summaries...")
summaries = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing issue"):
    text = f"Bug Title: {row.issue_title}\nBug Description: {row.issue_body}"
    sumamry = get_summary(text)
    sumamry = sumamry.replace("\n\n", "\n")
    summaries.append(sumamry)

df["summary"] = summaries
df.to_csv(OUTPUT_DATA)
print("Output saved.")
