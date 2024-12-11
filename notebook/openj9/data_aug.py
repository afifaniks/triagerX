import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

data_path = "/home/mdafifal.mamun/notebooks/triagerX/comp_train.csv"
df = pd.read_csv(data_path)

df = df[df["component"].notna()]
print("Base df size:", len(df))
print(df.component.value_counts())

class Llama3:
    def __init__(self, model_id, max_new_tokens=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.max_new_tokens = max_new_tokens

    def invoke(
        self, system_prompt, question, temperature=0.5, return_token_usage=False
    ):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            max_length=self.model.config.max_position_embeddings - self.max_new_tokens,
            truncation=True,
            return_tensors="pt",
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
        )
        response = outputs[0][input_ids.shape[-1] :]

        if return_token_usage:
            prompt_tokens = len(input_ids[0])
            completion_tokens = len(outputs[0]) - prompt_tokens

            token_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

            return (
                self.tokenizer.decode(response, skip_special_tokens=True),
                token_usage,
            )

        return self.tokenizer.decode(response, skip_special_tokens=True)


MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
llama3_model = Llama3(MODEL_ID)


SYSTEM_PROMPT = """You are an efficient and correct data augmentor. 
Your job is when provided a bug report text, you modify only the natural language part. 
You can use synonyms or change the structure of the sentence by swapping words. 
But you never change code artifacts like source codes, file paths, stack traces, logs, crash reports, version, etc.
Only return the augmented output. Nothing else.

Now augment the following bug report:
"""

augmented_rows = []

for component, num_components in df["component"].value_counts().items():
    if num_components < 350:
        print(f"Augmenting data for component: {component}")
        augmentation_needed = 350 - num_components

        # Filter rows for the current component
        original_data = df[df["component"] == component]

        # Augment data by sampling with replacement
        for _ in tqdm(range(augmentation_needed), total=augmentation_needed):
            # Sample one row randomly from the original_data
            row = original_data.sample(n=1, replace=True).iloc[0]
            print("Original data\n", row["text"])
            print("#################################")
            bug_report = row["text"]
            augmented_text = llama3_model.invoke(system_prompt=SYSTEM_PROMPT, question=f"Bug Report: {bug_report}")
            augmented_row = row.copy()
            augmented_text = augmented_text.replace("Bug Report:", "").strip()
            augmented_row["text"] = augmented_text
            print("Augmented data\n", augmented_text)
            print("#################################")

            augmented_rows.append(augmented_row)

        print(f"Saving at checkpoint: {component} | New entried added: {augmentation_needed}")
        augmented_df = pd.DataFrame(augmented_rows)
        result_df = pd.concat([df, augmented_df], ignore_index=True)

        print("Augmented df length:", len(result_df))
        print(result_df.component.value_counts())

        result_df.to_csv("comp_train_augmented_350.csv")
