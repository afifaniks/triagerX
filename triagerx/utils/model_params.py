from transformers import AutoModel
import torch

# List of model names
model_names = {
    "BERT-Base": "bert-base-uncased",
    "BERT-Large": "bert-large-uncased",
    "RoBERTa-Base": "roberta-base",
    "RoBERTa-Large": "roberta-large",
    "DeBERTa-Base": "microsoft/deberta-base",
    "DeBERTa-Large": "microsoft/deberta-large",
    "CodeBERT": "microsoft/codebert-base"
}

# Function to get model details
def get_model_info(model_name):
    model = AutoModel.from_pretrained(model_name)
    param_count = sum(p.numel() for p in model.parameters())
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    return param_count, hidden_size, num_layers, num_heads

# Collect model details
table_data = []
for model_name, hf_name in model_names.items():
    try:
        param_count, hidden_size, num_layers, num_heads = get_model_info(hf_name)
        table_data.append((model_name, param_count, hidden_size, num_layers, num_heads))
    except Exception as e:
        print(f"Error loading {model_name}: {e}")

# Print table
header = f"{'Model':<15}{'#Params':<12}{'Hidden Size':<12}{'Layers':<8}{'Attention Heads'}"
print(header)
print("=" * len(header))

for row in table_data:
    print(f"{row[0]:<15}{row[1]/1e6:<12.1f}{row[2]:<12}{row[3]:<8}{row[4]}")

