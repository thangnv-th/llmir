import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = 'hf_download/Meta-Llama-3.1-8B-Instruct'
tokenizer_name = name

model = AutoModelForCausalLM.from_pretrained(
    name,
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

print(tokenizer.pad_token_id)


tokenizer.add_special_tokens(
    {"pad_token": "<|finetune_right_pad_id|>"}
)
tokenizer.eos_token = "<|end_of_text|>"
print(tokenizer.pad_token_id)
model.config.bos_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

model.save_pretrained("hf_download/Meta-Llama-3.1-8B-Instruct-pad")
tokenizer.save_pretrained("hf_download/Meta-Llama-3.1-8B-Instruct-pad")