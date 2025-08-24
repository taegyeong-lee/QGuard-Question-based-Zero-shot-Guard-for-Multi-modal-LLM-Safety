import torch
from transformers import AutoModel, AutoTokenizer

def load_internvl_text_or_mm(model_path: str):
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=True,
        device_map="auto" if torch.cuda.is_available() else None,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer
