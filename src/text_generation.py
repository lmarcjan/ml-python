import torch
from transformers import LlamaTokenizer, AutoModelForCausalLM

model_name = "h2oai/h2ogpt-4096-llama2-7b-chat"

if __name__ == '__main__':

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, trust_remote_code=True)

    prompt = "My name is Julien and I like to"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model(input_ids=input_ids)
