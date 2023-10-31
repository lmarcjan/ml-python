import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"

if __name__ == '__main__':

    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    text = "What is love?"
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    print("Output:\n\n"+tokenizer.decode(output.logits[0], skip_special_tokens=True))
