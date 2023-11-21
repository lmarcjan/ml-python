from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"

if __name__ == '__main__':

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    query = "What is AI?"
    inputs = tokenizer.encode(query, return_tensors='pt')
    output = model.generate(inputs, max_length=200, do_sample=True)
    print("Output:\n\n"+tokenizer.decode(output[0], skip_special_tokens=True))
