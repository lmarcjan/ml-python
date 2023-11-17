from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"

if __name__ == '__main__':

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    input = "What is AI?"
    encoded_input = tokenizer.encode(input, return_tensors='pt')
    output = model.generate(encoded_input, max_length=200, do_sample=True)
    print("Output:\n\n"+tokenizer.decode(output[0], skip_special_tokens=True))
