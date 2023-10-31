from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"

if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    input = "What is AI?"
    encoded_input = tokenizer.encode(input, return_tensors='pt')
    output = model.generate(encoded_input, max_length=200, do_sample=True, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    print("Output:\n\n"+tokenizer.decode(output[0], skip_special_tokens=True))
