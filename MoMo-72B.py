import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_text(prompt, model, tokenizer, max_length=100, num_beams=5, no_repeat_ngram_size=2):
    # Tokenize the input prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Generate text
    output = model.generate(input_ids, max_length=max_length, num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size)

    # Decode the generated output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text

def main():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("moreh/MoMo-72B-lora-1.8.7-DPO")
    model = AutoModelForCausalLM.from_pretrained("moreh/MoMo-72B-lora-1.8.7-DPO")

    # Set the prompt inside the code
    prompt = "Why do veins appear blue?"

    # Generate text based on the prompt
    generated_text = generate_text(prompt, model, tokenizer)

    # Display the generated text
    print("\nGenerated Text:")
    print(generated_text)

if __name__ == "__main__":
    main()
