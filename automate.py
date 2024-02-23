import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

class CustomTextStreamer:
    def __init__(self, tokenizer, skip_prompt=True, skip_special_tokens=True):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens

    def __call__(self, input_ids, max_length, eos_token_id):
        return self.generate_text(input_ids, max_length, eos_token_id)

    def generate_text(self, input_ids, max_length, eos_token_id):
        output = self.tokenizer.decode(
            input_ids[0],
            skip_special_tokens=self.skip_special_tokens,
            clean_up_tokenization_spaces=True
        )
        return output

# Function to generate responses for MoMo model
def generate_responses(prompt: str) -> str:
    prompt = f"### User:\n{prompt}\n\n### Assistant:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Check if the 'token_type_ids' key exists before deleting it
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']

    streamer = CustomTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Set max_new_tokens to control the maximum length of the generated response
    output = model.generate(**inputs, streamer=streamer, max_new_tokens=500)  # Adjust the value as needed

    return output

# Load the dataset
df = pd.read_csv('hotpot_qa.csv')

# Set up model and tokenizer for MoMo
model_path = "moreh/MoMo-72B-lora-1.8.7-DPO"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Create a new column "generated_answer" by applying the generate_responses function
df["generated_answer"] = df["question"].apply(generate_responses)

# Save the result to a new CSV file
df.to_csv('results_momo.csv', index=False)

print(f'Results have been saved to results_momo.csv')
