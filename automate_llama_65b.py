import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import os

model_path = "upstage/llama-65b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True,
    rope_scaling={"type": "dynamic", "factor": 2}  # allows handling of longer inputs
)

# Get the current working directory
current_dir = os.path.dirname(os.path.realpath(__file__))

def generate_responses(prompt: str) -> str:
    prompt = f"### User:\n{prompt}\n\n### Assistant:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Check if the 'token_type_ids' key exists before deleting it
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    output = model.generate(**inputs, streamer=streamer, use_cache=True, max_new_tokens=float('inf'))
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Read data from truthful_qa.csv in the test_datasets directory
data = pd.read_csv(os.path.join(current_dir, "datasets/cut_datasets/subset_trivia_qa_web.csv"))

# Create a new column "generated_answer" by applying the generate_responses function
data["generated_answer"] = data["question"].apply(generate_responses)

# Save the result to a new CSV file in the test_models directory
data.to_csv(os.path.join(current_dir, "test.csv"), index=False)

print(f'Results have been saved to test.csv')
