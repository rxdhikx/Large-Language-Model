from networkx import generate_gexf
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from langchain.chains import LLMChain
from langchain import prompts
from langchain_community.llms import HuggingFacePipeline

# Function to generate responses for MoMo model
def generate_responses(prompt: str) -> str:
    prompt = f"### User:\n{prompt}\n\n### Assistant:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Check if the 'token_type_ids' key exists before deleting it
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']

    # Adjust parameters for MoMo model
    output = model.generate(**inputs, max_length=512, num_beams=1, no_repeat_ngram_size=2)

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Load the dataset
df = pd.read_csv('hotpot_qa.csv')

# Set up model and tokenizer for MoMo
model_path = "moreh/MoMo-72B-lora-1.8.7-DPO"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Set up langchain components
prompt = prompts(input_variables=["instruction"], template="{instruction}")
hf_pipeline = HuggingFacePipeline(pipeline=generate_gexf)
llm_chain = LLMChain(llm=hf_pipeline, prompt=prompt)

# Create a new column "generated_answer" by applying the generate_responses function
df["generated_answer"] = df["question"].apply(generate_responses)

# Save the result to a new CSV file
df.to_csv('results_momo.csv', index=False)

print(f'Results have been saved to results_momo.csv')
