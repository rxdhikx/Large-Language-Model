from networkx import generate_gexf
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import os
import pandas as pd
df = pd.read_csv('hotpot_qa.csv')
df.to_csv('hotpot_qa.csv')
model_path = "moreh/MoMo-72B-lora-1.8.7-DPO"

tokenizer = AutoTokenizer.from_pretrained("moreh/MoMo-72B-lora-1.8.7-DPO")
model = AutoModelForCausalLM.from_pretrained(
    "moreh/MoMo-72B-lora-1.8.7-DPO"
)

# Get the current working directory
current_dir = os.path.dirname(os.path.realpath(__file__))

from langchain import PromptTemplate, LLMChain
from langchain_community.llms import HuggingFacePipeline

# template for an instrution with no input
prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}")

# template for an instruction with input
prompt_with_context = PromptTemplate(
    input_variables=["instruction", "context"],
    template="{instruction}\n\nInput:\n{context}")

hf_pipeline = HuggingFacePipeline(pipeline=generate_gexf)

llm_chain = LLMChain(llm=hf_pipeline, prompt=prompt)
llm_context_chain = LLMChain(llm=hf_pipeline, prompt=prompt_with_context)

# Read data from hotpot_qa.csv in the datasets directory
data = pd.read_csv(os.path.join(current_dir, "hotpot_qa.csv"))

# Create a new column "generated_answer" by applying the generate_responses function
data["generated_answer"] = data["question"].apply(generate_responses)

def generate_responses(prompt: str) -> str:
    prompt = f"### User:\n{prompt}\n\n### Assistant:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Check if the 'token_type_ids' key exists before deleting it
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    output = model.generate(**inputs, streamer=streamer, use_cache=True, max_new_tokens=float('inf'))
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Save the result to a new CSV file in the models directory
data.to_csv(os.path.join(current_dir, "results_momo.csv"), index=False)

print(f'Results have been saved to results_momo.csv')
