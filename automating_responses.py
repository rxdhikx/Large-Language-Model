from networkx import generate_gexf
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import os

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

# Read data from hotpot_qa.parquet in the datasets directory
data = pd.read_parquet(os.path.join(current_dir, "../models/datasets/hotpot_qa.parquet"))

# Create a new column "generated_answer" by applying the generate_responses function
data["generated_answer"] = data["question"].apply(generate_responses)

# Save the result to a new CSV file in the models directory
data.to_parquet(os.path.join(current_dir, "results_momo.parquet"), index=False)

print(f'Results have been saved to results_momo.parquet')
