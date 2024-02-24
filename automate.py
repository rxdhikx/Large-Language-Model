import os
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.chains import LLMChain
from langchain import prompts
from langchain_community.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain_community.llms import HuggingFacePipeline

generate_text = pipeline(model="moreh/MoMo-72B-lora-1.8.7-DPO", torch_dtype=torch.bfloat16,
                         trust_remote_code=True, device_map="auto", return_full_text=True)

model_path = "moreh/MoMo-72B-lora-1.8.7-DPO" 

tokenizer = AutoTokenizer.from_pretrained("moreh/MoMo-72B-lora-1.8.7-DPO")
model = AutoModelForCausalLM.from_pretrained(
    "moreh/MoMo-72B-lora-1.8.7-DPO"
)

# Get the current working directory
current_dir = os.path.dirname(os.path.realpath(__file__))


def generate_responses(prompt: str) -> str:
    prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}")

# template for an instruction with no input
prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}")

hf_pipeline = HuggingFacePipeline(pipeline=generate_text)

llm_chain = LLMChain(llm=hf_pipeline, prompt=prompt)
# llm_context_chain = LLMChain(llm=hf_pipeline, prompt=prompt_with_context)

# Read data from truthful_qa.csv in the models directory
data = pd.read_csv(os.path.join(current_dir, "truthful_qa_copy.csv"))

# Create a new column "generated_answer" by applying the generate_responses function
data["generated_answer"] = data["question"].apply(generate_responses)

# Save the result to a new CSV file in the test_models directory
data.to_csv(os.path.join(current_dir, "results_truthful_qa_momo.csv"), index=False)



print(f'Results have been saved to results_truthful_qa_momo.csv')
