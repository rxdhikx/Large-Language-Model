# pip install transformers==4.35.2
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

generate_text = pipeline(model="moreh/MoMo-72B-lora-1.8.7-DPO", torch_dtype=torch.bfloat16,
                         trust_remote_code=True, device_map="auto", return_full_text=True)

tokenizer = AutoTokenizer.from_pretrained("moreh/MoMo-72B-lora-1.8.7-DPO")
model = AutoModelForCausalLM.from_pretrained(
    "moreh/MoMo-72B-lora-1.8.7-DPO"
)
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

hf_pipeline = HuggingFacePipeline(pipeline=generate_text)

llm_chain = LLMChain(llm=hf_pipeline, prompt=prompt)
llm_context_chain = LLMChain(llm=hf_pipeline, prompt=prompt_with_context)
print(llm_chain.predict(instruction="What is the capital city of India?").lstrip())
