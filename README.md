# LLM Hallucination Detection:

Objectives - 

• Collection of relevant datasets from various sources to work on more than 200,000 samples.
<br> • Construction of a robust discriminator to detect hallucinations in LLM's generated response, by reviewing the performance of different PLM's and training it, to minimize the cross-entropy loss function.
<br> • Utilization of transformers, pytorch and accelerate using 72b training models like Llama2 and MoMo - which is #3 on the open LLM leaderboard. (as of 02/20/2024)
<br> • Analysis on different types of hallucinations with LLM-assessment metrics, machine, human and composite metrics.

# ABOUT THE FILES

<h3> MoMo.py - </h3>
Functional LLM code on 72 billion parametric pre-trained MoMo model to run by manual prompting for response generation.

<h3> automate.py - </h3>
This code automates the prompts from dataset files and generates the answers through MoMo 72b model, and stores the answers into a new file in the specific directory.

<h3> convert_parquet_to_csv.py </h3>
Python code to convert the downloaded large size datasets which are in parquet format to functionable csv files to perform our further operations.

<h3> dolly-12B.py - </h3>
Functional LLM code on 12 billion parametric pre-trained dolly model to run by manual prompting for response generation.

<h3>automate_llama_65b.py</h3>
Functional automation code to generate responses with llama 65b model.

<h3>llama2.py</h3>
Functional LLM code on 30 billion parametric pre-trained llama2 model to run by manual prompting for response generation.

<h3> subset_of_dataset.py - </h3>
Used this for splitting the huge size of datasets (more than 300mb) to perform automation on a simpler size of dataset. <br>
Original number of rows = 8k+ <br>
Subset number of rows = 50

------------------------------------------------------------------------------------
<h4> MoMo - https://huggingface.co/moreh/MoMo-72B-lora-1.8.7-DPO </h4>
------------------------------------------------------------------------------------

