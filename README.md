# LLM Hallucination Detection:

Objectives - 

• Collection of relevant datasets from various sources to work on more than 200,000 samples.
<br> • Construction of a robust discriminator to detect hallucinations in LLM's generated response, by reviewing the performance of different PLM's and training it, to minimize the cross-entropy loss function.
<br> • Utilization of transformers, pytorch and accelerate using 72b training models like Llama2 and MoMo - which is #3 on the open LLM leaderboard. (as of 02/20/2024)
<br> • Analysis on different types of hallucinations with LLM-assessment metrics, machine, human and composite metrics.

# ABOUT THE FILES

<h3> subset_of_dataset.py - </h3>
Used this for splitting the huge size of datasets (more than 300mb) to perform automation on a simpler size of dataset. 
Original number of rows = 8k+ 
Subset number of rows = 50
