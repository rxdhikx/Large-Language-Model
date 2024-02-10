
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import wandb
from datasets import load_dataset, load_metric
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer, get_scheduler, set_seed
from langchain import PromptTemplate, LLMChain
from langchain_community.llms import HuggingFacePipeline

from petals import DistributedLlamaForSequenceClassification

set_seed(0)

MODEL_NAME = "enoch/llama-65b-hf"

# Choose a prompt-tuning mode ('ptune' or 'deep_ptune').
# The latter fine-tunes separate prefixes for each transformer block,
# so prompt-tuning will take more time but yield better results.
# See this paper for details of how it works: https://arxiv.org/pdf/2110.07602.pdf
TUNING_MODE = 'ptune'

NUM_PREFIX_TOKENS = 8
DEVICE = 'cuda'
BATCH_SIZE = 32
LR = 1e-2
WEIGHT_DECAY = 0.0
NUM_EPOCHS = 3
SEED = 42
MODEL_MAX_LENGTH = 64

tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = 'right'
tokenizer.model_max_length = MODEL_MAX_LENGTH
tokenizer.pad_token = tokenizer.unk_token
model = DistributedLlamaForSequenceClassification.from_pretrained(
    MODEL_NAME,
    pre_seq_len=NUM_PREFIX_TOKENS,
    tuning_mode=TUNING_MODE
).float().to(DEVICE)
model.config.pad_token_id = tokenizer.pad_token_id

task = 'sst2'

dataset = load_dataset("glue", task)

def preprocess_function(examples):
    return tokenizer(examples["sentence"], padding='max_length', truncation=True, return_token_type_ids=False)

tokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx", "attention_mask"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

train_dataset = tokenized_datasets["train"].shuffle(seed=SEED)
valid_dataset = tokenized_datasets["validation"].shuffle(seed=SEED)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

metric = load_metric('glue', task)

def eval_metrics(model, dataloader, device='cpu'):
    model.eval()
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    model.train()
    return metric.compute()

for n, p in model.named_parameters():
    if p.requires_grad:
        print(n, p.requires_grad, p.device)

        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * NUM_EPOCHS
)

wandb.init(
    project="bloom-sst-2",
    config={
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "num_prefix_tokens": NUM_PREFIX_TOKENS,
        "model_name": MODEL_NAME,
        "seed": SEED,
    }
)

scaler = torch.cuda.amp.GradScaler()

for epoch in range(NUM_EPOCHS):
    model.train()
    for batch in tqdm(train_dataloader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        with torch.autocast(device_type=DEVICE, dtype=torch.float16):
          outputs = model(**batch)
        loss = outputs.loss
        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad()

        wandb.log({"Train Loss": loss.detach()})

    accuracy = eval_metrics(model, valid_dataloader, device=DEVICE)
    wandb.log({"Valid Accuracy": accuracy}, commit=False)



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
print(llm_chain.predict(instruction="Which continent is India located in?").lstrip())
