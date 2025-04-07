import warnings
warnings.filterwarnings("ignore")


import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, DataCollatorForLanguageModeling, TrainingArguments
import random
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import os
import pickle
import pandas as pd
import json



def load_config(config_path):
    """Loads configuration parameters from a JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def query_lm(model, tokenizer_small, question, generator_params, device):
    messages = [{"role": "user", "content": question}]
    input_text=tokenizer_small.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer_small.encode(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(inputs, **generator_params)
    return tokenizer_small.decode(outputs[0])


def load_data(data_path):
    with open(data_path, 'rb') as f1:
        data = pickle.load(f1)

    questions = [i[0] for i in data]
    answers = [i[1] for i in data]

    return questions, answers




def generate_llm_answers(model_small, tokenizer_small, questions, generator_params, device):

    smollm_answers = []
    
    for q in questions:

        response = query_lm(model_small, tokenizer_small, q, generator_params, device)
        response = response.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].split("####")[-1]

        smollm_answers.append(response)

    return smollm_answers



def main():

    config = load_config("config.json")
    model_config = config["base_model"]
    
    
    checkpoint = model_config["checkpoint"]
    device = model_config["device"]
    generator_params = config["generator_params"]

    test_data_path = config["paths"]["test_data"]

    tokenizer_small = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer_small.add_eos_token=True
    tokenizer_small.pad_token_id=0
    tokenizer_small.padding_side="left"
    model_small = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

    generator_params["pad_token_id"] = tokenizer_small.eos_token_id,
    generator_params["eos_token_id"] = tokenizer_small.eos_token_id


    questions, answers = load_data(test_data_path)

    llm_answers = generate_llm_answers(model_small, tokenizer_small, questions, generator_params, device)


    results = pd.DataFrame({'question':questions, 'ground_truth_ans':answers, 'smollm_ans':llm_answers})

    results.to_excel(r'smollm_eval.xlsx')



if __name__ == "__main__":

    main()



