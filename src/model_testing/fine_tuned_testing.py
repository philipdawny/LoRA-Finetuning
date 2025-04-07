import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, DataCollatorForLanguageModeling, TrainingArguments
import random
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import math
import os
import pickle
import pandas as pd
import json
from lora.prompt_templates import question_template


def load_config(config_path):
    """Loads configuration parameters from a JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config



def load_data(data_path):
    with open(data_path, 'rb') as f1:
        data = pickle.load(f1)

    questions = [i[0] for i in data]
    answers = [i[1] for i in data]

    return questions, answers


def query_lm(model, tokenizer, question, generator_params, device):
    messages = [{"role": "user", "content": question}]
    input_text=tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(inputs, **generator_params)
    return tokenizer.decode(outputs[0])


def format_outputs(outputs):

    outputs = [output.split("<|im_start|>user")[-1].split("<|im_end|>")[0] for output in outputs]
    outputs = [output.strip("\nIn the context of aviation, the technical answer to this question is") for output in outputs]

    return outputs


def main():

    config = load_config("config.json")
    base_model_config = config["base_model"]
    tuned_model_config = config["tuned_model"]
    

    base_checkpoint = base_model_config["checkpoint"]
    tuned_checkpoint = tuned_model_config["checkpoint"]
    device = tuned_model_config["device"]
    generator_params = base_model_config["generator_params"]

    test_data_path = config["paths"]["test_data"]


    tokenizer_small = AutoTokenizer.from_pretrained(base_checkpoint)
    tokenizer_small.add_eos_token=True
    tokenizer_small.pad_token_id=0
    tokenizer_small.padding_side="left"
    
    
    model = AutoModelForCausalLM.from_pretrained(tuned_checkpoint).to(device)
    model.config.use_cache = True
    model.eval()

    questions, answers = load_data(test_data_path)


    test_questions = [question_template.format(question = question) for question in questions]

    outputs=[query_lm(model, tokenizer_small, q, generator_params) for q in test_questions]

    outputs = format_outputs(outputs)


    results = pd.DataFrame({'question':test_questions, 'ground_truth_ans':answers, 'smollm_ans':outputs})

    results.to_excel(r"smollm_finetuned_eval_0305_1143.xlsx")


    