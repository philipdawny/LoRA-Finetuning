import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, DataCollatorForLanguageModeling, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import pickle
import subprocess

from prompt_templates import question_template, answer_template, output_template

"""
    Python script to run unit testing of LoRA fine-tuning.
    The number of training examples supplied is restricted to 100 to speed up training.
    Number of steps is fixed at 10, with 2 warmup steps
    
"""



class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)
    


def load_config(config_path):
    """Loads configuration parameters from a JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config




def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




def print_trainable_parameters(model):
    """Prints the number and percentage of trainable parameters in the model."""
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"\n\n>>> Trainable params: {trainable_params} || All params: {all_params} || Trainable%: {100 * trainable_params / all_params}\n")



def prepare_and_tokenize(example, tokenizer, max_length):
    
    q = example["question"]
    a = example["answer"]
    template = output_template

    return tokenizer(
        template,
        truncation=True,
        padding='max_length',
        max_length=max_length
    )


def create_dataset(dataset_path):
    with open(dataset_path, r'rb') as f:
        data = pickle.load(f)

    data = random.sample(data, 10)  # Randomly sampling 10 training examples for unit testing
    
    questions = [i[0] for i in data]
    answers = [i[1] for i in data]

    questions = [question_template.format(question = question) for question in questions]
    answers = [answer_template.format(answer = answer) for answer in answers]

    qa_dict = {"question": questions, "answer": answers}
    train_dataset = Dataset.from_dict(qa_dict)

    return train_dataset







def main():

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    # subprocess.run("export CUDA_VISIBLE_DEVICES=2", shell=True, capture_output=True, text=True) 

    print(r">>> Training Using GPU") if torch.cuda.is_available() else print(r">>> Training using CPU")


    config = load_config("loraconfig.json")

    #
    set_seed(config.get("seed", 42))

    
    checkpoint = config["checkpoint"]
    device = config["device"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

    
    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    
    model.gradient_checkpointing_enable()

    
    model.lm_head = CastOutputToFloat(model.lm_head)

    # Seting up the LoRA configuration from the config file
    lora_params = config["lora"]

    lora_config = LoraConfig(
        r=lora_params["r"],
        lora_alpha=lora_params["lora_alpha"],
        target_modules=lora_params["target_modules"],
        lora_dropout=lora_params["lora_dropout"],
        bias=lora_params["bias"],
        task_type=lora_params["task_type"]
    )
    lora_config.inference_mode = lora_params.get("inference_mode", False)

    
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    
    # Loading the training data
    train_dataset = create_dataset(config["train_data"])

    
    max_length = config.get("max_length", 2048)
    tokenized_dataset = train_dataset.map(
        lambda example: prepare_and_tokenize(example, tokenizer, max_length),
        batched=False,
        remove_columns=["question", "answer"]
    )

    
    # Manually setting max steps as 10 and warmup as 2 for unit testing

    training_args_params = config["training"]
    training_args = TrainingArguments(
        per_device_train_batch_size=training_args_params["per_device_train_batch_size"],
        gradient_accumulation_steps=training_args_params["gradient_accumulation_steps"],
        warmup_steps=2,
        max_steps=10,
        learning_rate=training_args_params["learning_rate"],
        fp16=training_args_params["fp16"],
        logging_steps=training_args_params["logging_steps"],
        output_dir=training_args_params["output_dir"]
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=data_collator
    )

    trainer.args._n_gpu = 1  
    model.config.use_cache = False

    
    trainer.train()

    # Save the fine-tuned model
    # model.save_pretrained(cofig["model_save_path"])

if __name__ == "__main__":
    main()


