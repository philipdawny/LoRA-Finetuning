# LoRA Fine-Tuning



In this project, I fine-tune the small language model using knowledge distillation from a large language model – Llama 3.1-7B, with the aim of transferring some aviation specific domain knowledge to the small model. 

The training data for fine-tuning is a set of 1000 QA pairs extracted from aviation related technical documentation. 


Small language model: [SmolLM-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct)


### Fine-tuning Approach:

 * Generate training data using knowledge distillation from Llama 3.1- 7B
 * Training data consists of QA pairs extracted using LLM from aviation related technical documents
 * Splitting training data into training and validation sets - [Drive](https://drive.google.com/drive/folders/1UNjFHr7W71Tdxn2g5ur8XJ3ddRFTPmyN?usp=sharing)
 * Asessing performance of SmolLM-135M-Instruct on the validation data.
 * LoRA fine-tuning of SmolLM-135M-Instruct.
 * Assessing model performance on validation set, post fine-tuning.


## Steps to run unit testing code:

1. Run requirements.txt:

       !pip install requirements.txt


2. Update config variables in ```lora\loraconfig.json```:

    hf_data
    *  ```train_data``` - Path to QA pairs training data
    * ```checkpoint``` - HuggingFace model name of small langauge model to fine-tune
    * ```device``` - Specify GPU ID

    
    * ```lora``` - LoRA parameters
    * ```training``` - Training hyperparameters



3. Run ```lora_unit_test.py```

