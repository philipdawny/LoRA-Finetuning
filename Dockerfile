FROM python:3.10

SHELL ["/bin/bash", "-c"]
RUN apt-get update -qq && apt-get upgrade -qq &&\
    apt-get install -qq man wget sudo vim tmux 

RUN yes | pip install --upgrade pip

COPY requirements.txt /home/
WORKDIR /home
RUN yes | pip install -r requirements.txt

COPY src/llama_training_data_generation/llamaconfig.json /home/
COPY src/llama_training_data_generation/llama_qa_generation.py /home/
COPY src/llama_training_data_generation/prompts.py /home/
COPY src/llama_training_data_generation/train_test_separation.py /home/

COPY src/lora/loraconfig.json /home/
COPY src/lora/lora_finetuning.py /home/
COPY src/lora/lora_unit_test.py /home/
COPY src/lora/prompt_templates.py /home/


COPY src/model_testing/config.json /home/
COPY src/model_testing/base_model_testing.py /home/
COPY src/model_testing/fine_tuned_testing.py /home/