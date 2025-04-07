from langchain_ollama import ChatOllama
from langchain.output_parsers import PydanticOutputParser
from typing import Optional
from pydantic import BaseModel, Field
import pandas as pd 
import os 
import argparse
from tqdm import tqdm 
import pickle
import random
import json

from prompts import llm_prompt


# output model for the LLM Response
class QA(BaseModel):
    """Generated question and answer pairs"""
    classification: str
    question: list
    answer: list



def load_config(config_path):
    """Loads configuration parameters from a JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config





def load_chunked_text(path):

    with open(path, 'rb') as f1:
        data = pickle.load(f1)
        random.shuffle(data)  # Shuffling the order of the chunked texts


    return data


if __name__ == "__main__":


    config = load_config("llamaconfig.json")
    
    # Creating the LLM Chain
    
    model = config["model_params"]["model"]
    base_url = config["model_params"]["base_url"]
    data_path = config["paths"]["data_path"]
    output_path = config["paths"]["output_path"]

    llm = ChatOllama(model=model, base_url = base_url)
    output_parser = PydanticOutputParser(pydantic_object=QA)
    question_chain = llm_prompt | llm | output_parser
    print("\n\n>>> LLM Chain Created")

    
    data = load_chunked_text(data_path)

    classification = []
    question = []
    answer = []


    total = len(data)
    done = 0 
    failed = 0



    with tqdm(total=len(data), desc="Processing rows") as pbar:
        
        for doc in data:

            if len(question) > 1500:
                print(r"\n\n >>> 1500 training examples generated. BREAKING LOOP.")
                break
            
            try:
                input_dict =  {'text' : doc.page_content}
                response = question_chain.invoke(input_dict)
                classification.append(response.classification)
                question.extend(response.question)
                answer.extend(response.answer)
                done += 1

            except:
                classification.append(None)
                question.append(None)
                answer.append(None)
                failed += 1

            pbar.update(1)
            pbar.set_postfix(total=len(question), done=done, failed=failed)



    final_data = {'classification':classification, 'question':question, 'answer':answer}


    # Saving training data to pickle file

    with open(output_path, 'wb') as f1:
        pickle.dump(final_data, f1)

    print(r"\n\n >>>>SAVED pkl FILE.")

