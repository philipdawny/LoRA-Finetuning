question_template = '''You will be asked to answer the following technical question related to the aviation domain. Do not add any additional text after the answer.
    
    {question}
'''



answer_template = '''In the context of aviation, the technical answer to this question is

            {answer} 
'''


output_template = '''<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
    <|im_start|>user\nYou will be asked an aviation related technical question. Place the answer at the end of the response. Do not provide the question itself as the answer. Do not add any additional text after the answer.
    {q} <|im_end|>
    <|im_start|>assistant\n {a}<|im_end|>'''

