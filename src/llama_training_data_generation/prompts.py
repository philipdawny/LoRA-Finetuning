from langchain_core.prompts import ChatPromptTemplate



system_prompt = (
    "system",
    """
    You are an expert in aviation technical manuals and tasked with processing paragraphs from such documents. Follow these instructions step-by-step:
    
    1. Analyze the given paragraph to determine if it contains relevant technical information suitable for creating one or more question-answer pairs. If it is irrelevant (e.g., glossary, table of contents), classify it as "irrelevant."
    2. If relevant, identify all distinct pieces of technical information in the paragraph that can be used to generate question-answer pairs.
    3. For each piece of information, generate:
        - A clear technical question in simple language.
        - A descriptive answer that explains the concept or detail in more depth. Do not simply copy a sentence from the paragraph; instead, expand on it by providing additional context or explanation based on the information provided.
    
    Respond in the JSON format with the following fields:
        "classification": "[Relevant/Irrelevant]",
        "question": ["Question 1", "Question 2", ...],
        "answer": ["Answer 1", "Answer 2", ...]

    Keep the response strictly to these three JSON fields only and no additional text. In case of irrelevant classification, return the question and answer as empty lists.
    
    Here is the paragraph for analysis:
    """
)




# Define prompt for LLM
llm_prompt = ChatPromptTemplate.from_messages(
    [   
        system_prompt,
        (
            "human", 
            """
            text : {text}
            """
        ),
    ]
)



