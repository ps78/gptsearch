import numpy as np
import openai
import pandas as pd
import pickle
import tiktoken
import os
from pathlib import Path

from chat import Config, ChatInterface

openai.api_key = Path('apikey').read_text()

qa_file = "./resources/viseca_homepage/QA.xlsx"
embeddings_file = "./resources/viseca_homepage/QA_embeddings.csv"

chat = ChatInterface(qa_file, embeddings_file)

question = "was ist meine limite?"

answer = chat.get_answer(question, True)
print(f"\nAnswer:\n<<{answer}>>\n")

#for sim, entry in context[:5]:    
#    print(f"{sim:.4f} id {entry.id:3} : {entry.tokens:3} tokens : {entry.language:3} {entry.topic:20} {entry.question}")

#answer = .answer_query_with_context(question, True)
#print(answer)