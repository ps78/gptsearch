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
embeddings_file = "./resources/viseca_homepage/QA_embeddings2.csv"

chat = ChatInterface(qa_file, embeddings_file)

question = "was ist meine limite?"

sim = chat.get_relevant_context(question)
for item in sim[:5]:
    r = chat.context.iloc[item[1]]
    print(f"{item[0]:.4f} - {item[1]:3} - {r.tokens:3} tokens : {r.language:7} {r.topic:20} {r.question}")

#answer = prepro.answer_query_with_context(question, True)
#print(answer)