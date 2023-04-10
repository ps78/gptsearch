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

question = "I spent 2000 EUR in Germany with my Gold card, got 500 EUR cash from the ATM and paid the invoice at the post counter. How much fees have I accumulated?"

prompt = chat.get_prompt(question)

answer = chat.get_answer(question, True)
#print(f"\nAnswer:\n<<{answer}>>\n")
