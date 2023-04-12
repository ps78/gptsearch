# GPT Search

## Introduction

This is a small sample implementation of GPT-powered search on a domain specific Q&A corpus.
Code is based on open-ai's tutorial:
https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb

## Setup

In order to setup a working environment with anaconda and vscode, follow these steps:

```
git clone https://github.com/ps78/gptsearch
cd gptsearch
conda create --prefix ./envs python=3.10 pandas
conda activate ./envs
conda install -c conda-forge openai tiktoken
code ./
```
add a file named 'apikey' containing your openai-API key to the root folder of the project

## Usage

Sample context has been provided in resources/homepage/QA.xlsx, based on the content of the website viseca.ch. This file can be replaced with any other context.
Currently the file contains mainly English context.

Upon first execution of the following sample, embeddings will be calculated and persisted (takes ~1 minute)

main.py contains a sample call to the interface:

```text
from chat import ChatInterface

chat = ChatInterface(
    context_file="./resources/homepage/QA.xlsx",
    embeddings_file="./resources/homepage/QA_embeddings.csv"
)

question = "I spent 2000 EUR in Germany with my Gold card, got 500 EUR cash from the ATM and paid the invoice at the post counter. How much fees have I accumulated?"

answer = chat.get_answer(question)
```

An extensive diagnostic log will be written to ./gptsearch.log