from chat import Config, ChatInterface

chat = ChatInterface(
    context_file="./resources/homepage/QA.xlsx",
    embeddings_file="./resources/homepage/QA_embeddings.csv"
)

question = "I spent 2000 EUR in Germany with my Gold card, got 500 EUR cash from the ATM and paid the invoice at the post counter. How much fees have I accumulated?"

#answer = chat.get_answer(question)