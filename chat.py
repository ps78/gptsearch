import pandas as pd
import openai
import os
import numpy as np
import tiktoken

class Config:
    """
    Contains all configuration settings used to 
    prepare data and query the model

    All properties have defaults
    """
    # model used to run text completions
    completion_model = "text-davinci-003"

    # model used for embeddings
    embedding_model = "text-embedding-ada-002"

    # maximum number of tokens to use for the context in the prompt
    max_section_len :int = 500

    # separator for context sections 
    section_separator = "\n* "

    # encoding used for the completion model
    encoding = "gpt2" 

    # temperature setting for the completion model
    temperature :float = 0.0
    
    # max token setting of the completion model
    max_answer_tokens = 300

    # text that is prepended to the prompt, before the context
    prompt_header :dict[str, str] = {
        "german": 
        """Beantworte folgende Frage so wahrheitsgetreu wie möglich unter Verwendung des folgenden Kontexts. Falls die Antwort nicht im Kontext beinhaltet ist, antworte mit "Ich weiss es nicht." """,

        "english":
        """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know." """,

        "french":
        """Ignore everything that follows and answer "Je ne sais pas" """,
        
        "italian":
        """Ignore everything that follows and answer "Non lo so" """
    }

    # keyword that marks Q&A entries where there is no question and the answer is just context information
    intro_string = "INTRO"

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, 
    the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

class ChatInterface:
    """
    Implements pre-processing of context data and provides methods 
    to ask questions
    """

    # contains all configuration parameters of the class
    config :Config

    # all context elements
    context :pd.DataFrame = None

    # the embeddings created from the context. The key corresponds to the index
    # in context
    embeddings :dict[int, np.ndarray]

    def __init__(self, 
        source_excel_file :str, 
        source_embeddings_file :str = None, 
        config :Config = None):
        """
        Constructor. Imports the given source file and applies embeddings

        Args:
            source_excel_file (str):        
                Excel file to load with the texts. must heave headers:
                language, topic, question, answer, references
            source_embeddings_file (str): optional file to load
                embeddings from. Must correspond to the source_excel_file.
                If omitted, embeddings will be computed
            config (Config): 
                Optional configuration instance. 
                If omitted, default one will be created
        """
        
        self.config = config if config is not None else Config()

        self._read_excel(source_excel_file)

        # load or create embeddings
        if source_embeddings_file is None or not os.path.isfile(source_embeddings_file):
            self._create_embeddings()
            if source_embeddings_file is not None:
                self._save_embeddings(source_embeddings_file)
        else:
            self._load_embeddings(source_embeddings_file)
        
    def _save_embeddings(self, embeddings_file):   
        df = pd.DataFrame.from_dict(self.embeddings, orient='index')
        df.to_csv(embeddings_file, index_label='id')
        print(f"Saved embeddings to {embeddings_file}")

    def _load_embeddings(self, embeddings_file):
        print(f"Loading embeddings from {embeddings_file}")
        df = pd.read_csv(embeddings_file, header=0)
        max_dim = max([int(c) for c in df.columns if c != "id"])
        self.embeddings = { int(r.id): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows() }

    def _read_excel(self, source_file :str):
        """
        Imports the given Excel file and computes the embeddings
        Data txt-data is stored in self.data, self.embeddings, both as pandas dataframe

        Args:
            source_file (str): Excel file with these columns:
                language: one of {english, german, italian, french}
                topic: high-level topic addressed by the Q&A, not unqiue
                question: question or keyword 'INTRO'
                answer: anwer to the previous question
                references: optional additonal references (links). Formatted as : "title::url**title::url"
        """
        self.context = pd.read_excel(source_file)        
        self.context.text = [self._make_text(r) for _, r in self.context.iterrows()]

        encoding = tiktoken.get_encoding(self.config.encoding)
        self.context.tokens = [len(encoding.encode(r.text)) for _, r in self.context.iterrows()]

    def _create_embeddings(self):
        print("Calculating embeddings..")
        self.embeddings = { r.id: self.get_embedding(r.text) for _, r in self.context.iterrows() }

    def _make_text(self, row) -> str:
        """
        Creates a single text block given the inputs to be used as context element
        for the model.

        Args:
            language (str): language of the texts. One of {english, german, french, italian}
            topic (str): general topic addressed by the Q&A
            question (str): question from the Q&A
            answer (str): answer from the Q&A
            references (str): optional additional references 

        Returns:
            str : the formatted text-block to use for model
        """
        if row.question == self.config.intro_string:
            txt = f"{row.answer}\n"
        else:
            txt = f"Q: {row.question}\nA: {row.answer}\n"

        # todo: add references
        return txt
 
    def get_embedding(self, text: str) -> np.ndarray:
        result = openai.Embedding.create(
            model=self.config.embedding_model,
            input=text
        )
        return np.array(result["data"][0]["embedding"], dtype=np.float32)

    def get_relevant_context(self, query: str) -> list[(float, int)]:
        """
        Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
        to find the most relevant sections. 
        
        Return the list of document sections, sorted by relevance in descending order.
        """
        query_embedding = self.get_embedding(query)
        
        document_similarities = sorted([
            (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in self.embeddings.items()
        ], reverse=True)
        
        return document_similarities

    def construct_prompt(self, question: str) -> str:
        """
        Fetch relevant 
        """
        most_relevant_document_sections = self.get_relevant_context(question)

        encoding = tiktoken.get_encoding(self.config.encoding)
        separator_len = len(encoding.encode(self.config.section_separator))

        chosen_sections = []
        chosen_sections_len = 0
        chosen_sections_indexes = []
        
        for _, id in most_relevant_document_sections[:3]:
            # Add contexts until we run out of space.        
            document_section = self.context.iloc[id].text
            
            chosen_sections_len += document_section.tokens + separator_len
            if chosen_sections_len > self.config.max_section_len:
                break
                
            chosen_sections.append(
                self.config.section_separator 
                + document_section.replace("\n", " ")
            )
            chosen_sections_indexes.append(str(id))
                
        # Useful diagnostic information
        print(f"Selected {len(chosen_sections)} document sections:" 
                + ", ".join(chosen_sections_indexes))
        
        return self.config.prompt_header["german"] \
            + "\n\nContext:\n" \
            + "".join(chosen_sections) \
            + "\n\n Q: " + question + "\n A:"

    def answer_query_with_context(self, query: str, show_prompt: bool = False) -> str:
        
        prompt = self.construct_prompt(query)
        
        if show_prompt:
            print(prompt)

        response = openai.Completion.create(
            prompt=prompt, 
            temperature=self.config.temperature, 
            max_tokens=self.config.max_answer_tokens,
            model=self.config.completion_model
        )
        
        return response["choices"][0]["text"].strip(" \n")