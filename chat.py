import os
import time
import pandas as pd
import numpy as np
import logging
import tiktoken
import openai
from dataclasses import dataclass, field

# setup loggers
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('gptsearch.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

@dataclass
class Config:
    """
    Contains all configuration settings used to 
    prepare data and query the model

    All properties have defaults
    """
    
    # model used to run text completions
    completion_model :str = "text-davinci-003"

    # model used for embeddings
    embedding_model :str = "text-embedding-ada-002"

    # encoding used for the completion model
    encoding :str = "gpt2" 

    # maximum number of tokens to use for the context in the prompt
    max_section_len :int = 500

    # separator for context sections 
    section_separator :str = "\n* "

    # temperature setting for the completion model
    temperature :float = 0.0
    
    # max token setting of the completion model
    max_answer_tokens :int = 300

    # keyword that marks Q&A entries where there is no question and the answer is just context information
    intro_string :str = "INTRO"

    # text that is prepended to the prompt, before the context
    # key = language | value = text to prepend
    prompt_header :dict[str, str] = field(default_factory=lambda: {
        "de": 
        "Beantworte folgende Frage so wahrheitsgetreu wie möglich unter Verwendung des folgenden Kontexts. Falls die Antwort nicht im Kontext beinhaltet ist, antworte mit \"Ich weiss es nicht.\"",

        "en": 
        "Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say \"I don't know.\"",

        "fr":
        "Répondez à la question le plus sincèrement possible en utilisant le contexte fourni et, si la réponse n'est pas contenue dans le texte ci-dessous, dites \"Je ne sais pas\"",
        
        "it":
        "Rispondete alla domanda nel modo più veritiero possibile utilizzando il contesto fornito e, se la risposta non è contenuta nel testo sottostante, dite \"Non lo so\""
    })

@dataclass
class ContextEntry:
    """
    Represents one entry of context information
    """    
    id :int             # unique id of the entry
    language :str       # one of {de, en, it, fr}
    topic :str          # general Q&A topic
    question :str       # question or config.intro_string to indicate answer is an introduction section
    answer :str         # answer to question or introduction section
    references :str     # optional additional references (empty string if missing)
    text :str           # the text used as context, built from the other attributes
    tokens :int         # the number of tokens of text

def vector_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, 
    the cosine similarity is the same as the dot product.
    """
    return np.dot(x, y)

class ChatInterface:
    """
    Implements pre-processing of context data and provides methods 
    to ask questions
    """

    # contains all configuration parameters of the class
    config :Config

    # all context elements. The key corresponds to ContextEntry.Id
    context :dict[int, ContextEntry]

    # the embeddings created from the context. The key corresponds to the index
    # in context
    embeddings :dict[int, np.ndarray]

    def __init__(self, context_file :str, embeddings_file :str|None = None, 
                    config :Config|None = None):
        """
        Constructor. Imports the given context file and embeddings, or creates the 
        embeddings if necessary

        Args:
            context_file (str):        
                File to load context from. Must heave headers:
                id, language, topic, question, answer, references.
                Supported formats: xlsx
            embeddings_file (str): optional file to load
                embeddings from. Must correspond to content in the context_file.
                If omitted, embeddings will be computed automatically
            config (Config): 
                Optional configuration instance. 
                If omitted, default one will be created
        """
        
        self.config = config if config is not None else Config()

        self._read_context(context_file)

        # load or create embeddings
        if embeddings_file is None or not os.path.isfile(embeddings_file):
            self._create_embeddings()
            if embeddings_file is not None:
                self._save_embeddings(embeddings_file)
        else:
            self._load_embeddings(embeddings_file)
        
    def _save_embeddings(self, embeddings_file :str):   
        """
        Saves the embeddings to the given file

        Args:
            embeddings_file (str): filename where to store embeddings
        """
        start = time.time()

        df = pd.DataFrame.from_dict(self.embeddings, orient='index')
        df.to_csv(embeddings_file, index_label='id')
        
        logger.info(f"Saved embeddings to {embeddings_file} in {time.time()-start:.3f}s")

    def _load_embeddings(self, embeddings_file :str):
        """
        Loads the embeddings from the given file

        Args:
            embeddings_file (str): csv file to load embeddings from
        """
        start = time.time()

        df = pd.read_csv(embeddings_file, header=0)
        max_dim = max([int(c) for c in df.columns if c != "id"])
        self.embeddings = { 
            int(r.id): np.array([ r[str(i)] for i in range(max_dim + 1)], dtype=np.float64) for _, r in df.iterrows() 
        }

        logger.info(f"Loaded embeddings from {embeddings_file} in {time.time()-start:.3f}s")

    def _read_context(self, context_file :str):
        """
        Imports the given Excel file and computes the embeddings
        Data txt-data is stored in self.data, self.embeddings, both as pandas dataframe

        Args:
            context_file (str): Excel file with these columns:
                id: unique integer id of the entry
                language: one of {english, german, italian, french}
                topic: high-level topic addressed by the Q&A, not unqiue
                question: question or keyword 'INTRO'
                answer: anwer to the previous question
                references: optional additonal references (links). Formatted as : "title::url**title::url"
        """
        start = time.time()

        self.context = dict()
        for _, r in pd.read_excel(context_file).fillna('').iterrows():
            text, tokens = self._make_text(r.language, r.topic, r.question, r.answer, r.references)
            entry = ContextEntry(r.id, r.language, r.topic, r.question, r.answer, r.references, text, tokens)
            self.context[r.id] = entry
        
        logger.info(f"Read context from {context_file} in {time.time()-start:.3f}s")

    def _create_embeddings(self):
        """
        Calculates the embeddings from the context
        """
        start = time.time()

        if self.context is not None:
            self.embeddings = { 
                int(r.id): self.get_embedding(r.text) for r in self.context.values() 
            }
        
        logger.info(f"Created embeddings for {len(self.embeddings)} texts in {time.time()-start:.3f}s")
    
    def _make_text(self, language :str, topic :str, question :str, answer :str, references :str) -> tuple[str, int]:
        """
        Creates a single text block given the inputs to be used as context element
        for the model.

        Args:
            language (str)
                one of the supported languages (de, en, it, fr)
            topic (str)
                general topic of the Q&A
            question (str)
                actual question or config.intro_string, for introduction entries
            answer (str)
                actual answer to question or introduction content
            references (str)
                optional references, can be ""

        Returns:
            tuple[str,int]
                The formatted text-block to use for model and the number of tokens it contains
        """
        if str.upper(question) == self.config.intro_string:
            txt = f"{answer}\n"
        else:
            txt = f"Q: {question}\nA: {answer}\n"

        #TODO: add references

        encoding = tiktoken.get_encoding(self.config.encoding)
        tokens = len(encoding.encode(txt))

        return (txt, tokens)

    def _get_main_language(self, entries :list[ContextEntry]) -> str:
        lang_count :dict[str, int] = dict()
        for entry in entries:
            if entry.language in lang_count:
                lang_count[entry.language] += 1
            else:
                lang_count[entry.language] = 1

        return max(lang_count, key=lambda x: lang_count[x])

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Computes an embedding for the given text

        Args:
            text (str)
                input to compute embedding

        Returns:
            np.ndarray
                embedding vector
        """
        result = openai.Embedding.create(
            model=self.config.embedding_model,
            input=text
        )
        return np.array(result["data"][0]["embedding"], dtype=np.float64) # type: ignore

    def get_relevant_context(self, query: str) -> list[tuple[float, ContextEntry]]:
        """        
        Find the entries from context which are most similar to the given query

        Args:
            query (str)
                The query to compare against the context

        Returns:
            list[Tuple[float, ContextEntry]]
                The similarity and context entries, ordered by similarity (most similar first)
        """
        start = time.time()

        query_embedding = self.get_embedding(query)        
        entries = [(vector_similarity(query_embedding, emb), self.context[id]) for id, emb in self.embeddings.items()]
        similarities = sorted(entries, key=lambda x: x[0], reverse=True)
        
        logger.info(f"Computed similarities between query and {len(self.embeddings)} context items in {time.time()-start:.3f}s")

        return similarities

    def get_prompt(self, question: str) -> str:
        """
        Creates the gpt-prompt given a question

        Args:
            question (str)
                question to answer
        
        Returns:
            str
                The full prompt to send to the language model
        """
        start = time.time()
        
        relevant_context = self.get_relevant_context(question)

        encoding = tiktoken.get_encoding(self.config.encoding)
        separator_len = len(encoding.encode(self.config.section_separator))

        chosen_context_str :list[str] = []
        chosen_context_len = 0
        chosen_context :list[ContextEntry] = []
        # add context until max number of tokens is reached
        for _, entry in relevant_context:            
            chosen_context_len += entry.tokens + separator_len
            if chosen_context_len > self.config.max_section_len:
                break                
            chosen_context_str.append(self.config.section_separator + entry.text.replace("\n", " "))
            chosen_context.append(entry)

        # detect language from the chosen entries
        lang = self._get_main_language(chosen_context)
        
        prompt = self.config.prompt_header[lang] \
            + "\n\nContext:\n" \
            + "".join(chosen_context_str) \
            + "\n\n Q: " + question + "\n A:"    

        logger.info(f"Generated prompt (language: {lang}) using context sections {', '.join([str(c.id) for c in chosen_context])} in {time.time()-start:.3f}s")
        
        return prompt

    def get_answer(self, query: str, show_prompt: bool = False) -> str:
        """
        Ask the language model to answer the given query

        Args:
            query (str):
                question to answer
            show_prompt (bool, optional):
                if true, prints the full prompt

        Returns:
            str:
                Reply from the language model
        """        
        prompt = self.get_prompt(query)        
        if show_prompt:
            logger.debug(f"Prompt:\n<<{prompt}>>\n")

        start = time.time()

        response = openai.Completion.create(
            prompt=prompt, 
            temperature=self.config.temperature, 
            max_tokens=self.config.max_answer_tokens,
            model=self.config.completion_model
        )                
        answer = response["choices"][0]["text"].strip(" \n") # type: ignore

        logger.debug(f"answer: {answer}")
        logger.info(f"Querying language model took {time.time()-start:.3f}s")

        return answer 