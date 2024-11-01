import os
from gc import collect
from typing import Dict, Any

import vertexai
import yaml
from langchain.vectorstores import Chroma
from langchain_google_vertexai import VertexAIEmbeddings, VertexAI

# Initialize Vertex AI
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\RajanRayappan\CREDENTIALS\rtc-ai-20240203-c915144c2c55.json"
PROJECT_ID = "rtc-ai-20240203"
LOCATION = "us-central1"
vertexai.init(project=PROJECT_ID, location=LOCATION)


embedder = VertexAIEmbeddings(model_name="textembedding-gecko@003")



PERSISTENT_DISK_DIR = "chromadb"
collection_name = "coderag"

# Create langchain chroma instance for our rag db
chroma = Chroma(
    persist_directory = PERSISTENT_DISK_DIR,
    collection_name = collection_name,
    embedding_function = embedder
)

from pydantic import BaseModel
from abc import ABC

class AssistantResponse(ABC, BaseModel):

    class Config:
        arbitrary_types_allowed = True

    input_query: str
    prompt: str
    model_name: str
    model_parameters: Dict[str, Any]
    intent: str
    response: str

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class PairProgrammingAssistant:
    def __init__(self, chroma: Chroma, prompt_tag: str):
        self.config = self.read_config("./prompts.yml")
        self.chroma = chroma
        self.prompt_tag = prompt_tag
        self.model_name = None
        self.intent = None
        self.temperature = 0.2
        self.max_output_tokens = 1024

    @staticmethod
    def read_config(config_file: str):
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config

    def get_llm_model(self):
        self.model_name = self.config[self.intent]["model"]["name"]
        llm = VertexAI(model_name = self.model_name)
        return llm

    def detect_language(self, text: str):
        """
        Function to identity the dialect of code input by user.

        :param text: Raw code input by user.
        :return: Detected programming language name
        """
        language_detect_prompt = None
        if self.intent == "code_completion":
            language_detect_prompt = self.config["detect_language"]["contextual"]
        elif self.intent == "code_generation":
            language_detect_prompt = self.config["detect_language"]["natural_language"]

        llm = self.get_llm_model()
        language = llm.predict(language_detect_prompt.format(text=text))
        return language.strip().lower()

    def match_chroma(self, snippet: str, language: str):
        """
        Function to match similar code snippets from Chroma DB.

        :param snippet: Raw code input by user.
        :param language: Detected programming language name.
        :return: Similar code snippets from Chroma DB.
        """
        language_filter = (
            {"languages": language.lower()} if language.lower() != "none" else {}
        )
        code_docs = self.chroma.similarity_search_with_relevance_scores(
            query = snippet, k = 5, filter=language_filter
        )

        code_blobs = {}
        for code_doc in code_docs:
            blob_name = code_doc[0].metadata["filename"]
            if not code_blobs.get(blob_name):
                code_blobs[blob_name] = (code_doc[0].page_content, code_doc[1])
            break

        response = []
        for filename, code_doc in code_blobs.items():
            response.append(
                {
                    "snippet": code_doc[0],
                    "source": filename,
                    "score": code_doc[1],
                }
            )

        return response

    def get_prompt(self):
        prompt = self.config[self.intent]["prompts"][self.prompt_tag]
        return prompt

    def get_model_parameters(self):
        self.temperature = self.config[self.intent]["model"]["temperature"]
        self.max_output_tokens = self.config[self.intent]["model"]["max_output_tokens"]
        parameters = dict(
            temperature = self.temperature,
            max_output_tokens = self.max_output_tokens
        )
        return parameters

    def perform_coding_task(self, text, **prompt_args):
        llm = self.get_llm_model()
        prompt = self.get_prompt()
        parameters = self.get_model_parameters()

        prefix = prompt.format(text=text, **prompt_args)
        print(f"Formatted prompt: {prefix}")

        result = llm.predict(prefix, **parameters)
        response = result

        result_object = AssistantResponse(
            input_query = text,
            prompt = prefix,
            model_name = self.model_name,
            model_parameters = parameters,
            intent = self.intent,
            response = response
        )
        return result_object




pp_assistant = PairProgrammingAssistant(chroma, prompt_tag="zero_shot")


def code_completion_generation(code_input, code_task):
    """

    :param code_input: User provided code / prompt.
    :param code_task: Type of task to perform (Completion or Generation)
    :return:
    """
    global pp_assistant

    intent_mapping = {
        "Completion": "code_completion",
        "Generation": "code_generation"
    }
    intent = intent_mapping[code_task]
    pp_assistant.intent = intent

    # Detect programming language
    language = pp_assistant.detect_language(code_input)
    print(f"Detected language: {language}")

    similar_docs = pp_assistant.match_chroma(snippet=code_input, language=language)

    rag_context = ""
    base_path = "./"
    if similar_docs:
        rag_context = similar_docs[0]["snippet"]
        file_path = base_path + similar_docs[0]["source"]

    prompt_args = {"context": rag_context}
    if  language.lower() != "none":
        prompt_args = {"context": rag_context, "language": language}

    result_object = pp_assistant.perform_coding_task(code_input, **prompt_args)
    print(f"Result from LLM: {result_object.to_dict()}")

    bot_message = str(result_object.response)
    bot_message = "\n".join(bot_message.split("\n")[1:-1])
    return bot_message


response = code_completion_generation(
    code_input="""
def Script(name, component, default_options=None, shell='bash'):
  if shell == 'fish':
    return _FishScript(name, _Commands(component), default_options)
  return _BashScript(name, _Commands(component), default_options)
    """,
    code_task="Generation"
)


print(response)