import glob
import uuid

import chromadb
import tree_sitter_python as tspython
from aiohttp.log import client_logger
from langchain_google_vertexai import VertexAIEmbeddings
from loguru import logger
from tree_sitter import Language, Parser
import dataclasses
import os
import vertexai
import pandas as pd


# Initialize Vertex AI
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"/Users/ravi/projects/genai-434714-5b6098f8999f.json"
PROJECT_ID = "genai-434714"
LOCATION = "us-central1"
vertexai.init(project=PROJECT_ID, location=LOCATION)


@dataclasses.dataclass
class Snippet:
    """Dataclass for storing Embedded Snippets"""
    id: str
    embedding: list[float] | None
    snippet: str
    filename: str
    language: str

class CodeParser:
    def __init__(self, language: str, node_types: list[str]):
        self.language = language
        self.node_types = node_types

        if self.language == "python":
            PY_LANGUAGE = Language(tspython.language())
            self.parser = Parser(PY_LANGUAGE)

    def parse_file(self, file_path):
        # Extract file content
        with open(file_path, "rb") as f:
            content = f.read()

        try:
            tree = self.parser.parse(content)
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            return

        parsed_snippets = []
        cursor = tree.walk()

        # Walk through the tree
        while cursor.goto_first_child():
            if cursor.node.type in self.node_types:
                parsed_snippets.append(
                    Snippet(
                        id=str(uuid.uuid4()),
                        embedding=None,
                        snippet=cursor.node.text,
                        filename=file_path,
                        language=self.language,
                    )
                )
            while cursor.goto_next_sibling():
                if cursor.node.type in self.node_types:
                    print("###############################")
                    print(cursor.node.text)
                    print("###############################")

                    parsed_snippets.append(
                        Snippet(
                            id=str(uuid.uuid4()),
                            embedding=None,
                            snippet=cursor.node.text,
                            filename=file_path,
                            language=self.language,
                        )
                    )

        return parsed_snippets

    def parse_directory(self, directory_path):
        parsed_contents = []

        for filename in glob.glob(f"{directory_path}/**//*.py", recursive=True):
            parsed_content = self.parse_file(filename)
            parsed_contents.extend(parsed_content)

        return parsed_contents

# Parse python files from local code repo.
parser = CodeParser(
    language = "python",        # TODO: Identify the programming language automatically future.
    node_types = ["class_definition", "function_definition"]
)

parsed_snippets = parser.parse_directory("python-fire")
print("Parsing done")


# Generate embeddings for the parsed contents.
embedder = VertexAIEmbeddings(model_name="textembedding-gecko@003")
snippet_texts = list(map(lambda x: x.snippet.decode("ISO-8859-1"), parsed_snippets))   # Convert to UTF-8 format
embedded_texts = embedder.embed_documents(texts = snippet_texts)

embedded_snippets = []
for code_text, embedding, snippet in zip(snippet_texts, embedded_texts, parsed_snippets):
    snippet.snippet = code_text
    snippet.embedding = embedding
    embedded_snippets.append(snippet)

print("Embedding done")

# Convert snippets to DataFrame for ChromaDB Ingestion
def to_dataframe_row(embedded_snippets: list[Snippet]):
    outputs = []
    for embedded_snippet in embedded_snippets:
        output = {
            "ids": embedded_snippet.id,
            "embeddings": embedded_snippet.embedding,
            "snippets": embedded_snippet.snippet,
            "metadatas": {
                "filename": embedded_snippet.filename,
                "language": embedded_snippet.language,
            },
        }
        outputs.append(output)
    return outputs

data = pd.DataFrame(to_dataframe_row(embedded_snippets))

PERSISTENT_DISK_DIR = "chromadb"
collection_name = "coderag"

client = chromadb.PersistentClient(PERSISTENT_DISK_DIR)
collection = client.get_or_create_collection(
    name = collection_name,
    metadata={"hnsw:space": "cosine"}
)
collection.add(
    documents = data["snippets"].tolist(),
    embeddings = data["embeddings"].tolist(),
    metadatas = data["metadatas"].tolist(),
    ids = data["ids"].tolist(),
)
client = None
print ("Code Ingestion Completed")


