""""
This .py file creates an interactive Q&A bot.

This assumes that your enviromental variables are stored in a .env file that includes the following:
OPENAI_DEPLOYMENT_ENDPOINT = ???
OPENAI_API_KEY = ???
OPENAI_DEPLOYMENT_NAME = ??? 
OPENAI_DEPLOYMENT_VERSION = ???
OPENAI_MODEL_NAME = ???
OPENAI_EMBEDDING_DEPLOYMENT_NAME = ???
OPENAI_EMBEDDING_MODEL_NAME = ???
"""

import os
from llama_index.llms import AzureOpenAI
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
import logging
import sys
from dotenv import load_dotenv

load_dotenv('/Users/jeana/.env')

logging.basicConfig(
    stream=sys.stdout, level=logging.WARNING
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

api_key = os.environ['OPENAI_API_KEY']
azure_endpoint = os.environ['OPENAI_DEPLOYMENT_ENDPOINT']
api_version = os.environ['OPENAI_DEPLOYMENT_VERSION']


llm = AzureOpenAI(
    model= os.environ['OPENAI_MODEL_NAME'],
    deployment_name= os.environ['OPENAI_DEPLOYMENT_NAME'],
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

# # You need to deploy your own embedding model as well as your own chat completion model
embed_model = AzureOpenAIEmbedding(
    model=os.environ['OPENAI_EMBEDDING_MODEL_NAME'],
    deployment_name=os.environ['OPENAI_EMBEDDING_DEPLOYMENT_NAME'],
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

from llama_index import set_global_service_context

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
)

set_global_service_context(service_context)

documents = SimpleDirectoryReader(
    input_files=["/Users/jeana/Retrieval-Augmented-Generation/LlamaIndex/paul_graham_essay.txt"] #or just indicate the fullpath of the folder containing the data
).load_data()
index = VectorStoreIndex.from_documents(documents)

while True:
    query = input("User: ")
    if query == "exit":
        break
    query_engine = index.as_query_engine()
    answer = query_engine.query(query)
    print(answer.get_formatted_sources())
    print("query was:", query)
    print("answer was:", answer)