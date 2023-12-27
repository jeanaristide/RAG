import os
import pandas as pd
from llama_index.llms import AzureOpenAI
from llama_index.embeddings import AzureOpenAIEmbedding
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext, load_index_from_storage, set_global_service_context
import logging
import sys
from dotenv import load_dotenv
from llama_index.llms import Ollama
import numpy as np
from trulens_eval import TruLlama, Feedback, Tru, feedback
from langchain.llms import Ollama
from trulens_eval.tru_custom_app import instrument
from trulens_eval import LiteLLM
import litellm
litellm.set_verbose=False
tru = Tru()


class modelException(Exception):
    def __init__(self, invalid_value, allowed_values):
        self.invalid_value = invalid_value
        self.allowed_values = allowed_values
        message = f"Invalid value: {invalid_value}. Allowed values are: {', '.join(allowed_values)}"
        super().__init__(message)

def feedbacks(llm_name):

    ##### INITIALZE FEEDBACK FUNCTION(S)#######
    if llm_name == 'azureoai':
        # Initialize AzureOpenAI-based feedback function collection class:
        llm_provider = feedback.AzureOpenAI(
                                        deployment_name=os.environ['OPENAI_DEPLOYMENT_NAME'],
                                        api_key = os.environ['OPENAI_API_KEY'],
                                        api_version=os.environ['OPENAI_DEPLOYMENT_VERSION'],
                                        azure_endpoint=os.environ['OPENAI_DEPLOYMENT_ENDPOINT'],
                                        # model = os.environ['OPENAI_MODEL_NAME']
                                        )
    elif llm_name == 'llama2':
        llm_provider = LiteLLM(model_engine="ollama/llama2", api_base='http://localhost:11434')

    # Question/answer relevance between overall question and answer.
    f_qa_relevance = Feedback(llm_provider.relevance, name = "Answer Relevance").on_input_output()

    # Question/statement relevance between question and each context chunk.
    f_qs_relevance = Feedback(llm_provider.qs_relevance, name = "Context Relevance").on_input().on(
        TruLlama.select_source_nodes().node.text
    ).aggregate(np.mean)

    # groundedness of output on the context
    groundedness = feedback.Groundedness(
                    # summarize_provider=azopenai, 
                    groundedness_provider=llm_provider)
    f_groundedness = Feedback(groundedness.groundedness_measure, name = "Groundedness").on(TruLlama.select_source_nodes().node.text).on_output()
    
    feedbacks=[f_groundedness, f_qa_relevance, f_qs_relevance]
    
    return feedbacks

class RAG():
    llm_list = ['llama2', 'gpt-35-turbo']
    embedding_list = ['text-embedding-ada-002', 'sentence-transformers/all-mpnet-base-v2', "BAAI/bge-small-en-v1.5"]

    def __init__(self, llm_name, embedding_model_name, documents, feedbacks):

        self.feedbacks = feedbacks
        self.llm_name = llm_name
        self.embedding_model_name = embedding_model_name
           
        if llm_name not in self.llm_list:
            raise modelException(llm_name, self.llm_list)
        if embedding_model_name not in self.embedding_list:
            raise modelException(embedding_model_name, self.embedding_list)        

        ## Getting the LLM for prediction
        if llm_name in ['gpt-35-turbo']:
            self.llm = AzureOpenAI(
                    # model= os.environ['OPENAI_MODEL_NAME'],
                    model = llm_name,
                    deployment_name= os.environ['OPENAI_DEPLOYMENT_NAME'],
                    api_key=os.environ['OPENAI_API_KEY'],
                    azure_endpoint=os.environ['OPENAI_DEPLOYMENT_ENDPOINT'],
                    api_version=os.environ['OPENAI_DEPLOYMENT_VERSION'],
                )
        elif llm_name == 'llama2':
            self.llm = Ollama(model="llama2")
        
        ## Gettting the embedding model
        if embedding_model_name == 'text-embedding-ada-002':
            self.embedding_model = AzureOpenAIEmbedding(
                    # model=os.environ['OPENAI_EMBEDDING_MODEL_NAME'],
                    model=embedding_model_name,
                    deployment_name=os.environ['OPENAI_EMBEDDING_DEPLOYMENT_NAME'],
                    api_key=os.environ['OPENAI_API_KEY'],
                    azure_endpoint=os.environ['OPENAI_DEPLOYMENT_ENDPOINT'],
                    api_version=os.environ['OPENAI_DEPLOYMENT_VERSION'],
                )
        elif embedding_model_name in ['sentence-transformers/all-mpnet-base-v2', "BAAI/bge-small-en-v1.5"]:
            self.embedding_model = HuggingFaceEmbeddings(
                    model_name=embedding_model_name)
        
        #Set service context, documents and Index
        self.service_context = ServiceContext.from_defaults(embed_model=self.embedding_model, llm = self.llm)

        set_global_service_context(self.service_context)

        # check if storage already exists (under the specific embedding model)
        if not os.path.exists("./storage/" + embedding_model):
            # load the documents and create the index
            self.documents = documents
            self.index = VectorStoreIndex.from_documents(self.documents, service_context=self.service_context)
            # store it for later
            self.index.storage_context.persist("./storage/" + embedding_model)
            self.storage_context = None
        else:
            # load the existing index from the specific embedding model
            self.storage_context = StorageContext.from_defaults(persist_dir="./storage/" + embedding_model)
            self.index = load_index_from_storage(self.storage_context)

        # self.documents = SimpleDirectoryReader(
        #             input_files=[r"/Users/jeana/Retrieval-Augmented-Generation/LlamaIndex/paul_graham_essay.txt"] #or just indicate the fullpath of the folder containing the data
        #                         ).load_data()
        # self.index = VectorStoreIndex.from_documents(self.documents, service_context=self.service_context)

    def query(self, query: str) -> str:
        ### INSTRUMENT CHAIN FOR LOGGING WITH TRULENS

        query_engine = self.index.as_query_engine()

        tru_query_engine_recorder = TruLlama(query_engine,
                app_id= "RAG_" + self.llm_name + '_' + self.embedding_model_name,
                feedbacks=self.feedbacks)

        with tru_query_engine_recorder as recorder:
            answer = query_engine.query(query)
            print(answer.get_formatted_sources())
            print("query was:", query)
            print("answer was:", answer)

def show_metrics(tru):
    records, feedback = tru.get_records_and_feedback(app_ids=[])
    pd.set_option("display.max_colwidth", None)
    return records[['app_id','ts', "input", "output", 'latency', 'total_cost'] + feedback]

def show_leaderboard(tru):
    return tru.get_leaderboard(app_ids=[])