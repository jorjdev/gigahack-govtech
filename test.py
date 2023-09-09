import os
import json
import openai
import pickle
from langchain.embeddings import OpenAIEmbeddings
from llama_index.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from llama_index import LangchainEmbedding, VectorStoreIndex, SimpleDirectoryReader, ServiceContext, set_global_service_context, StorageContext, load_index_from_storage
import logging
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationChain,ConversationalRetrievalChain
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

openai.api_type = "azure"
openai.api_base = "https://gigahack.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = 'c5ff58bc471549f0afb6c2ced47d7c45'

llm = AzureChatOpenAI(openai_api_base=openai.api_base,
                      openai_api_version=openai.api_version,
                      deployment_name="team-busgo",
                      openai_api_key=openai.api_key,
                      openai_api_type=openai.api_type)

# You need to deploy your own embedding model as well as your own chat completion model
embedding_llm = LangchainEmbedding(
    OpenAIEmbeddings(
        model="text-embedding-ada-002",
        deployment="team-busgo-ada",
        openai_api_key=openai.api_key,
        openai_api_base=openai.api_base,
        openai_api_type=openai.api_type,
        openai_api_version=openai.api_version,
    ),
    embed_batch_size=1,
)


documents = SimpleDirectoryReader("./data").load_data()

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embedding_llm,
)

set_global_service_context(service_context)

index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist("dog_index")
storage_context = StorageContext.from_defaults(persist_dir="dog_index")

# Load index from the storage context
new_index = load_index_from_storage(storage_context)
new_query_engine = new_index.as_query_engine()


memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)
query = "What is the name of the dog?"
conversation = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=ConversationBufferMemory()
)
response = conversation.predict(input=query)

print(response)
