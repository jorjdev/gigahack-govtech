from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain. prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from llama_index import LangchainEmbedding, LLMPredictor, VectorStoreIndex, SimpleDirectoryReader, ServiceContext, set_global_service_context, StorageContext, load_index_from_storage
import openai
from llama_index.evaluation import ResponseEvaluator
import sys
import os
openai.api_type = "azure"
openai.api_base = "https://gigahack.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = 'c5ff58bc471549f0afb6c2ced47d7c45'


llm = AzureChatOpenAI(openai_api_base=openai.api_base,
                      openai_api_version=openai.api_version,
                      deployment_name="team-busgo",
                      openai_api_key=openai.api_key,
                      openai_api_type=openai.api_type, temperature=0)
embedding_llm = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    deployment="team-busgo-ada",
    openai_api_key=openai.api_key,
    openai_api_base=openai.api_base,
    openai_api_type=openai.api_type,
    openai_api_version=openai.api_version,
)
loader = TextLoader("./data/test.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

vectorstore = Chroma.from_documents(documents, embedding_llm)

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

qa_template = """Your goal is to provide accurate answers from the context provided.Don't include any prefixes.Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
CONTEXT :
    <ctx> {context} </ctx>  HISTORY :
 <hs> {history} </hs> Question: {question}"""
QA_PROMPT = PromptTemplate(template=qa_template, input_variables=[
                           "history", "context", "question"])
retrieve_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(), chain_type_kwargs={
    "verbose": False,
    "prompt": QA_PROMPT,
    "memory": ConversationBufferMemory(
        memory_key="history",
        input_key="question"),
},
    verbose=True)


def chatbot(pt):
    result = retrieve_qa.run(pt)
    return result


if __name__ == '__main__':
    while True:
        print('########################################\n')
        pt = input('ASK: ')
        if pt.lower() == 'end':
            break
        response = chatbot(pt)
        print('\n----------------------------------------\n')
        print('ChatGPT says: \n')
        print(response, '\n')
