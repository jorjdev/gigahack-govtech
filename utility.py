from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain. prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from llama_index import LangchainEmbedding, LLMPredictor, VectorStoreIndex, SimpleDirectoryReader, ServiceContext, set_global_service_context, StorageContext, load_index_from_storage
import openai
from llama_index.retrievers import VectorIndexRetriever

openai.api_type = "azure"
openai.api_base = "https://gigahack.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = 'c5ff58bc471549f0afb6c2ced47d7c45'


llm = AzureChatOpenAI(openai_api_base=openai.api_base,
                      openai_api_version=openai.api_version,
                      deployment_name="team-busgo",
                      openai_api_key=openai.api_key,
                      openai_api_type=openai.api_type, temperature=0)
embedding_llm = LangchainEmbedding(
    OpenAIEmbeddings(
        model="text-embedding-ada-002",
        deployment="team-busgo-ada",
        openai_api_key=openai.api_key,
        openai_api_base=openai.api_base,
        openai_api_type=openai.api_type,
        openai_api_version=openai.api_version,
    ),
    embed_batch_size=1
)
prompt = PromptTemplate(input_variables=['history', 'input'], output_parser=None, partial_variables={
}, template='The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details only from its context.The AI does not answer to questions outside the context, it truthfully says it does not know.\n\nCurrent conversation:\n{history}\nHuman: {input}\nAI:', template_format='f-string', validate_template=True)
documents = SimpleDirectoryReader("./data").load_data()
llm_predictor = LLMPredictor(llm=llm)
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    embed_model=embedding_llm,
    prompt_helper=prompt
)

set_global_service_context(service_context)

index = VectorStoreIndex.from_documents(
    documents, service_context=service_context)
index.storage_context.persist("dog_index")
storage_context = StorageContext.from_defaults(persist_dir="dog_index")

# Load index from the storage context
new_query_engine = index.as_query_engine()
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=2,
)
conversation = ConversationChain(
    llm=llm,
    prompt=prompt,
    memory=ConversationBufferMemory()
)


def chatbot(pt):
    res = conversation.predict(input=pt)

    return res


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
