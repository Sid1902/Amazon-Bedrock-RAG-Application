import boto3
import streamlit as st
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter # to create chunks
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

prompt_template="""
Human : Use the following pieces if cotext to provide a concise answer to the question at the 
end but use atleast summarize with 250 words with detailed explanations. If you dont know the +
answer, just say that you dont know, don't try to make up an answer.
<context>
{context}


</context>

Question = {question}

Assistant :

"""

# Create Bedrock client
bedrock_client = boto3.client(
    service_name = "bedrock-runtime",
    region_name = "us-east-1"

)

# get Embedding model

bedrock_embedding = BedrockEmbeddings(model_id = "amazon.titan-embed-text-v1", 
                                      client = bedrock_client)


# how to load the documents

def get_documents() :
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    # Create chunks 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 500)

    docs = text_splitter.split_documents(documents)

    return docs

# now create knowledge base using FaissDB
def get_vector_store(docs) :
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embedding
    )
    vectorstore_faiss.save_local("faiss_index")

def get_llm():

    llm = Bedrock(model_id="ai21.j2-mid-v1",
                  client=bedrock_client,model_kwargs={'maxTokens':512,'temperature':0.9})
    
    return llm

 
PROMPT = PromptTemplate(
    template=prompt_template,input_variables=["context","question"]
)

def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type = "stuff",
    retriever = vectorstore_faiss.as_retriever(
        search_type = "similarity", search_kwargs={"k":3} # Till 3 ranked documents
    ),
    return_source_documents = True,
    chain_type_kwargs={"prompt":PROMPT}

    )
    answer = qa({"query":query})

    return answer['result']



def main():
    st.set_page_config("Nutrichat")
    st.header("NutriChat : End to end RAG Application")

    user_question = st.text_input("Ask a question from the PDF files")

    # load the documents . Extract them . create the embedding and Store them in vectorDB

    with st.sidebar :
        st.title("Update or create Vector Store : ")

        if st.button('Get Vector Embedding') :
            with st.spinner("Processing......") :
                docs = get_documents()

                get_vector_store(docs)
                st.success("Done")

    if st.button("Ask Question") :
        with st.spinner("Processing...") :
            faiss_index = FAISS.load_local(index_name="index",
                                           embeddings=bedrock_embedding,allow_dangerous_deserialization=True,folder_path="faiss_index")
            llm = get_llm()
            st.write(get_response_llm(llm,faiss_index,user_question))


if __name__ =="__main__":
    main()
#E:\Project\Amazon-Bedrock-RAG-Application\faiss_index