from langchain.llms.bedrock import Bedrock
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import boto3 # connect with amazon client
import streamlit as st 

# Create Bedrock client
bedrock_client = boto3.client(
    service_name = "bedrock-runtime",
    region_name = "us-east-1"

)

model_id =  "meta.llama3-70b-instruct-v1:0"

# Create the LLm object
llm = Bedrock(
    model_id=model_id,
    client = bedrock_client,
    model_kwargs = {"temperature" : 0.9},#ranges from 0 to 1 : 0.9 means more creative output

)

def my_chatbot(language, user_text) :
    prompt = PromptTemplate(
        input_variables=["languages","user_text"],
        template= "You are chatbot. You are in {language}. \n \n {user_text}"
    )

    bedrock_chain = LLMChain(llm = llm, prompt = prompt)

    response = bedrock_chain({'language' : language, 'user_text': user_text })

    return response


st.title("NutriChat Bedrock LLM")

language = st.sidebar.selectbox("Language", ["English","Spanish","Hindi"])

if language :
    user_text = st.sidebar.text_area(label= "What is your question?", max_chars=250)

if user_text :
    response = my_chatbot(language, user_text)
    st.write(response['text'])





