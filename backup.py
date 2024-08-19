# Suicide Prevention - Therapy ChatBot
# Guys this file contains the backup. Here the voice feature will not be there.

import smtplib # Protocol to interact with email
from email.mime.text import MIMEText # Mime Object
from langchain_community.document_loaders import PyPDFLoader # For loading the Pdf
from langchain.prompts import PromptTemplate # Prompt template
from langchain_pinecone import PineconeVectorStore # Vector Database
from langchain.text_splitter import RecursiveCharacterTextSplitter # 
from langchain.chains import RetrievalQA # For Retrieval
from langchain_groq import ChatGroq # Inference Engine
from dotenv import load_dotenv # For detecting env variables
from langchain.embeddings import OllamaEmbeddings # To perform vector embeddings
import chainlit as cl # For user interface

load_dotenv() # Detecting env 

# Defining prompt 
prompt_template = """ 

    You are a therapy assistant. If a person comes with sad or dull mood because of some issues, you are there to help them. Don't answer to unrelated context.
    Context: {context} Question: {question}

    Helpful answer:
    
"""

# Just created a function to interact with the prompt template
def set_custom_prompt():
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])  
    return prompt

# Defined this function to perform retrieval
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=db.as_retriever(), chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# This function is for defining llm model
def load_llm():
    groqllm = ChatGroq(
        model ="llama3-8b-8192", temperature=0
    )
    return groqllm

# Here just loading the pdf and transforming it to chunks, and performing vector embeddings as well as storing the vector embeddings in Pinecone vector database.
def qa_bot():
    
    data = PyPDFLoader('R:\\SPD\\TherapyBot\\Psychology-of-Human-Relations-1695056913.pdf')
    loader = data.load()
    chunk = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=0)
    splitdocs = chunk.split_documents(loader)
    index_name = "langchain4"
    db = PineconeVectorStore.from_documents(splitdocs[:5], OllamaEmbeddings(model ="mxbai-embed-large"),index_name=index_name)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

# This functionality is for redirecting to a special email when the system found words related to suicide (static).
def send_notification(email, message):
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    smtp_username = 'badrisrp3836@gmail.com'
    smtp_password = 'hngb nzfa prsd adcy'

    subject = 'Suicidal Attempt Detected'
    body = f"Conversation related to suicidal attempts:\n\n{message}"

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = 'therapybot@example.com'
    msg['To'] = email

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.sendmail(msg['From'], msg['To'], msg.as_string())


# This chainlit decorator is for starting the app
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to the Therapy Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

suicidal_keywords = ['suicide', 'self-harm', 'end my life', 'suicidal thoughts', 'kill myself']

# Here defined the main functionality for redirection 
@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    if chain is None:
        return
    for keyword in suicidal_keywords:
        if keyword in message.content.lower():
            # This is the email that I want to redirect, in future it will be any organization email.
            email = 'badrisrp3836@gmail.com'   
            send_notification(email, message.content)
            break  

    res = await chain.acall({'query': message.content})
    answer = res['result']

    await cl.Message(content=answer).send()