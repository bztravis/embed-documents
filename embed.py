from pymilvus import MilvusClient
from langchain.document_loaders import UnstructuredFileLoader

import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
import textract


doc = textract.process("./dispactManual.pdf")

with open('dispactManul.txt', 'w') as f:
    f.write(doc.decode('utf-8'))

with open('dispactManul.txt', 'r') as f:
    text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=512,
    chunk_overlap=200,
)

chunks = text_splitter.create_documents([text])
for chunk in chunks:
    print(chunk, "\n\n\n\n\n")

# Get embedding model
# embeddings = OpenAIEmbeddings()


# Initialize a MilvusClient instance
# Replace uri and token with your own
client = MilvusClient(
    # Cluster endpoint obtained from the console
    uri="https://in03-575b131d648efbe.api.gcp-us-west1.zillizcloud.com",
    # - For a serverless cluster, use an API key as the token.
    # - For a dedicated cluster, use the cluster credentials as the token
    # in the format of 'user:password'.
        token="ade46ba572172c1a5655d38a460e85df620e2367068dbec49cb913ff491249881e5182d8f07dbacd0af53751289a389eef8715b8"
)

res = client.describe_collection(
    collection_name='OperatorTraining'
)

print(res)

# loader = UnstructuredFileLoader("./example.txt", mode="elements")
# docs = loader.load()
# # print(docs)
# print(docs[:5])
