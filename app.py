from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import pinecone
from langchain_community import vectorstores
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from src.prompt import *
import os
from pinecone import Pinecone

app=Flask(__name__)

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')



pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = 'mchatbot'
# index = pc.Index(index_name)

embeddings = download_hugging_face_embeddings()


docsearch=PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)


PROMPT=PromptTemplate(template=prompt_template, input_variables=["context","question"])
chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="llama-2-7b-chat.ggmlv3.q8_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.4})

qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)

# result=qa({"query": "How to treat IBS?"})
# print("Response : ", result["result"])

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(msg)
    result=qa({"query": msg})
    print("Response : ", result["result"])
    return str(result["result"])

if __name__=="__main__":
    app.run(debug=True)


