from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import pinecone
import pinecone
from dotenv import load_dotenv
import os
load_dotenv()


PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')

extracted_data=extracted_data = load_pdf("data/")


text_chunks=text_split(extracted_data)

embeddings = download_hugging_face_embeddings()

from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = 'mchatbot'
index = pc.Index(index_name)

from langchain_pinecone import PineconeVectorStore
docsearch=PineconeVectorStore.from_documents(documents=text_chunks, embedding=embeddings, index_name=index_name)


