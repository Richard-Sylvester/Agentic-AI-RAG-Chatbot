import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # NEW: Local tool
from langchain_pinecone import PineconeVectorStore

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PDF_FILE = "Ebook-Agentic-AI.pdf"

def main():
    print("--- STARTING LOCAL INGESTION (NO API LIMITS) ---")
    
    # 1. Load PDF
    if not os.path.exists(PDF_FILE):
        print(f"Error: {PDF_FILE} not found.")
        return

    print(f"Loading {PDF_FILE}...")
    loader = PyPDFLoader(PDF_FILE)
    raw_docs = loader.load()
    print(f"   Loaded {len(raw_docs)} pages.")

    # 2. Split Text
    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    documents = text_splitter.split_documents(raw_docs)
    print(f"   Created {len(documents)} chunks.")

    # 3. Embed & Store (LOCALLY)
    print("Initializing Local Model (all-MiniLM-L6-v2)...")
    # This runs on your laptop, not the cloud. 
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print(f"Uploading to Pinecone index '{INDEX_NAME}'...")
    
    # We can send all at once because Pinecone handles the traffic, not Google
    PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
    print("--- SUCCESS: All data stored in Pinecone! ---")

if __name__ == "__main__":
    main()