import os
import time
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# Imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from google.api_core.exceptions import ResourceExhausted

load_dotenv()

# --- 1. STATE ---
class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str

# --- 2. TOOLS ---
# A. Search (Local)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME"), 
    embedding=embeddings
)
retriever = vectorstore.as_retriever()

# B. Writer (Google) - switched to "Lite" for better limits
# We use the specific version from your list
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# --- 3. NODES ---
def retrieve_node(state: AgentState):
    question = state["question"]
    print(f"--- RETRIEVING: {question} ---")
    documents = retriever.invoke(question)
    context_text = [doc.page_content for doc in documents]
    return {"context": context_text}

def generate_node(state: AgentState):
    print("--- GENERATING ANSWER ---")
    question = state["question"]
    context = "\n\n".join(state["context"])
    
    template = """You are a helpful AI assistant for the 'Agentic AI' eBook.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    # --- RETRY LOGIC (The Fix) ---
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = chain.invoke({"context": context, "question": question})
            return {"answer": response}
        except ResourceExhausted:
            print(f"⚠️ Quota hit. Waiting 30s before retry {attempt+1}/{max_retries}...")
            time.sleep(30)
        except Exception as e:
            return {"answer": f"Error: {str(e)}"}
            
    return {"answer": "I am currently overloaded with requests. Please try again in 1 minute."}

# --- 4. GRAPH ---
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()
