# 📖 Agentic AI RAG Chatbot

An "Agentic" RAG (Retrieval-Augmented Generation) chatbot capable of answering questions strictly based on the content of the **Agentic AI eBook**.

Built for the **AI Engineer Intern** assignment.

## 🚀 Features
- **Strict RAG Pipeline:** Answers are grounded strictly in the provided PDF.
- **Agentic Logic:** Uses **LangGraph** to orchestrate the retrieval and generation workflow.
- **Hybrid Architecture (Cost-Optimized):**
  - **Vector DB:** Pinecone (Serverless) for fast retrieval.
  - **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Local/CPU-optimized).
  - **LLM:** Google Gemini 2.0 Flash (via `langchain-google-genai`).
- **Robust Error Handling:** Handles API rate limits with automatic retries.
- **Transparency:** Displays the exact "Source Context" chunks used for answers.

## 🛠️ Tech Stack
- **Python 3.11**
- **LangChain & LangGraph**
- **Pinecone**
- **Google Gemini API**
- **Streamlit**

## ⚙️ Setup Instructions

### 1. Clone the Repository
````bash
git clone [https://github.com/Richard-Sylvester/Agentic-AI-RAG-Chatbot.git](https://github.com/Richard-Sylvester/Agentic-AI-RAG-Chatbot.git)
cd Agentic-AI-RAG-Chatbot 
````

### 2. Create a Virtual Environment
Windows:
```bash
python -m venv venv
venv\Scripts\activate
````

### 3. Install Dependencies
```bash
pip install -r requirements.txt
````

### 4. Configure Environment Variables
Create a .env file in the root directory and add your API keys:
```bash
GOOGLE_API_KEY="your_google_api_key_here"
PINECONE_API_KEY="your_pinecone_api_key_here"
PINECONE_INDEX_NAME="agentic-ai"
````

### 5. Ingest Data
Run the ingestion script to process the PDF:
```bash
python ingest.py
````

### 6. Run the Chatbot
Launch the Streamlit app:
```bash
streamlit run app.py
````


## 🧠 Architecture
The system follows a graph-based workflow using LangGraph:

**Ingestion:** The PDF is chunked and embedded using the local all-MiniLM-L6-v2 model, then stored in a Pinecone Serverless index (Dimensions: 384).

**Retrieve Node:** The user's query is converted into an embedding locally. It queries Pinecone to find the top relevant text chunks.

**Generate Node:** The retrieved context + the user's question are sent to Google Gemini 2.0 Flash. A strict system prompt ensures the answer is based only on the context.

**UI:** Streamlit renders the chat interface and exposes the "Source Context" expander for validation.


### 🧪 Sample Queries
Try asking these questions to test the RAG capability:

"What is the definition of Agentic AI?"

"How is Agentic AI different from standard Generative AI?"

"What are the core components of an Agentic system?"

"Explain the 'Agency' component in the framework."



Submitted by Richard Sylvester
