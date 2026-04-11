from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import chromadb
import os
import logging
import pdfplumber
import re
import pandas as pd
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Configure environment variables
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
CHROMADB_PATH = os.getenv("CHROMADB_PATH", os.path.join(os.getcwd(), "chroma_db"))

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)

# Define the custom embedding function for ChromaDB using Ollama
class ChromaDBEmbeddingFunction:
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return self.langchain_embeddings.embed_documents(input)

embedding = ChromaDBEmbeddingFunction(
    OllamaEmbeddings(model=LLM_MODEL, base_url=OLLAMA_URL)
)

# Define collection in ChromaDB
collection_name = "rag_collection_demo_1"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={"description": "A collection for RAG with Ollama - Demo1"},
    embedding_function=embedding
)

# Pydantic model for input query
class QueryRequest(BaseModel):
    question: str

# Function to extract questions and answers from a PDF
def extract_questions_and_answers(pdf_path: str):
    questions, answers, current_answer = [], [], ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                words = page.extract_words()
                page_text = " ".join([word['text'] for word in words])
                segments = re.split(r'(\d+\..*?\?)', page_text)
                
                for segment in segments:
                    if re.match(r'^\d+\..*?\?', segment.strip()):
                        if current_answer:
                            answers.append(current_answer.strip())
                            current_answer = ""
                        questions.append(segment.strip())
                    else:
                        current_answer += segment.strip() + " "

            if current_answer:
                answers.append(current_answer.strip())
    except Exception as e:
        logger.error(f"Error extracting Q&A from PDF: {e}")
        raise HTTPException(status_code=500, detail="Error processing PDF")
    
    return questions, answers

# Function to add documents to the ChromaDB collection
def add_documents_to_collection(documents, ids):
    try:
        collection.add(documents=documents, ids=ids)
    except Exception as e:
        logger.error(f"Error adding documents to ChromaDB: {e}")
        raise HTTPException(status_code=500, detail="Error adding documents to database")

# Function to query ChromaDB
def query_chromadb(query_text, n_results=1):
    try:
        results = collection.query(query_texts=[query_text], n_results=n_results)
        return results["documents"], results["metadatas"]
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}")
        raise HTTPException(status_code=500, detail="Error querying the database")

# Function to interact with Ollama
def query_ollama(prompt: str) -> str:
    try:
        llm = OllamaLLM(model=LLM_MODEL)
        return llm.invoke(prompt)
    except Exception as e:
        logger.error(f"Error querying Ollama LLM: {e}")
        raise HTTPException(status_code=500, detail="Error querying LLM")

# Function for RAG pipeline
def rag_pipeline(query_text: str) -> str:
    retrieved_docs, _ = query_chromadb(query_text)
    context = " ".join(retrieved_docs[0]) if retrieved_docs else "No relevant documents found."
    augmented_prompt = f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
    logger.info(f"Augmented prompt: {augmented_prompt}")
    response = query_ollama(augmented_prompt)
    return response

@app.post("/ingest/")
async def ingest():
    background=BackgroundTasks()
    background.add_task(func=add_documents_background_task)
    
    return {"answer": "your request is being processed check logs for completion"}


@app.post("/ask/")
async def ask(query_request: QueryRequest, background_tasks: BackgroundTasks):
    bot_instruction = "You are an expert in vannamei shrimp, and users are asking the below queries. Please respond in a way that avoids causing financial loss to the users."
    user_query = query_request.question
    query = bot_instruction + " " + user_query
    answer=rag_pipeline(query)
    
    #background_tasks.add_task(rag_pipeline, query)  # Offload the RAG pipeline to background task
    return {"answer": answer}

# Background task to add documents to ChromaDB (if not already done)
def add_documents_background_task():
    try:
        pdf_path = 'vannamei_FAQ.pdf'
        questions, answers = extract_questions_and_answers(pdf_path)
        df = pd.DataFrame({"questions": questions, "answers": answers[1:]})
        documents = df["answers"].to_list()
        doc_ids = [f"doc_{i}" for i in range(len(df))]
        add_documents_to_collection(documents, doc_ids)
    except Exception as e:
        logger.error(f"Error in background task: {e}")

# Start the FastAPI application
if __name__ == "__main__":
    import uvicorn
    #background=BackgroundTasks()
    # Add a background task for adding documents to the collection when the server starts
    #background.add_task(func=add_documents_background_task)
    uvicorn.run(app, host="0.0.0.0", port=5000)
