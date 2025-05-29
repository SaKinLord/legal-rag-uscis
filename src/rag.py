# src/rag.py

import os
import chromadb
from chromadb.utils import embedding_functions # Or from chromadb import embedding_functions
import google.generativeai as genai
from src.config import GEMINI_API_KEY # Import the API key

# --- Configuration ---
VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_db')
COLLECTION_NAME = "aao_decisions"
EMBEDDING_MODEL_NAME_FOR_QUERY = "BAAI/bge-small-en-v1.5" # Must match model used for storing
LLM_MODEL_NAME = "gemini-1.5-flash-latest" # Or "gemini-1.0-pro" or other compatible model

# Number of relevant chunks to retrieve
TOP_K_CHUNKS = 8 # Increased from 5 to capture more relevant context

# --- Initialize ChromaDB Client and Embedding Function for Querying ---
client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

# Embedding function for user queries (must be the same as used for ingestion)
try:
    query_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME_FOR_QUERY
    )
except AttributeError:
    from chromadb.embedding_functions import SentenceTransformerEmbeddingFunction
    query_ef = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME_FOR_QUERY
    )

# Get the existing collection
try:
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=query_ef # Good to specify, ensures consistency
    )
except Exception as e:
    print(f"Error getting collection '{COLLECTION_NAME}': {e}")
    print("Please ensure you have run store.py to create and populate the collection.")
    collection = None # Set to None if collection doesn't exist

# --- Initialize Google Gemini ---
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # model = genai.GenerativeModel(LLM_MODEL_NAME) # Initialize model here or per request
else:
    print("Gemini API Key not configured. LLM generation will not be available.")
    # model = None

# --- Core RAG Functions ---

def retrieve_relevant_chunks(query_text, n_results=TOP_K_CHUNKS):
    """
    Embeds the query and retrieves the top_k relevant chunks from ChromaDB.
    """
    if not collection:
        print("ChromaDB collection is not available. Cannot retrieve chunks.")
        return []
    if not query_text:
        print("Query text is empty. Cannot retrieve chunks.")
        return []

    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances'] # documents are the chunk texts
        )
        return results
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return []

def format_chunks_for_prompt(retrieved_chunks_results):
    """
    Formats the retrieved chunks into a string for the LLM prompt,
    including metadata for citation.
    """
    if not retrieved_chunks_results or not retrieved_chunks_results.get('documents'):
        return "No relevant context found in the documents."

    context_str = "Context Passages:\n\n"
    # retrieved_chunks_results['documents'] is a list of lists (one per query_text)
    # retrieved_chunks_results['metadatas'] is also a list of lists
    
    doc_list = retrieved_chunks_results['documents'][0]
    meta_list = retrieved_chunks_results['metadatas'][0]
    dist_list = retrieved_chunks_results['distances'][0] # For showing relevance scores

    for i, (chunk_text, chunk_meta) in enumerate(zip(doc_list, meta_list)):
        context_str += f"--- CONTEXT {i+1} ---\n"
        context_str += f"Source Document ID: {chunk_meta.get('document_id', 'N/A')}\n"
        context_str += f"Chunk ID: {chunk_meta.get('chunk_id', 'N/A')}\n" # We added this in store.py
        context_str += f"Case Name: {chunk_meta.get('case_name', 'N/A')}\n"
        context_str += f"Publication Date: {chunk_meta.get('publication_date', 'N/A')}\n"
        context_str += f"Relevance Score (Distance): {dist_list[i]:.4f}\n" # Added for debugging
        # context_str += f"Original Start Index: {chunk_meta.get('start_index', 'N/A')}\n" # If needed
        context_str += f"Passage Text:\n{chunk_text}\n"
        context_str += "--- END CONTEXT ---\n\n"
        
    return context_str.strip()

def generate_answer_with_llm(query_text, formatted_context):
    """
    Sends the query and context to the LLM (Gemini) and gets an answer.
    """
    if not GEMINI_API_KEY:
        return "LLM generation is not available (API key missing)."
    if not query_text:
        return "Query is empty."

    # System prompt (or instruction part of the user prompt for Gemini)
    # For Gemini, it's often better to put instructions directly in the prompt to the model.
    # The concept of a separate "system prompt" is more prominent in OpenAI's chat models.
    
    # Gemini often works well with direct instructions.
    # We need to be very specific about using ONLY the provided context and the citation format.
    
    prompt_template = f"""You are a specialized legal assistant. Your task is to answer questions about USCIS AAO I-140 Extraordinary Ability decisions.
You MUST use ONLY the information contained within the 'Context Passages' provided below to answer the 'User Query'. Do not use any external knowledge or make assumptions.
Synthesize your answer using any relevant information found across any of the provided context passages.
Focus on how the details in the context relate to the specific aspects of the User Query, even if the term 'sustained acclaim' is discussed separately from award characteristics.

For every piece of information you use from the context to construct your answer, you MUST provide a precise citation AT THE END OF THE SENTENCE OR CLAUSE that uses the information.
The citation format to follow is: [Case Name, Publication Date, Document ID: ACTUAL_DOCUMENT_ID, Chunk ID: ACTUAL_CHUNK_ID]. 
Ensure you replace ACTUAL_DOCUMENT_ID and ACTUAL_CHUNK_ID with the real values from the context metadata.
For example: [In Re: 12345678, 2025-02-13, Document ID: February_13__2025_FEB132025_03B2203, Chunk ID: February_13__2025_FEB132025_03B2203_chunk_a1fc7e2b]

Be concise and directly answer the user's query.

If, after careful review of all provided context passages, the information needed to directly answer the User Query is not present, you MUST state: "The provided context does not contain sufficient information to answer this question."

{formatted_context}

User Query: {query_text}

Answer:
"""

    try:
        print("\n--- Sending prompt to LLM (Gemini)... ---")
        # print(f"Full Prompt being sent:\n{prompt_template}") # For debugging, can be very long
        
        model = genai.GenerativeModel(LLM_MODEL_NAME) # Initialize the model
        
        # Configure generation parameters if needed (optional)
        # generation_config = genai.types.GenerationConfig(
        #     temperature=0.2, # Lower for more factual, less creative
        #     # top_p=0.9,
        #     # top_k=40,
        #     # max_output_tokens=1024 
        # )

        response = model.generate_content(
            prompt_template,
            # generation_config=generation_config # Optional
            # safety_settings=... # Optional, to adjust safety filters
        )
        
        print("--- LLM response received. ---")
        # Accessing the text part of the response.
        # For Gemini, response.text is usually the direct way.
        # Check response.parts if response.text is empty or for more complex outputs.
        if response.parts:
            # Concatenate text from all parts if multiple exist
            llm_answer = "".join(part.text for part in response.parts if hasattr(part, 'text'))
        elif hasattr(response, 'text') and response.text:
            llm_answer = response.text
        else:
            llm_answer = "Error: LLM response was empty or in an unexpected format."
            print(f"Full LLM Response object: {response}") # For debugging

        return llm_answer.strip()

    except Exception as e:
        print(f"Error during LLM generation: {e}")
        # You might want to inspect `response.prompt_feedback` if available for safety blocks
        if hasattr(e, 'response') and hasattr(e.response, 'prompt_feedback'):
             print(f"Prompt Feedback: {e.response.prompt_feedback}")
        return f"Error generating answer from LLM: {e}"


def answer_query(user_query):
    """
    Full RAG pipeline: retrieve, format context, generate answer.
    """
    print(f"Received user query: {user_query}")
    
    retrieved_chunks = retrieve_relevant_chunks(user_query)
    if not retrieved_chunks or not retrieved_chunks.get('documents') or not retrieved_chunks['documents'][0]:
        print("No relevant chunks found for the query.")
        # Decide if you want to still try the LLM or return a message
        # For now, let's try sending to LLM, it should say "context does not contain..."
        formatted_context_for_llm = "No relevant context passages were found in the database for this query."
    else:
        print(f"Retrieved {len(retrieved_chunks['documents'][0])} chunks.")
        formatted_context_for_llm = format_chunks_for_prompt(retrieved_chunks)
    
    print("\n--- Formatted Context for LLM ---")
    print(formatted_context_for_llm) # Now uncommented for debugging
    
    llm_answer = generate_answer_with_llm(user_query, formatted_context_for_llm)
    
    return llm_answer

# Example usage (for testing directly within this file)
if __name__ == '__main__':
    if not collection:
        print("Exiting: ChromaDB collection not loaded.")
    elif not GEMINI_API_KEY:
        print("Exiting: Gemini API Key not available.")
    else:
        # Example queries from the PDF
        test_query_1 = "How do recent AAO decisions evaluate an applicant's Participation as a Judge service criteria?"
        test_query_2 = "What characteristics of national or international awards persuade the AAO that they constitute 'sustained acclaim'?"
        
        print("\n--- Testing RAG Pipeline with Query 1 ---")
        answer1 = answer_query(test_query_1)
        print("\n--- Final Answer for Query 1 ---")
        print(answer1)

        print("\n\n--- Testing RAG Pipeline with Query 2 ---")
        answer2 = answer_query(test_query_2)
        print("\n--- Final Answer for Query 2 ---")
        print(answer2)