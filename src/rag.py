# src/rag.py

import os
import chromadb
from chromadb.utils import embedding_functions # Or from chromadb import embedding_functions
from src.config import GEMINI_API_KEY, CLAUDE_API_KEY # Import API keys
from src.query_enhancement import enhance_retrieval_query
from src.llm_client import get_llm_client

# --- Configuration ---
VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_db')
COLLECTION_NAME = "aao_decisions"
EMBEDDING_MODEL_NAME_FOR_QUERY = "BAAI/bge-small-en-v1.5" # Must match model used for storing
LLM_MODEL_NAME = "gemini-1.5-flash-latest" # Or "gemini-1.0-pro" or other compatible model

# Number of relevant chunks to retrieve - optimized for legal documents
TOP_K_CHUNKS = 5 # Reduced to focus on most relevant chunks, avoiding noise

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

# --- Initialize LLM Client ---
llm_client = get_llm_client()
if not llm_client.is_available():
    print("Warning: No LLM API keys configured. Answer generation will not be available.")
    print("Please set CLAUDE_API_KEY or GEMINI_API_KEY in your .env file")

# --- Core RAG Functions ---

def retrieve_relevant_chunks(query_text, n_results=TOP_K_CHUNKS):
    """
    Embeds the query and retrieves the top_k relevant chunks from ChromaDB
    with improved deduplication and relevance filtering.
    """
    if not collection:
        print("ChromaDB collection is not available. Cannot retrieve chunks.")
        return []
    if not query_text:
        print("Query text is empty. Cannot retrieve chunks.")
        return []

    try:
        # Retrieve more candidates initially to allow for deduplication and better matching
        retrieve_candidates = min(n_results * 5, 100)  # Increased from 3x to 5x
        
        results = collection.query(
            query_texts=[query_text],
            n_results=retrieve_candidates,
            include=['documents', 'metadatas', 'distances'],
            # Add metadata filtering for better relevance if needed
            # where={"document_type": "legal_decision"}  # Uncomment if you have this metadata
        )
        
        # Deduplicate by document_id while preserving best chunks per document
        if results and results.get('documents') and results['documents'][0]:
            doc_chunks = {}  # document_id -> (best_distance, chunk_data)
            
            for i, (doc, meta, dist) in enumerate(zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            )):
                doc_id = meta.get('document_id', f'unknown_{i}')
                
                # Keep the best (lowest distance) chunk per document
                if doc_id not in doc_chunks or dist < doc_chunks[doc_id][0]:
                    doc_chunks[doc_id] = (dist, {'doc': doc, 'meta': meta, 'dist': dist})
            
            # Enhanced ranking: combine distance with text relevance
            def calculate_relevance_score(chunk_data):
                distance = chunk_data[0]
                text = chunk_data[1]['doc'].lower()
                query_lower = query_text.lower()
                
                # Boost score for exact query term matches
                term_matches = sum(1 for word in query_lower.split() 
                                 if len(word) > 3 and word in text)
                
                # Legal-specific term boost with more comprehensive matching
                legal_terms = ['judge', 'award', 'criteria', 'evaluation', 'extraordinary', 'sustained', 'acclaim', 
                              'participation', 'judging', 'national', 'international', 'recognition', 'excellence']
                legal_matches = sum(1 for term in legal_terms if term in query_lower and term in text)
                
                # Phrase matching bonus
                key_phrases = []
                if 'judge' in query_lower:
                    key_phrases.extend(['participation as judge', 'judging role', 'peer review', 'evaluation panel', 
                                      'consistent history of reviewing', 'judging recognized', 'level of candidates'])
                if 'award' in query_lower:
                    key_phrases.extend(['national award', 'international award', 'sustained acclaim', 'recognition',
                                      'nationally recognized', 'internationally recognized'])
                
                phrase_matches = sum(1 for phrase in key_phrases if phrase in text)
                
                # Professional field context bonus - prefer documents that discuss evaluation methods
                analysis_terms = ['does not demonstrate', 'evidence presented', 'record does not', 'establish', 
                                'evaluation', 'criteria judges should use', 'consistent history', 'setting her apart']
                analysis_matches = sum(1 for term in analysis_terms if term in text)
                
                # Combined score (lower is better for distance, higher is better for matches)
                relevance_boost = (term_matches * 0.08) + (legal_matches * 0.04) + (phrase_matches * 0.12) + (analysis_matches * 0.06)
                adjusted_score = distance - relevance_boost
                
                return adjusted_score
            
            # Sort by enhanced relevance score and take top n_results
            enhanced_chunks = [(calculate_relevance_score(chunk_data), chunk_data) for chunk_data in doc_chunks.values()]
            sorted_chunks = sorted(enhanced_chunks, key=lambda x: x[0])[:n_results]
            sorted_chunks = [item[1] for item in sorted_chunks]  # Extract chunk_data
            
            # Reconstruct results format
            results = {
                'documents': [[item[1]['doc'] for item in sorted_chunks]],
                'metadatas': [[item[1]['meta'] for item in sorted_chunks]],
                'distances': [[item[1]['dist'] for item in sorted_chunks]]
            }
        
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
    Sends the query and context to the LLM using unified client and gets an answer.
    """
    if not llm_client.is_available():
        return "LLM generation is not available (no API keys configured)."
    if not query_text:
        return "Query is empty."

    prompt_template = f"""You are an expert legal analyst specializing in USCIS AAO I-140 Extraordinary Ability decisions. Answer the user's question using ONLY the provided context passages.

**INSTRUCTIONS:**
1. Use ONLY information from the context passages - no external knowledge
2. Synthesize information across ALL relevant passages to provide comprehensive answers
3. For legal criteria or requirements, be specific about standards and evaluation methods
4. Cite sources using this format: [Context N] where N is the context number
5. If insufficient information exists, state: "Insufficient information in provided context"

**CONTEXT PASSAGES:**

{formatted_context}

**USER QUESTION:** {query_text}

**ANSWER (be thorough but concise, focus on practical legal standards):**
"""

    try:
        print(f"\n--- Sending prompt to LLM ({llm_client.primary_client})... ---")
        
        response = llm_client.generate_content(prompt_template)
        
        print("--- LLM response received. ---")
        return response.strip()

    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return f"Error generating answer from LLM: {e}"


def answer_query(user_query):
    """
    Full RAG pipeline with enhanced query processing: enhance query, retrieve, format context, generate answer.
    """
    print(f"Received user query: {user_query}")
    
    # Enhance query for better retrieval
    query_info = enhance_retrieval_query(user_query)
    enhanced_query = query_info['enhanced_query']
    print(f"Enhanced query: {enhanced_query}")
    
    # Try retrieval with enhanced query first, fallback to original if needed
    retrieved_chunks = retrieve_relevant_chunks(enhanced_query)
    
    # If enhanced query doesn't return good results, try original
    if not retrieved_chunks or not retrieved_chunks.get('documents') or len(retrieved_chunks['documents'][0]) < 3:
        print("Enhanced query returned few results, trying original query...")
        retrieved_chunks = retrieve_relevant_chunks(user_query)
    
    if not retrieved_chunks or not retrieved_chunks.get('documents') or not retrieved_chunks['documents'][0]:
        print("No relevant chunks found for the query.")
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
    elif not llm_client.is_available():
        print("Exiting: No LLM API keys available.")
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