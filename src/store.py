# src/store.py

import os
import json
import chromadb
from chromadb.utils import embedding_functions # For newer Chroma versions
# If using an older version of chromadb, sentence_transformer_ef might be directly under chromadb.
# from chromadb import embedding_functions # Check your chromadb version if import fails
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid # For generating unique chunk IDs

# --- Configuration ---
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_db') # Directory to store ChromaDB
COLLECTION_NAME = "aao_decisions"

# Embedding model
# Using the one specified in our design: BAAI/bge-small-en-v1.5
# ChromaDB's SentenceTransformerEmbeddingFunction handles the model loading.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Text splitting parameters - UPDATED AS PER YOUR NOTE
CHUNK_SIZE = 1250  # Characters, not tokens.
CHUNK_OVERLAP = 125 # Characters of overlap between chunks.

# --- ChromaDB Setup ---
# Create a persistent client. Data will be stored in the VECTOR_DB_PATH directory.
client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

# Use the embedding function provided by ChromaDB that wraps sentence-transformers
try:
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
except AttributeError: # Fallback for older chromadb versions or different import structure
    try:
        from chromadb.embedding_functions import SentenceTransformerEmbeddingFunction
        sentence_transformer_ef = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
    except ImportError:
        print("Error: Could not import SentenceTransformerEmbeddingFunction from chromadb or chromadb.utils.")
        print("Please ensure chromadb and sentence-transformers are installed correctly.")
        exit()


collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=sentence_transformer_ef
)

# --- Text Splitter Setup ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    add_start_index=True,
)

# --- Main Script Logic ---
if __name__ == "__main__":
    print(f"Starting storage process: Loading JSON files from {PROCESSED_DATA_DIR}")
    print(f"Storing vectors in ChromaDB collection '{COLLECTION_NAME}' at {VECTOR_DB_PATH}")
    print(f"Using embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"Text chunk size: {CHUNK_SIZE} chars, Overlap: {CHUNK_OVERLAP} chars")

    json_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith(".json")]

    if not json_files:
        print("No processed JSON files found to store.")
    else:
        print(f"Found {len(json_files)} JSON files to process and store.")

    total_chunks_added = 0
    for i, json_filename in enumerate(json_files):
        print(f"\n--- Processing JSON file {i+1}/{len(json_files)}: {json_filename} ---")
        json_filepath = os.path.join(PROCESSED_DATA_DIR, json_filename)

        try:
            with open(json_filepath, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON file {json_filename}: {e}. Skipping.")
            continue

        doc_id = doc_data.get("document_id", json_filename.replace(".json", ""))
        text_to_chunk = doc_data.get("cleaned_text", "")
        
        doc_case_name = doc_data.get("case_name", "Unknown Case")
        doc_publication_date = doc_data.get("publication_date_on_website", "Unknown Date")
        doc_source_url = doc_data.get("source_url", "Unknown Source URL")

        if not text_to_chunk:
            print(f"No 'cleaned_text' found in {json_filename}. Skipping.")
            continue

        print(f"Chunking document: {doc_id} (Length: {len(text_to_chunk)} chars)")
        chunks_with_metadata = text_splitter.create_documents(
            [text_to_chunk],
            metadatas=[{"document_id": doc_id, 
                        "case_name": doc_case_name,
                        "publication_date": doc_publication_date,
                        "source_url": doc_source_url 
                       }] 
        )
        
        if not chunks_with_metadata:
            print(f"Text splitting resulted in no chunks for {doc_id}. Skipping.")
            continue
            
        print(f"Generated {len(chunks_with_metadata)} chunks for {doc_id}.")

        batch_ids = []
        batch_embeddings = []
        batch_documents = []
        batch_metadatas = []

        for chunk_obj in chunks_with_metadata:
            chunk_text = chunk_obj.page_content
            chunk_metadata = chunk_obj.metadata

            chunk_id = f"{doc_id}_chunk_{uuid.uuid4().hex[:8]}"
            
            batch_ids.append(chunk_id)
            batch_documents.append(chunk_text)
            
            current_chunk_metadata = chunk_metadata.copy()
            current_chunk_metadata["chunk_id"] = chunk_id
            batch_metadatas.append(current_chunk_metadata)
        
        print(f"Generating embeddings for {len(batch_documents)} chunks...")
        try:
            batch_embeddings = sentence_transformer_ef(batch_documents) 
            print(f"Embeddings generated successfully.")
        except Exception as e:
            print(f"Error generating embeddings for document {doc_id}: {e}")
            continue

        if batch_ids and batch_documents and batch_metadatas and batch_embeddings:
            try:
                print(f"Adding {len(batch_ids)} chunks to ChromaDB for document {doc_id}...")
                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas
                )
                total_chunks_added += len(batch_ids)
                print(f"Successfully added {len(batch_ids)} chunks for {doc_id}.")
            except Exception as e:
                print(f"Error adding chunks to ChromaDB for document {doc_id}: {e}")
        else:
            print(f"No valid data to add to ChromaDB for {doc_id} after processing chunks.")

    print(f"\n--- Storage process complete. ---")
    print(f"Total documents processed: {len(json_files)}")
    print(f"Total chunks added to ChromaDB collection '{COLLECTION_NAME}': {total_chunks_added}")
    print(f"Number of items in collection '{COLLECTION_NAME}': {collection.count()}")