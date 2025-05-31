# src/main.py

import sys
import os

# Assuming execution with `python -m src.main` from the project root.
from src.rag_enhanced import EnhancedRAGPipeline # Import the enhanced pipeline
from src.config import GEMINI_API_KEY

# Global instance of the RAG pipeline
# Initialize with use_cache=True as per your instruction
rag_pipeline_instance = EnhancedRAGPipeline(use_cache=True)

def main_cli():
    """
    Main command-line interface for the Legal RAG System.
    """
    print("\nWelcome to the Legal RAG System for USCIS AAO Decisions (Enhanced).")
    print("===================================================================")

    # Perform initial checks
    if not rag_pipeline_instance.collection:
        print("\nError: The vector database collection could not be loaded by the RAG pipeline.")
        print("Please ensure you have run 'python -m src.store' successfully to create and populate it.")
        return

    if collection_is_empty():
        print("\nWarning: The vector database collection 'aao_decisions' appears to be empty.")
        print("Please ensure you have run 'python -m src.store' successfully to populate it with data.")
        print("The RAG system may not provide useful answers without data.")

    if not GEMINI_API_KEY:
        print("\nWarning: GEMINI_API_KEY is not configured.")
        print("The system will attempt to retrieve context but cannot generate answers from the LLM.")

    print("\nType your legal query regarding I-140 Extraordinary Ability AAO decisions.")
    print("Type 'quit' or 'exit' to stop.")
    print("--------------------------------------------------------")

    while True:
        user_query = input("\nEnter your query: ")

        if user_query.lower() in ["quit", "exit"]:
            break
        
        if not user_query.strip():
            print("Please enter a query.")
            continue

        print(f"\nProcessing your query: \"{user_query}\"")
        print("Please wait, this may take a moment...\n")
        
        try:
            # Use the answer_query method from the EnhancedRAGPipeline instance
            response_data = rag_pipeline_instance.answer_query(user_query)
            
            print("\n--- Answer ---")
            print(response_data.get('answer', "No answer generated."))
            print("--------------")
            
            # Optionally print more details from response_data for debugging/info
            # print(f"\nQuery Intent: {response_data.get('query_intent')}")
            # print(f"Retrieval Time: {response_data.get('performance_metrics', {}).get('retrieval_time'):.4f}s")
            # print(f"Generation Time: {response_data.get('performance_metrics', {}).get('generation_time'):.4f}s")
            # print(f"Used Cache: {response_data.get('performance_metrics', {}).get('used_cache')}")

        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            import traceback
            traceback.print_exc() # For more detailed error information
            print("Please try again or check the logs if the issue persists.")

        print("\n--------------------------------------------------------")

    print("\nThank you for using the Legal RAG System. Goodbye!")

def collection_is_empty():
    """Checks if the ChromaDB collection used by the RAG pipeline is empty."""
    if rag_pipeline_instance and rag_pipeline_instance.collection:
        try:
            return rag_pipeline_instance.collection.count() == 0
        except Exception as e:
            print(f"Error checking collection count: {e}")
            return True 
    return True

if __name__ == "__main__":
    main_cli()