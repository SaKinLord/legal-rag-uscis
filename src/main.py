# src/main.py

import sys
import os

# Ensure the 'src' directory is treated as a package by adding its parent to sys.path
# This allows 'from src.rag import answer_query' to work correctly when running
# 'python src/main.py' from the project root, or 'python main.py' if main.py is moved to root.
# However, the standard way for a project structured like this is to run from root using -m
# e.g., python -m src.main
# If you are running `python src/main.py` directly from the project root, this path adjustment
# might not be strictly necessary if Python's default behavior correctly identifies 'src'
# as being on the path. But it's safer for different execution contexts.

# If you intend to run `python src/main.py` from the project root:
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# This line above is generally NOT needed if you run with `python -m src.main` from root.
# Let's assume we will use `python -m src.main` for execution.

from src.rag import answer_query, collection as rag_collection # Import the collection for a check
from src.config import GEMINI_API_KEY

def main_cli():
    """
    Main command-line interface for the Legal RAG system.
    """
    print("\nWelcome to the Legal RAG System for USCIS AAO Decisions.")
    print("========================================================")

    # Perform initial checks
    if not rag_collection:
        print("\nError: The vector database collection could not be loaded.")
        print("Please ensure you have run 'python -m src.store' successfully to create and populate it.")
        return

    if collection_is_empty():
        print("\nWarning: The vector database collection 'aao_decisions' appears to be empty.")
        print("Please ensure you have run 'python -m src.store' successfully to populate it with data.")
        print("The RAG system may not provide useful answers without data.")
        # We can still proceed, but the user should be warned.

    if not GEMINI_API_KEY:
        print("\nWarning: GEMINI_API_KEY is not configured.")
        print("The system will attempt to retrieve context but cannot generate answers from the LLM.")
        # Proceeding without LLM is not very useful for RAG, but let's allow it for context retrieval check.

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
            answer = answer_query(user_query)
            print("\n--- Answer ---")
            print(answer)
            print("--------------")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            print("Please try again or check the logs if the issue persists.")

        print("\n--------------------------------------------------------")

    print("\nThank you for using the Legal RAG System. Goodbye!")

def collection_is_empty():
    """Checks if the ChromaDB collection is empty."""
    if rag_collection:
        try:
            return rag_collection.count() == 0
        except Exception as e:
            print(f"Error checking collection count: {e}")
            return True # Assume empty or problematic if count fails
    return True # Assume empty if collection object itself is None

if __name__ == "__main__":
    main_cli()