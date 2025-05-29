# src/config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
# It's good practice to call load_dotenv() as early as possible.
# It will search for a .env file in the current directory or parent directories.
# If your .env file is in the project root (where you run python src/main.py from),
# this should find it.

# Determine the path to the .env file (assuming it's in the project root,
# one level above the 'src' directory)
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
else:
    # Fallback if .env is not found where expected (e.g., if running script from src dir directly)
    # This will load .env if it's in the current working directory of the script.
    load_dotenv() 


# Retrieve the API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in environment variables or .env file.")
    print("Please ensure your .env file is in the project root and contains GEMINI_API_KEY='your_key'")
    # You might want to raise an error or exit if the API key is crucial for all operations
    # For now, we'll just print a warning.

# You can add other configurations here if needed, e.g.,
# DEFAULT_MODEL_NAME = "gemini-1.5-flash-latest" (or whatever the exact model ID is)
# VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_db') # Already in store.py
# COLLECTION_NAME = "aao_decisions" # Already in store.py