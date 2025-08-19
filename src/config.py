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


# Retrieve API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# Default to Claude if available, fallback to Gemini
PREFERRED_LLM = "claude" if CLAUDE_API_KEY else "gemini"

if not GEMINI_API_KEY and not CLAUDE_API_KEY:
    print("Warning: Neither GEMINI_API_KEY nor CLAUDE_API_KEY found in environment variables or .env file.")
    print("Please ensure your .env file contains at least one of:")
    print("GEMINI_API_KEY='your_gemini_key'")
    print("CLAUDE_API_KEY='your_claude_key'")
elif CLAUDE_API_KEY:
    print(f"Using Claude API as preferred LLM provider")
elif GEMINI_API_KEY:
    print(f"Using Gemini API as LLM provider")

# Model configurations
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"
CLAUDE_MODEL_NAME = "claude-3-5-sonnet-20241022"  # Latest Claude model with better performance