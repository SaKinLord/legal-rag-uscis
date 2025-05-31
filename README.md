# Legal RAG System for USCIS AAO Decisions

## 1. About The Project

This project implements a Retrieval-Augmented Generation (RAG) system designed to answer legal queries based on a specific set of USCIS Administrative Appeals Office (AAO) non-precedent decisions. Specifically, it targets I-140 Extraordinary Ability petitions published in February 2025.

The system performs the following key functions:
*   **Data Acquisition**: Programmatically identifies and downloads relevant AAO decisions from the USCIS website.
*   **Processing & Structuring**: Extracts text and metadata (case name, decision date, headings) from the downloaded PDF documents and structures it into a standardized JSON format.
*   **Storage**: Chunks the processed text, generates vector embeddings, and stores these along with metadata in a ChromaDB vector database for efficient similarity search.
*   **Retrieval & Generation**: Takes a user's legal query, retrieves the most relevant text chunks from the database, and uses Google's Gemini Flash LLM to generate a concise, cited answer based solely on the provided context.

This project was developed as an AI Internship Homework assignment, focusing on the design, thought process, and implementation of a legal-focused RAG pipeline.

## 2. Setup Instructions

### Prerequisites
*   Python 3.9 or higher
*   Access to a terminal or command prompt

### Steps

1.  **Clone the Repository (or Create Project Directory):**
    If this were a Git repository:
    ```bash
    git clone <repository_url>
    cd legal-rag-uscis-internship
    ```
    Otherwise, ensure you have the project files in a root directory (e.g., `legal_rag_uscis/`).

2.  **Create a Python Virtual Environment:**
    Navigate to the project root directory (`legal_rag_uscis`) and run:
    ```bash
    python -m venv .venv
    ```
    Activate the virtual environment:
    *   Windows:
        ```bash
        .\.venv\Scripts\activate
        ```
    *   macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```

3.  **Install Requirements:**
    With the virtual environment activated, install the necessary Python libraries:
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure `requirements.txt` is populated, ideally by running `pip freeze > requirements.txt` in your final working environment).*

4.  **Configure Environment Variables (API Key):**
    *   In the project root directory, create a file named `.env`.
    *   Add your Google Gemini API key to this file in the following format:
        ```env
        GEMINI_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY"
        ```
    *   A `.env.example` file is provided as a template.
    *   **Important**: Ensure the `.env` file is listed in your `.gitignore` file (if using Git) to prevent committing your API key.

## 3. Order of Execution for Scripts

The scripts should be run from the **project root directory** (`legal_rag_uscis/`) using the `python -m src.<script_name>` module execution pattern.

1.  **Acquire Data (`acquire.py`):**
    Downloads the PDF decision documents from the USCIS website.
    ```bash
    python -m src.acquire
    ```
    This will populate the `data/raw/` directory with PDF files.

2.  **Process Documents (`process.py`):**
    Extracts text and metadata from the downloaded PDFs and saves them as JSON files.
    ```bash
    python -m src.process
    ```
    This will populate the `data/processed/` directory with JSON files.

3.  **Store Data in Vector DB (`store.py`):**
    Chunks the processed text, generates embeddings, and loads everything into ChromaDB.
    ```bash
    python -m src.store
    ```
    This will create a `vector_db/` directory in the project root containing the ChromaDB data. *If you re-run this script, it's recommended to delete the `vector_db/` directory first to ensure a clean database, as the current script version might add duplicate entries if run multiple times on the same processed data.*

4.  **Run the RAG System CLI (`main.py`):**
    Starts the command-line interface to ask questions.
    ```bash
    python -m src.main
    ```

## 4. How to Use `main.py` (CLI)

Once the previous scripts (`acquire`, `process`, `store`) have been run successfully:

1.  Execute `python -m src.main` from the project root directory.
2.  You will see a welcome message and then be prompted to:
    ```
    Enter your query:
    ```
3.  Type your legal question related to the I-140 Extraordinary Ability AAO decisions from February 2025 and press Enter.
    *Example Queries (from assignment PDF):*
    *   `How do recent AAO decisions evaluate an applicant's Participation as a Judge service criteria?`
    *   `What characteristics of national or international awards persuade the AAO that they constitute 'sustained acclaim'?`
4.  The system will process your query, retrieve relevant context, and generate an answer with citations.
5.  After the answer is displayed, you will be prompted for another query.
6.  To exit the application, type `quit` or `exit` at the query prompt and press Enter.
