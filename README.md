# Enhanced Legal RAG System for USCIS AAO Decisions

## 1. About The Project

This project implements an **Enhanced Retrieval-Augmented Generation (RAG)** system designed to answer legal queries based on a specific set of USCIS Administrative Appeals Office (AAO) non-precedent decisions. It targets I-140 Extraordinary Ability petitions published in February 2025 (using this as a structural example, actual data acquisition may use available dates).

The system has been significantly enhanced from a basic RAG pipeline and now includes:

*   **Data Acquisition**: Programmatically identifies and downloads relevant AAO decisions.
*   **Processing & Structuring**: Extracts text and metadata (case identifier, decision date, headings) from PDFs into standardized JSON.
*   **Optimized Storage**: Chunks processed text using parameters informed by a dedicated **Chunking Optimizer** (`chunking_optimizer.py`), generates vector embeddings (`BAAI/bge-small-en-v1.5`), and stores them in ChromaDB.
*   **Advanced RAG Pipeline (`rag_enhanced.py`):**
    *   **Query Preprocessing (`query_preprocessor.py`):** Normalizes queries, expands terms, identifies legal entities, determines query intent, and generates multiple weighted query variations.
    *   **Multi-Query Retrieval:** Retrieves relevant chunks using multiple query variations for improved recall and relevance.
    *   **Intent-Based Prompting:** Tailors prompts to the LLM (Google Gemini Flash) based on the identified query intent.
    *   **Precise Passage-Level Citations:** Generates answers with citations linking back to the specific document and chunk.
*   **Caching (`cache_manager.py`):** Implements a hybrid caching system (memory, disk, with Redis support if available) for embeddings, query results, and LLM answers to improve performance.
*   **Evaluation Framework (`evaluation_metrics.py`):** Provides tools to quantitatively assess retrieval and generation quality.
*   **Testing Suite (`tests/`):** Includes various scripts to test components, run benchmarks, and perform analyses.

This project was developed as an AI Internship Homework assignment, with a focus on demonstrating a deep understanding of RAG architecture, component design, iterative refinement, and addressing technical challenges in building a sophisticated legal AI assistant.

## 2. Setup Instructions

### Prerequisites
*   Python 3.9 or higher
*   Git (for cloning, if applicable)
*   Access to a terminal or command prompt
*   (Optional) Redis server installed and running if you want to leverage Redis for caching (the system will fallback to memory/disk cache if Redis is unavailable).

### Steps

1.  **Clone the Repository (or Unzip Project Files):**
    If this were a Git repository:
    ```bash
    git clone <repository_url>
    cd legal_rag_uscis_enhanced # Or your project directory name
    ```
    Otherwise, extract your project files to a root directory.

2.  **Create and Activate a Python Virtual Environment:**
    Navigate to the project root directory and run:
    ```bash
    python -m venv .venv
    ```
    Activate:
    *   Windows: `.\.venv\Scripts\activate`
    *   macOS/Linux: `source .venv/bin/activate`

3.  **Install Requirements:**
    With the virtual environment activated:
    ```bash
    pip install -r requirements.txt
    ```
    *(This file should contain all necessary libraries including `google-generativeai`, `chromadb`, `sentence-transformers`, `langchain`, `pdfplumber`, `python-dotenv`, `requests`, `beautifulsoup4`, `lxml`, `spacy`, `nltk`, `lz4`, `redis` (if using), `matplotlib`, `seaborn`, `pandas`, `rouge-score`, etc.)*
    After installation, ensure spaCy English model and NLTK data are available (the scripts attempt to download them on first run if missing):
    ```bash
    python -m spacy download en_core_web_sm
    # NLTK resources like 'punkt', 'wordnet', 'averaged_perceptron_tagger_eng' are checked by query_preprocessor.py
    ```


4.  **Configure Environment Variables (API Key):**
    *   In the project root directory, create a file named `.env`.
    *   Add your Google Gemini API key:
        ```env
        GEMINI_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY"
        ```
    *   Refer to `.env.example` for the template.
    *   **Important**: Ensure `.env` is in your `.gitignore`.

## 3. Order of Execution for Core RAG Pipeline

Run these scripts from the **project root directory** using the `python -m src.<script_name>` pattern.

1.  **Acquire Data (`acquire.py`):**
    Downloads PDF decisions.
    ```bash
    python -m src.acquire
    ```
    Populates `data/raw/`.

2.  **Process Documents (`process.py`):**
    Extracts text/metadata from PDFs to JSON.
    ```bash
    python -m src.process
    ```
    Populates `data/processed/`.

3.  **Store Data in Vector DB (`store.py`):**
    Chunks, embeds, and loads data into ChromaDB. Uses chunking parameters defined within the script (e.g., size 1250, overlap 125, or 1000/200 as per your last run).
    ```bash
    python -m src.store
    ```
    Creates/populates `vector_db/`. **Delete `vector_db/` before re-running if you change chunking parameters or want a fresh database.**

4.  **Run the Enhanced RAG System CLI (`main.py`):**
    Starts the command-line interface to ask questions using the enhanced pipeline.
    ```bash
    python -m src.main
    ```

## 4. How to Use `main.py` (CLI)

1.  Ensure steps 1-3 above are completed and the `vector_db/` is populated.
2.  Execute `python -m src.main` from the project root.
3.  Follow the prompts to enter your legal query.
    *Example Queries (from assignment PDF):*
    *   `How do recent AAO decisions evaluate an applicant's Participation as a Judge service criteria?`
    *   `What characteristics of national or international awards persuade the AAO that they constitute 'sustained acclaim'?`
4.  The system will display detailed processing information (if logging is enabled in `rag_enhanced.py`) and then the LLM's answer with citations.
5.  Type `quit` or `exit` to stop.

## 5. Advanced Components & Testing (Optional)

This project includes advanced modules for optimization and evaluation, primarily located in the `src/` and `tests/` directories.

### 5.1. Chunking Optimizer (`src/chunking_optimizer.py`)

This script helps determine optimal chunking parameters.
*   **To Run (example):**
    ```bash
    python -m src.chunking_optimizer
    ```
    (Note: The `if __name__ == "__main__":` block in `chunking_optimizer.py` loads sample data and queries. This may take several minutes.)
*   **Output:** Generates `chunking_optimizer_report.json` and visualizations in `chunking_optimizer_visuals/`. The report suggests an optimal configuration based on defined metrics. The constants `OPTIMAL_CHUNK_SIZE` and `OPTIMAL_CHUNK_OVERLAP` in `rag_enhanced.py` can be updated based on these findings, and then `src/store.py` would need to be modified and re-run to use these new parameters.

### 5.2. Evaluation Framework (`src/evaluation_metrics.py`)

This script provides tools to evaluate the RAG system's performance.
*   **To Run (example):**
    ```bash
    python -m src.evaluation_metrics
    ```
    (Note: The `if __name__ == "__main__":` block uses predefined test queries and requires an API key for LLM-based evaluations. It may make calls to the Gemini API.)
*   **Output:** Generates `evaluation_report.json` and visualizations in `evaluation_results/`.

### 5.3. Test Suite (`tests/`)

The `tests/` directory contains various scripts to test individual components and the integrated system.
*   **To Run All Core Tests:**
    Navigate to the `tests/` directory and run:
    ```bash
    cd tests
    python run_all_tests.py
    cd .. 
    ```
*   This script orchestrates several other test scripts (`test_query_preprocessor.py`, `test_cache_manager.py`, `test_enhanced_rag.py`).
*   Other scripts like `run_comparative_analysis.py` and `run_full_benchmark.py` offer more in-depth performance insights and may take longer to run.

## 6. Project Structure Overview
```
legal_rag_uscis_enhanced/
├── .env                           # Local environment variables (API KEY - NOT COMMITTED)
├── .env.example                   # Example for .env
├── .gitignore                     # Specifies intentionally untracked files
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── cache/                         # Default directory for disk cache (if used by CacheManager)
├── data/
│   ├── raw/                       # Stores downloaded PDF files from acquire.py
│   └── processed/                 # Stores processed JSON files from process.py
├── notebooks/                     # Jupyter notebooks for experimentation (if any)
├── src/                           # Source code
│   ├── __init__.py
│   ├── acquire.py                 # Data acquisition
│   ├── process.py                 # PDF processing and metadata extraction
│   ├── store.py                   # Chunking, embedding, and storing in ChromaDB
│   ├── config.py                  # Configuration (e.g., API key loading)
│   ├── query_preprocessor.py      # Query normalization, expansion, intent
│   ├── cache_manager.py           # Caching logic
│   ├── rag_enhanced.py            # Core enhanced RAG pipeline logic
│   ├── main.py                    # CLI entry point
│   ├── chunking_optimizer.py      # Script to find optimal chunking params
│   └── evaluation_metrics.py      # Script for RAG evaluation
├── tests/                         # Test scripts and test-generated outputs
│   ├── __init__.py
│   ├── run_all_tests.py
│   ├── test_*.py                  # Individual component tests
│   ├── run_*.py                   # Benchmarking/analysis scripts
│   ├── chunking_results/          # Output from chunking optimizer
│   └── evaluation_results/        # Output from evaluation framework
└── vector_db/                     # ChromaDB persistent storage (NOT COMMITTED)
```


## 7. Key Design Aspects of the Enhanced System

*   **Modular Design:** Components like query preprocessing, caching, retrieval, and generation are separated for clarity and maintainability.
*   **Query Understanding:** The `LegalQueryPreprocessor` enhances user queries to improve the relevance of retrieved context.
*   **Intent-Driven Prompts:** Prompts sent to the LLM are tailored based on the identified intent of the user's query.
*   **Caching Strategy:** A hybrid caching approach (memory, disk, with Redis support) is implemented to optimize performance for repeated operations.
*   **Data-Driven Optimization:** The `ChunkingOptimizer` provides a means to empirically determine effective chunking strategies.
*   **Comprehensive Evaluation:** The `RAGEvaluator` allows for detailed metrics on both retrieval and generation quality.

