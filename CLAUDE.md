# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core RAG Pipeline

### Sequential Pipeline Execution
```bash
# 1. Acquire USCIS AAO decision documents
python -m src.acquire

# 2. Process PDFs to structured JSON
python -m src.process

# 3. Create vector embeddings and store in ChromaDB
python -m src.store

# 4. Run the enhanced RAG CLI system
python -m src.main
```

### Python Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows WSL/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install spaCy model
python -m spacy download en_core_web_sm
```

## Architecture Overview

### Core Components

**Enhanced RAG Pipeline (`src/rag_enhanced.py`)**
- Main orchestrator using `EnhancedRAGPipeline` class
- Integrates query preprocessing, hybrid retrieval, and legal specialization
- Supports caching through `RAGCacheManager`
- Uses ChromaDB for vector storage with `BAAI/bge-small-en-v1.5` embeddings

**Legal Query Processing (`src/query_preprocessor.py`)**
- `LegalQueryPreprocessor` handles query normalization and expansion
- Identifies legal entities and determines query intent
- Generates multiple weighted query variations for improved retrieval

**Multi-LLM Support (`src/config.py`)**
- Supports both Claude (Anthropic) and Gemini (Google) APIs
- Automatically defaults to Claude if available, fallback to Gemini
- Environment variables: `CLAUDE_API_KEY`, `GEMINI_API_KEY`

**Specialized Legal Components**
- `src/legal_specialized_ranker.py`: Legal document ranking
- `src/legal_document_analyzer.py`: Document structure analysis  
- `src/legal_concept_matcher.py`: Legal concept identification
- `src/legal_authority_analyzer.py`: Authority and precedent analysis

### Data Pipeline Architecture

1. **Acquisition (`src/acquire.py`)**: Downloads USCIS AAO decisions from web sources
2. **Processing (`src/process.py`)**: Extracts text/metadata from PDFs to JSON format
3. **Storage (`src/store.py`)**: Chunks documents (optimal: 1250 chars, 125 overlap), creates embeddings, stores in ChromaDB
4. **Retrieval**: Multi-stage retrieval with hybrid approaches and legal specialization
5. **Generation**: Intent-based prompting to LLM with precise citations

### Caching Strategy (`src/cache_manager.py`)
- Multi-tier caching: memory → disk → Redis (optional)
- Caches embeddings, query results, and LLM responses
- Configurable TTL and size limits

## Environment Configuration

### Required Environment Variables (.env file)
```bash
# At least one API key required
GEMINI_API_KEY="your_gemini_api_key"
CLAUDE_API_KEY="your_claude_api_key"  # Preferred if available
```

### Important Paths
- Vector database: `vector_db/` (ignored in git)
- Raw PDFs: `data/raw/`
- Processed JSON: `data/processed/`
- Cache directory: `cache/`

## Development Notes

### Chunking Configuration
- Optimal parameters determined through `chunking_optimizer.py`
- Current settings: 1250 character chunks with 125 character overlap
- Stored in `OPTIMAL_CHUNK_SIZE` and `OPTIMAL_CHUNK_OVERLAP` constants in `rag_enhanced.py`

### Testing Structure
- Phase-based testing approach (test_phase1, test_phase2, test_phase3)
- Comprehensive evaluation includes retrieval metrics, generation quality, and error analysis
- Benchmark scripts generate detailed JSON reports and visualizations

### Performance Optimization
- `src/performance_optimizer.py`: General performance tuning
- `parameter_optimizer.py`: System-wide parameter optimization
- Results saved to `parameter_optimization_results.json`

### Vector Database Management
- ChromaDB collection name: "aao_decisions"  
- **Important**: Delete `vector_db/` directory before re-running `src/store.py` if chunking parameters change
- Collection automatically recreated on store operations

### Legal Domain Specialization
- Multiple specialized retrievers and rankers for legal content
- Legal authority analysis for precedent evaluation  
- Query-aware retrieval adapts to different legal query types
- Intent-based prompting optimizes LLM responses for legal context