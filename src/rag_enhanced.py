# src/rag_enhanced.py

import os
import time
from typing import List, Dict, Tuple, Optional
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from src.config import GEMINI_API_KEY
from src.query_preprocessor import LegalQueryPreprocessor
from src.cache_manager import RAGCacheManager
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_db')
COLLECTION_NAME = "aao_decisions"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "gemini-1.5-flash-latest"

# Optimal chunking parameters from experiments
OPTIMAL_CHUNK_SIZE = 1250
OPTIMAL_CHUNK_OVERLAP = 125


class EnhancedRAGPipeline:
    """Enhanced RAG pipeline with query preprocessing and caching."""
    
    def __init__(self, 
                 use_cache: bool = True,
                 cache_config: Optional[Dict] = None):
        # Initialize components
        self.query_preprocessor = LegalQueryPreprocessor()
        
        # Initialize cache
        if use_cache:
            cache_config = cache_config or {
                'cache_type': 'hybrid',
                'max_memory_size_mb': 500,
                'max_disk_size_mb': 2000,
                'eviction_policy': 'lru',
                'compression': True
            }
            self.cache_manager = RAGCacheManager(**cache_config)
        else:
            self.cache_manager = None
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        
        try:
            self.collection = self.client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_function
            )
        except Exception as e:
            logger.error(f"Error loading collection: {e}")
            self.collection = None
        
        # Initialize Gemini
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            self.llm_model = genai.GenerativeModel(LLM_MODEL_NAME)
        else:
            logger.warning("Gemini API key not configured")
            self.llm_model = None
    
    def answer_query(self, 
                    user_query: str,
                    n_results: int = 8,
                    use_query_expansion: bool = True) -> Dict:
        """
        Enhanced query answering with preprocessing and caching.
        """
        start_time = time.time()
        
        # Step 1: Preprocess query
        logger.info(f"Processing query: {user_query}")
        preprocessed = self.query_preprocessor.preprocess_query(user_query)
        
        # Step 2: Check cache for query results
        cached_results = None
        if self.cache_manager:
            cached_results = self.cache_manager.get_query_results(user_query)
        
        # Step 3: Retrieve relevant chunks
        if cached_results:
            logger.info("Using cached retrieval results")
            retrieval_results = cached_results
            retrieval_time = 0
        else:
            retrieval_start = time.time()
            
            if use_query_expansion and preprocessed['query_variations']:
                # Use multiple query variations for better retrieval
                retrieval_results = self._multi_query_retrieval(
                    queries=preprocessed['search_queries'],
                    n_results=n_results
                )
            else:
                # Single query retrieval
                retrieval_results = self.collection.query(
                    query_texts=[preprocessed['normalized_query']],
                    n_results=n_results,
                    include=['documents', 'metadatas', 'distances']
                )
            
            retrieval_time = time.time() - retrieval_start
            
            # Cache retrieval results
            if self.cache_manager:
                self.cache_manager.cache_query_results(user_query, retrieval_results)
        
        # Step 4: Format context for LLM
        formatted_context = self._format_retrieval_results(retrieval_results)

        # --- START: ADDED TRANSPARENCY LOGGING (like Candidate B) ---
        print(f"Retrieved {len(formatted_context)} chunks for context.") # Mimics Candidate B's "Retrieved X chunks"
        if formatted_context:
            print("\n--- Retrieved Context Details ---")
            for i, ctx_item in enumerate(formatted_context):
                meta = ctx_item.get('metadata', {})
                # The 'document_id' from store.py is the full filename like "February_03__2025_FEB032025_01B2203"
                # The 'chunk_id' from store.py is like "February_03__2025_FEB032025_01B2203_chunk_uuid"
                print(f"CONTEXT {i+1}:") # Simplified header for each chunk
                print(f"  Source Document ID (Original Filename based): {meta.get('document_id', 'N/A')}")
                print(f"  Chunk ID (Generated in store.py): {meta.get('chunk_id', 'N/A')}")
                print(f"  Case Name: {meta.get('case_name', 'N/A')}")
                # 'publication_date' in metadata is 'publication_date_on_website' from process.py
                print(f"  Publication Date: {meta.get('publication_date', 'N/A')}")
                print(f"  Relevance Score (1-distance): {ctx_item.get('relevance_score', 0.0):.4f}")
                # Passage text will be shown in the next logging block
            print("--- End Retrieved Context Details ---\n")
        # --- END: ADDED TRANSPARENCY LOGGING ---
        
        # Step 5: Check cache for answer
        cached_answer = None
        if self.cache_manager and formatted_context:
            context_texts = [doc['text'] for doc in formatted_context]
            cached_answer = self.cache_manager.get_answer(user_query, context_texts)
        
        # Step 6: Generate answer
        if cached_answer:
            logger.info("Using cached answer")
            llm_answer = cached_answer
            generation_time = 0
        else:
            generation_start = time.time()
            llm_answer = self._generate_answer_with_llm(
                query=preprocessed['normalized_query'],
                context=formatted_context, # This list of dicts is passed
                query_intent=preprocessed['query_intent']
            )
            generation_time = time.time() - generation_start
            
            # Cache answer
            if self.cache_manager and formatted_context and llm_answer:
                context_texts = [doc['text'] for doc in formatted_context]
                self.cache_manager.cache_answer(user_query, context_texts, llm_answer)
        
        total_time = time.time() - start_time
        
        # Step 7: Prepare response
        response = {
            'query': user_query,
            'preprocessed_query': preprocessed['normalized_query'],
            'query_intent': preprocessed['query_intent'],
            'answer': llm_answer,
            'retrieved_documents': formatted_context, # This already contains detailed chunk info
            'key_concepts': preprocessed['key_concepts'],
            'entities': preprocessed['entities'],
            'performance_metrics': {
                'total_time': total_time,
                'retrieval_time': retrieval_time,
                'generation_time': generation_time,
                'used_cache': cached_results is not None or cached_answer is not None
            }
        }
        
        return response
    
    def _multi_query_retrieval(self, 
                             queries: List[Tuple[str, float]], 
                             n_results: int) -> Dict:
        """
        Retrieve using multiple weighted queries and merge results.
        """
        all_results = {
            'ids': [[]],
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        # Track unique documents and their best scores
        doc_scores = {}
        doc_data = {}
        
        for query_text, weight in queries[:3]:  # Limit to top 3 queries
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results * 2,  # Get more results to merge
                include=['documents', 'metadatas', 'distances']
            )
            
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    # Weight the distance by query weight
                    weighted_distance = results['distances'][0][i] * (2 - weight)
                    
                    if doc_id not in doc_scores or weighted_distance < doc_scores[doc_id]:
                        doc_scores[doc_id] = weighted_distance
                        doc_data[doc_id] = {
                            'document': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i]
                        }
        
        # Sort by score and take top n_results
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1])[:n_results]
        
        # Format results
        for doc_id, distance in sorted_docs:
            all_results['ids'][0].append(doc_id)
            all_results['distances'][0].append(distance)
            all_results['documents'][0].append(doc_data[doc_id]['document'])
            all_results['metadatas'][0].append(doc_data[doc_id]['metadata'])
        
        return all_results
    
    def _format_retrieval_results(self, retrieval_results: Dict) -> List[Dict]:
        """Format retrieval results for context."""
        if not retrieval_results or not retrieval_results.get('documents') or not retrieval_results['documents'][0]:
            return []
        
        formatted = []
        # Ensure we handle the case where retrieval_results might be structured differently
        # or some keys might be missing, especially 'distances'.
        docs = retrieval_results['documents'][0]
        metas = retrieval_results.get('metadatas', [[]])[0] # Default to empty list if 'metadatas' or its first element is missing
        dists = retrieval_results.get('distances', [[]])[0] # Default to empty list for distances
        
        for i, doc_text in enumerate(docs):
            # Ensure metadata and distance are available for the current index
            current_meta = metas[i] if i < len(metas) else {}
            current_dist = dists[i] if i < len(dists) else None # Use None if distance is missing

            formatted_doc = {
                'text': doc_text,
                'metadata': current_meta,
                'relevance_score': (1 - current_dist) if current_dist is not None else 0.0 # Calculate score if dist is available
            }
            formatted.append(formatted_doc)
        
        return formatted
    
    def _generate_answer_with_llm(self, 
                                query: str, 
                                context: List[Dict], # This is the formatted_context list of dicts
                                query_intent: str) -> str:
        """Generate answer using LLM with intent-aware prompting."""
        if not self.llm_model or not context:
            # --- MODIFICATION: Log if no context ---
            if not context:
                print("\n--- Formatted Context for LLM ---")
                print("No context passages to provide to the LLM.")
                print("--- End Formatted Context for LLM ---\n")
            return "Unable to generate answer. Please check system configuration or context availability."
        
        # Build context string with proper citations
        context_str = self._build_context_string(context) # This method now also logs the context_str

        # --- START: ADDED TRANSPARENCY LOGGING (like Candidate B) ---
        # This log shows the exact string being sent to the LLM as context.
        logger.info("Formatted context string prepared for LLM.")
        print("\n--- Formatted Context for LLM ---")
        print(context_str) # context_str is built by _build_context_string
        print("--- End Formatted Context for LLM ---\n")
        # --- END: ADDED TRANSPARENCY LOGGING ---
        
        # Select prompt template based on query intent
        prompt = self._get_intent_based_prompt(query_intent, query, context_str)
        
        try:
            logger.info("Generating answer with LLM")
            # --- MODIFICATION: Log which LLM is being used ---
            print(f"--- Sending prompt to LLM ({LLM_MODEL_NAME})... ---")
            response = self.llm_model.generate_content(prompt)
            print("--- LLM response received. ---")
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    
    def _build_context_string(self, context: List[Dict]) -> str:
        context_parts = []
    
        for i, retrieved_chunk_info in enumerate(context):
            meta = retrieved_chunk_info.get('metadata', {})
            text_content = retrieved_chunk_info.get('text', '')
            relevance = retrieved_chunk_info.get('relevance_score', 0.0)
    
            original_doc_id = meta.get('document_id', 'N/A') 
            citation_doc_id_to_use_in_prompt = original_doc_id 
    
            id_pattern_match = re.search(r'([A-Z]{3}\d{2}\d{4}_\w+)', original_doc_id)
            
            if id_pattern_match:
                citation_doc_id_to_use_in_prompt = id_pattern_match.group(1)
            elif original_doc_id != 'N/A': 
                parts = original_doc_id.split('_')
                temp_id_parts = []
                found_year_underscore = False
                for part_idx, part_val in enumerate(parts):
                    if re.match(r"^\d{4}$", part_val) and part_idx > 0 and (parts[part_idx-1] == "" or (part_idx > 1 and parts[part_idx-2] == "")): # Handles __YYYY or _YYYY
                        found_year_underscore = True
                        if part_idx + 1 < len(parts):
                            temp_id_parts = parts[part_idx+1:] 
                            break
                if temp_id_parts:
                    citation_doc_id_to_use_in_prompt = "_".join(temp_id_parts)
            
            if citation_doc_id_to_use_in_prompt == "N/A" and original_doc_id != "N/A":
                 citation_doc_id_to_use_in_prompt = original_doc_id

            # This is the chunk_id from store.py (e.g., "February_03__2025_FEB032025_01B2203_chunk_uuid")
            chunk_id_for_citation = meta.get('chunk_id', 'N/A')

            context_parts.append(
                f"--- CONTEXT {i+1} ---\n"
                # These are the values the LLM prompt will refer to for constructing citations
                f"Source Document ID (Original Filename based): {meta.get('document_id', 'N/A')}\n" # For reference/debug
                f"Case Name (for citation): {meta.get('case_name', 'N/A')}\n"
                f"Publication Date (for citation): {meta.get('publication_date', 'N/A')}\n"
                f"Document ID (for citation): {citation_doc_id_to_use_in_prompt}\n"  
                f"Chunk ID (for citation): {chunk_id_for_citation}\n" 
                f"Relevance Score (1-distance): {relevance:.4f}\n" # For reference/debug
                f"Passage Text:\n{text_content}\n" # This is the actual text for the LLM
                f"--- END CONTEXT ---"
            )
    
        return "\n\n".join(context_parts)
    
    
    def _get_intent_based_prompt(self, intent: str, query: str, context_str: str) -> str:
        # --- MODIFICATION: Updated base_instructions for clarity on citation components ---
        base_instructions = """You are a legal research assistant specializing in USCIS AAO decisions.
Answer based ONLY on the provided context. Use precise citations for every fact you reference.
The required citation format is: [Case Name, Publication Date, Document ID: ACTUAL_DOCUMENT_ID, Chunk ID: ACTUAL_CHUNK_ID].
To construct this citation:
1. Use the 'Case Name (for citation)' from the relevant CONTEXT passage.
2. Use the 'Publication Date (for citation)' from the relevant CONTEXT passage.
3. For 'ACTUAL_DOCUMENT_ID', use the value from 'Document ID (for citation)' from the relevant CONTEXT passage.
4. For 'ACTUAL_CHUNK_ID', use the value from 'Chunk ID (for citation)' from the relevant CONTEXT passage.
Ensure all four components (Case Name, Publication Date, Document ID, Chunk ID) are included in each citation, correctly labeled as shown in the format."""
    
        intent_specific = {
            'criterion_judge': """Focus on explaining how AAO evaluates the "participation as a judge" criterion.
    Identify specific requirements, examples of qualifying/non-qualifying activities, and any patterns in decisions.""",
            
            'criterion_awards': """Focus on what makes awards qualify as "nationally or internationally recognized."
    Explain factors AAO considers, level of recognition required, and provide examples from the context.""",
            
            'concept_acclaim': """Explain how AAO interprets "sustained national or international acclaim."
    Include duration requirements, types of evidence considered, and standards applied.""",
            
            'procedural_definitional': """Provide clear procedural guidance based on AAO decisions.
    Include requirements, timelines, and common issues identified in the decisions."""
        }
        
        specific_instructions = intent_specific.get(intent, 
            "Provide a comprehensive answer addressing all aspects of the query. Synthesize your answer using any relevant information found across any of the provided context passages. Focus on how the details in the context relate to the specific aspects of the User Query.")
    
        refusal_instruction = """If, after careful review of all provided context passages, the information needed to directly answer the User Query is not present, you MUST state: "The provided context does not contain sufficient information to answer this question." """
    
        return f"""{base_instructions}
    
    {specific_instructions}
    
    {refusal_instruction}
    
    Context Passages:
    {context_str}
    
    User Query: {query}
    
    Answer:"""
    
    def warm_cache(self, common_queries: List[str]) -> Dict[str, int]:
        """Pre-warm cache with common queries."""
        if not self.cache_manager:
            return {'warmed': 0}
        
        return self.cache_manager.warm_cache(
            common_queries=common_queries,
            collection=self.collection,
            embedding_function=self.embedding_function
        )
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        stats = {
            'collection_count': self.collection.count() if self.collection else 0
        }
        
        if self.cache_manager:
            cache_stats = self.cache_manager.get_stats()
            stats.update({
                'cache_hit_rate': cache_stats.get('hit_rate', 0),
                'cache_entries': cache_stats.get('memory_entries', 0) + 
                                cache_stats.get('disk_entries', 0),
                'cache_memory_mb': cache_stats.get('memory_size_mb', 0)
            })
        
        return stats


# Example usage
if __name__ == "__main__":
    # Initialize enhanced RAG pipeline
    rag_pipeline = EnhancedRAGPipeline(use_cache=True)
    
    # Warm cache with common queries
    common_queries = [
        "How do recent AAO decisions evaluate an applicant's Participation as a Judge service criteria?",
        "What characteristics of national or international awards persuade the AAO that they constitute sustained acclaim?",
        "What are the requirements for extraordinary ability?",
        "How does AAO evaluate original contributions of major significance?"
    ]
    
    # warm_stats = rag_pipeline.warm_cache(common_queries) # Optional: uncomment to test warming
    # print(f"Cache warmed with {warm_stats} entries")
    
    # Test queries
    test_query_1 = "How do recent AAO decisions evaluate an applicant's Participation as a Judge service criteria?"
    test_query_2 = "What characteristics of national or international awards persuade the AAO that they constitute sustained acclaim?"
    
    # First run (cold cache for these specific queries if not in common_queries or cache cleared)
    print(f"\nQuery: {test_query_1}")
    print("First run (potentially cold cache for this query):")
    result1 = rag_pipeline.answer_query(test_query_1)
    print("\n--- Answer ---")
    print(result1['answer'])
    print("--------------")
    print(f"Performance: {result1['performance_metrics']}")
    
    # Second run (should be warm cache for query results and possibly answer)
    print(f"\nQuery: {test_query_1}")
    print("\nSecond run (warm cache for this query):")
    result2 = rag_pipeline.answer_query(test_query_1)
    print("\n--- Answer ---")
    print(result2['answer'])
    print("--------------")
    print(f"Performance: {result2['performance_metrics']}")

    print(f"\nQuery: {test_query_2}")
    print("First run (potentially cold cache for this query):")
    result3 = rag_pipeline.answer_query(test_query_2)
    print("\n--- Answer ---")
    print(result3['answer'])
    print("--------------")
    print(f"Performance: {result3['performance_metrics']}")
    
    # Show performance stats
    stats = rag_pipeline.get_performance_stats()
    print(f"\nSystem Statistics:")
    print(f"Collection size: {stats['collection_count']} chunks")
    if 'cache_hit_rate' in stats:
        print(f"Cache hit rate: {stats.get('cache_hit_rate', 0):.2%}")
        print(f"Cache entries: {stats.get('cache_entries', 0)}")
        print(f"Cache memory: {stats.get('cache_memory_mb', 0):.2f} MB")