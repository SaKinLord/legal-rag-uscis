"""
Hybrid retrieval system implementing Phase 2 enhancements from RAG_Enhancement_Strategy.md
Combines dense (vector), sparse (BM25), and legal-specific retrieval methods.
"""

import os
import re
import math
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict, Counter
import chromadb
from chromadb.utils import embedding_functions
import logging

logger = logging.getLogger(__name__)


class SparseRetriever:
    """BM25-based sparse retrieval for keyword matching."""
    
    def __init__(self, collection_path: str):
        self.collection_path = collection_path
        self.documents = {}
        self.term_frequencies = {}
        self.doc_frequencies = {}
        self.doc_lengths = {}
        self.avg_doc_length = 0
        self.total_docs = 0
        self.k1 = 1.5  # BM25 parameter
        self.b = 0.75  # BM25 parameter
        
    def build_index(self, documents: List[Dict[str, Any]]):
        """Build BM25 index from documents."""
        logger.info(f"Building BM25 index for {len(documents)} documents")
        
        self.documents = {doc['id']: doc for doc in documents}
        self.total_docs = len(documents)
        
        # Calculate term frequencies and document frequencies
        all_lengths = []
        
        for doc in documents:
            doc_id = doc['id']
            text = doc['text'].lower()
            terms = self._tokenize(text)
            
            # Calculate term frequencies for this document
            tf = Counter(terms)
            self.term_frequencies[doc_id] = tf
            self.doc_lengths[doc_id] = len(terms)
            all_lengths.append(len(terms))
            
            # Update document frequencies
            for term in set(terms):
                if term not in self.doc_frequencies:
                    self.doc_frequencies[term] = 0
                self.doc_frequencies[term] += 1
        
        self.avg_doc_length = sum(all_lengths) / len(all_lengths) if all_lengths else 0
        logger.info(f"BM25 index built: {len(self.doc_frequencies)} unique terms, avg doc length: {self.avg_doc_length:.1f}")
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms."""
        # Simple tokenization - can be enhanced with better preprocessing
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        terms = [term for term in text.split() if len(term) > 2]
        return terms
    
    def retrieve(self, query: str, k: int = 10) -> List[Tuple[str, float, Dict]]:
        """Retrieve documents using BM25 scoring."""
        if not self.documents:
            return []
        
        query_terms = self._tokenize(query)
        if not query_terms:
            return []
        
        scores = {}
        
        for doc_id in self.documents:
            score = self._calculate_bm25_score(query_terms, doc_id)
            if score > 0:
                scores[doc_id] = score
        
        # Sort by score and return top k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        results = []
        for doc_id, score in sorted_results:
            doc = self.documents[doc_id]
            results.append((doc_id, score, doc))
        
        return results
    
    def _calculate_bm25_score(self, query_terms: List[str], doc_id: str) -> float:
        """Calculate BM25 score for a document given query terms."""
        if doc_id not in self.term_frequencies:
            return 0.0
        
        tf = self.term_frequencies[doc_id]
        doc_length = self.doc_lengths[doc_id]
        score = 0.0
        
        for term in query_terms:
            if term in tf and term in self.doc_frequencies:
                # BM25 formula
                term_freq = tf[term]
                doc_freq = self.doc_frequencies[term]
                
                idf = math.log((self.total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
                tf_component = (term_freq * (self.k1 + 1)) / (
                    term_freq + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                )
                
                score += idf * tf_component
        
        return score


class LegalConceptRetriever:
    """Legal domain-specific retrieval using legal patterns and concepts."""
    
    def __init__(self):
        # Legal concept patterns for enhanced matching
        self.legal_patterns = {
            'extraordinary_ability_criteria': {
                'patterns': [
                    r'8\s+C\.F\.R\.\s+§\s+204\.5\(h\)\(3\)',
                    r'outstanding achievements',
                    r'recognized national or international experts',
                    r'small percentage.*top.*field',
                    r'sustained national.*international acclaim'
                ],
                'boost': 2.0
            },
            'judging_criteria': {
                'patterns': [
                    r'participation.*judge',
                    r'judging.*work of others',
                    r'evaluation.*criteria',
                    r'peer review',
                    r'selection committee',
                    r'editorial board',
                    r'review panel'
                ],
                'boost': 1.8
            },
            'awards_recognition': {
                'patterns': [
                    r'nationally.*recognized.*award',
                    r'internationally.*recognized.*award',
                    r'prestigious.*award',
                    r'competitive.*selection',
                    r'merit.*based.*recognition',
                    r'professional.*honor'
                ],
                'boost': 1.5
            },
            'legal_standards': {
                'patterns': [
                    r'preponderance.*evidence',
                    r'burden.*proof',
                    r'regulatory.*criteria',
                    r'evidentiary.*standard',
                    r'Matter\s+of\s+[A-Z]',
                    r'precedent.*analysis'
                ],
                'boost': 1.3
            }
        }
        
        self.documents = {}
    
    def build_index(self, documents: List[Dict[str, Any]]):
        """Build legal concept index."""
        logger.info(f"Building legal concept index for {len(documents)} documents")
        
        self.documents = {doc['id']: doc for doc in documents}
        
        # Pre-compute pattern matches for efficiency
        for doc in documents:
            doc_id = doc['id']
            text = doc['text']
            doc['legal_concept_matches'] = self._extract_legal_concepts(text)
    
    def _extract_legal_concepts(self, text: str) -> Dict[str, int]:
        """Extract legal concept matches from text."""
        matches = {}
        
        for concept_type, concept_data in self.legal_patterns.items():
            patterns = concept_data['patterns']
            match_count = 0
            
            for pattern in patterns:
                pattern_matches = len(re.findall(pattern, text, re.IGNORECASE))
                match_count += pattern_matches
            
            matches[concept_type] = match_count
        
        return matches
    
    def retrieve(self, query: str, k: int = 10) -> List[Tuple[str, float, Dict]]:
        """Retrieve documents using legal concept matching."""
        if not self.documents:
            return []
        
        query_lower = query.lower()
        query_concepts = self._identify_query_concepts(query_lower)
        
        if not query_concepts:
            return []
        
        scores = {}
        
        for doc_id, doc in self.documents.items():
            score = self._calculate_legal_concept_score(query_concepts, doc)
            if score > 0:
                scores[doc_id] = score
        
        # Sort by score and return top k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        results = []
        for doc_id, score in sorted_results:
            doc = self.documents[doc_id]
            results.append((doc_id, score, doc))
        
        return results
    
    def _identify_query_concepts(self, query_lower: str) -> List[str]:
        """Identify which legal concepts are relevant to the query."""
        relevant_concepts = []
        
        concept_indicators = {
            'extraordinary_ability_criteria': ['extraordinary', 'ability', 'acclaim', 'outstanding'],
            'judging_criteria': ['judge', 'judging', 'evaluation', 'panel', 'review'],
            'awards_recognition': ['award', 'recognition', 'prize', 'honor', 'acclaim'],
            'legal_standards': ['criteria', 'standard', 'evidence', 'precedent']
        }
        
        for concept, indicators in concept_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                relevant_concepts.append(concept)
        
        return relevant_concepts
    
    def _calculate_legal_concept_score(self, query_concepts: List[str], doc: Dict) -> float:
        """Calculate legal concept relevance score."""
        if 'legal_concept_matches' not in doc:
            return 0.0
        
        matches = doc['legal_concept_matches']
        score = 0.0
        
        for concept in query_concepts:
            if concept in matches and concept in self.legal_patterns:
                match_count = matches[concept]
                boost = self.legal_patterns[concept]['boost']
                score += match_count * boost
        
        # Normalize by document length (rough approximation)
        doc_length = len(doc['text'].split())
        normalized_score = score / (doc_length / 1000.0) if doc_length > 0 else 0
        
        return normalized_score


class HybridRetriever:
    """
    Hybrid retrieval system combining dense, sparse, and legal-specific retrieval.
    Implements Phase 2 enhancement strategy.
    """
    
    def __init__(self, vector_db_path: str, collection_name: str, embedding_model_name: str):
        self.vector_db_path = vector_db_path
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        
        # Initialize dense retriever (ChromaDB)
        self.dense_retriever = None
        self.embedding_function = None
        
        # Initialize sparse retriever (BM25)
        self.sparse_retriever = SparseRetriever(vector_db_path)
        
        # Initialize legal retriever
        self.legal_retriever = LegalConceptRetriever()
        
        self._initialize_retrievers()
    
    def _initialize_retrievers(self):
        """Initialize all retrieval components."""
        try:
            # Initialize dense retriever
            client = chromadb.PersistentClient(path=self.vector_db_path)
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name
            )
            
            self.dense_retriever = client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            
            # Build indexes for sparse and legal retrievers
            self._build_additional_indexes()
            
            logger.info("Hybrid retriever initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing hybrid retriever: {e}")
            raise
    
    def _build_additional_indexes(self):
        """Build indexes for sparse and legal retrievers."""
        if not self.dense_retriever:
            return
        
        # Get all documents from dense retriever
        try:
            # Get document count
            count = self.dense_retriever.count()
            logger.info(f"Building additional indexes for {count} documents")
            
            # Retrieve all documents in batches
            batch_size = 100
            all_documents = []
            
            for offset in range(0, count, batch_size):
                limit = min(batch_size, count - offset)
                results = self.dense_retriever.get(
                    limit=limit,
                    offset=offset,
                    include=['documents', 'metadatas']
                )
                
                for i, doc_id in enumerate(results['ids']):
                    doc = {
                        'id': doc_id,
                        'text': results['documents'][i],
                        'metadata': results['metadatas'][i] if results['metadatas'] else {}
                    }
                    all_documents.append(doc)
            
            # Build indexes
            self.sparse_retriever.build_index(all_documents)
            self.legal_retriever.build_index(all_documents)
            
        except Exception as e:
            logger.error(f"Error building additional indexes: {e}")
    
    def hybrid_retrieve(self, query: str, k: int = 10, weights: Optional[Dict[str, float]] = None) -> List[Tuple[str, float, Dict]]:
        """
        Perform hybrid retrieval combining all three methods.
        
        Args:
            query: Search query
            k: Number of results to return
            weights: Optional weights for different retrievers
            
        Returns:
            List of (doc_id, score, document) tuples
        """
        if weights is None:
            weights = {"dense": 0.4, "sparse": 0.4, "legal": 0.2}
        
        # Get results from each retriever
        dense_results = self._get_dense_results(query, k * 2)
        sparse_results = self._get_sparse_results(query, k * 2)
        legal_results = self._get_legal_results(query, k * 2)
        
        # Ensemble ranking
        return self.ensemble_rank(
            [dense_results, sparse_results, legal_results],
            weights,
            k
        )
    
    def _get_dense_results(self, query: str, k: int) -> List[Tuple[str, float, Dict]]:
        """Get results from dense retriever."""
        if not self.dense_retriever:
            return []
        
        try:
            results = self.dense_retriever.query(
                query_texts=[query],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            
            formatted_results = []
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    # Convert distance to similarity score
                    distance = results['distances'][0][i]
                    similarity_score = max(0, 1 - distance)
                    
                    doc = {
                        'id': doc_id,
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'retriever': 'dense'
                    }
                    
                    formatted_results.append((doc_id, similarity_score, doc))
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in dense retrieval: {e}")
            return []
    
    def _get_sparse_results(self, query: str, k: int) -> List[Tuple[str, float, Dict]]:
        """Get results from sparse retriever."""
        try:
            results = self.sparse_retriever.retrieve(query, k)
            
            # Add retriever info to metadata
            formatted_results = []
            for doc_id, score, doc in results:
                doc['retriever'] = 'sparse'
                formatted_results.append((doc_id, score, doc))
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in sparse retrieval: {e}")
            return []
    
    def _get_legal_results(self, query: str, k: int) -> List[Tuple[str, float, Dict]]:
        """Get results from legal concept retriever."""
        try:
            results = self.legal_retriever.retrieve(query, k)
            
            # Add retriever info to metadata
            formatted_results = []
            for doc_id, score, doc in results:
                doc['retriever'] = 'legal'
                formatted_results.append((doc_id, score, doc))
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in legal retrieval: {e}")
            return []
    
    def ensemble_rank(self, 
                     results_lists: List[List[Tuple[str, float, Dict]]], 
                     weights: Dict[str, float],
                     k: int) -> List[Tuple[str, float, Dict]]:
        """
        Ensemble ranking algorithm to combine results from multiple retrievers.
        """
        retriever_names = ["dense", "sparse", "legal"]
        combined_scores = defaultdict(float)
        doc_data = {}
        
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {key: val / total_weight for key, val in weights.items()}
        
        # Combine scores from all retrievers
        for i, results in enumerate(results_lists):
            retriever_name = retriever_names[i]
            weight = normalized_weights.get(retriever_name, 0.0)
            
            if weight == 0:
                continue
            
            # Normalize scores within this retriever's results
            if results:
                max_score = max(score for _, score, _ in results)
                min_score = min(score for _, score, _ in results)
                score_range = max_score - min_score if max_score > min_score else 1.0
                
                for doc_id, score, doc in results:
                    # Normalize score to 0-1 range
                    normalized_score = (score - min_score) / score_range if score_range > 0 else 0.5
                    
                    # Add weighted score
                    combined_scores[doc_id] += normalized_score * weight
                    
                    # Store document data (use the one with highest individual score)
                    if doc_id not in doc_data or score > doc_data[doc_id][1]:
                        doc_data[doc_id] = (doc, score)
        
        # Sort by combined score and return top k
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        final_results = []
        for doc_id, combined_score in sorted_results:
            if doc_id in doc_data:
                doc, original_score = doc_data[doc_id]
                doc['ensemble_score'] = combined_score
                doc['original_score'] = original_score
                final_results.append((doc_id, combined_score, doc))
        
        logger.info(f"Ensemble ranking: combined {len(results_lists)} retrievers -> {len(final_results)} results")
        return final_results


def create_hybrid_retriever(vector_db_path: str, collection_name: str, embedding_model_name: str) -> HybridRetriever:
    """Factory function to create a hybrid retriever."""
    return HybridRetriever(vector_db_path, collection_name, embedding_model_name)


if __name__ == "__main__":
    # Test the hybrid retriever
    import os
    
    vector_db_path = os.path.join(os.path.dirname(__file__), '..', 'vector_db')
    collection_name = "aao_decisions"
    embedding_model_name = "BAAI/bge-small-en-v1.5"
    
    try:
        hybrid_retriever = create_hybrid_retriever(vector_db_path, collection_name, embedding_model_name)
        
        test_query = "How do recent AAO decisions evaluate participation as a judge criteria?"
        results = hybrid_retriever.hybrid_retrieve(test_query, k=5)
        
        print(f"Hybrid retrieval results for: {test_query}")
        print("-" * 60)
        
        for i, (doc_id, score, doc) in enumerate(results, 1):
            retriever = doc.get('retriever', 'unknown')
            ensemble_score = doc.get('ensemble_score', 0)
            print(f"{i}. {doc_id} (retriever: {retriever}, score: {score:.3f}, ensemble: {ensemble_score:.3f})")
            print(f"   {doc['text'][:100]}...")
            print()
            
    except Exception as e:
        print(f"Error testing hybrid retriever: {e}")