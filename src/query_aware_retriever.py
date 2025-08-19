"""
Query-aware retrieval optimization implementing Phase 2 enhancements.
Provides query classification and adaptive retrieval strategies.
"""

import re
import math
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class QueryClassification:
    """Query classification result."""
    primary_type: str
    confidence: float
    all_scores: Dict[str, float]
    complexity: float


class QueryClassifier:
    """
    Advanced query type classification for adaptive retrieval strategies.
    Implements Phase 2 enhancement from RAG strategy.
    """
    
    def __init__(self):
        # Enhanced classification patterns
        self.classifications = {
            'procedural': {
                'patterns': ['how do', 'what process', 'how does', 'procedure', 'steps', 'method'],
                'weight': 1.0,
                'description': 'Queries about processes and procedures'
            },
            'criteria_based': {
                'patterns': ['criteria', 'requirements', 'standards', 'qualifications', 'conditions', 'must'],
                'weight': 1.2,
                'description': 'Queries about criteria and requirements'
            },
            'comparative': {
                'patterns': ['characteristics', 'features', 'differences', 'distinguish', 'compare', 'versus'],
                'weight': 1.1,
                'description': 'Queries comparing or analyzing characteristics'
            },
            'evaluative': {
                'patterns': ['evaluate', 'assess', 'determine', 'consider', 'analyze', 'review'],
                'weight': 1.3,
                'description': 'Queries about evaluation and assessment'
            },
            'definitional': {
                'patterns': ['what is', 'what are', 'definition', 'meaning', 'define', 'means'],
                'weight': 0.8,
                'description': 'Queries seeking definitions or explanations'
            },
            'evidential': {
                'patterns': ['evidence', 'documentation', 'proof', 'demonstrate', 'establish', 'show'],
                'weight': 1.1,
                'description': 'Queries about evidence and documentation'
            }
        }
        
        # Legal-specific query patterns
        self.legal_patterns = {
            'extraordinary_ability': ['extraordinary ability', 'exceptional ability', 'outstanding achievement'],
            'sustained_acclaim': ['sustained acclaim', 'national acclaim', 'international acclaim'],
            'judging_criteria': ['participation as judge', 'judging work', 'peer review'],
            'awards_recognition': ['national award', 'international award', 'recognition', 'prize'],
            'regulatory': ['8 cfr', 'regulation', 'uscis', 'aao decision', 'matter of']
        }
    
    def classify_query(self, query: str) -> QueryClassification:
        """
        Classify query type with confidence scoring.
        
        Args:
            query: User query to classify
            
        Returns:
            QueryClassification with type, confidence, and detailed scores
        """
        query_lower = query.lower().strip()
        
        # Calculate scores for each query type
        type_scores = {}
        
        for query_type, type_data in self.classifications.items():
            patterns = type_data['patterns']
            weight = type_data['weight']
            
            # Count pattern matches
            matches = sum(1 for pattern in patterns if pattern in query_lower)
            
            # Calculate weighted score
            pattern_density = matches / len(patterns)
            type_scores[query_type] = pattern_density * weight
        
        # Find primary type
        if type_scores:
            primary_type = max(type_scores, key=type_scores.get)
            max_score = type_scores[primary_type]
            
            # Calculate confidence based on score separation
            sorted_scores = sorted(type_scores.values(), reverse=True)
            if len(sorted_scores) > 1:
                score_separation = sorted_scores[0] - sorted_scores[1]
                confidence = min(1.0, max_score + score_separation)
            else:
                confidence = max_score
            
            # Ensure minimum confidence
            confidence = max(0.1, confidence)
        else:
            primary_type = 'definitional'  # Default
            confidence = 0.1
            type_scores = {t: 0.0 for t in self.classifications.keys()}
        
        # Calculate query complexity
        complexity = self._calculate_query_complexity(query)
        
        return QueryClassification(
            primary_type=primary_type,
            confidence=confidence,
            all_scores=type_scores,
            complexity=complexity
        )
    
    def _calculate_query_complexity(self, query: str) -> float:
        """Calculate query complexity score (0-1)."""
        complexity_score = 0.0
        
        # Length factor
        word_count = len(query.split())
        if word_count > 20:
            complexity_score += 0.3
        elif word_count > 10:
            complexity_score += 0.2
        elif word_count > 5:
            complexity_score += 0.1
        
        # Legal terminology density
        legal_term_count = 0
        query_lower = query.lower()
        
        for category, terms in self.legal_patterns.items():
            for term in terms:
                if term in query_lower:
                    legal_term_count += 1
        
        complexity_score += min(0.4, legal_term_count * 0.1)
        
        # Structural complexity indicators
        complex_structures = [
            r'\b(how|what|when|where|why)\b.*\b(do|does|are|is)\b',  # Question structures
            r'\b(such as|including|for example)\b',  # Elaboration
            r'\b(both|either|neither|not only)\b',  # Logical structures
            r'\b(criteria|requirements|standards)\b.*\b(and|or)\b'  # Multiple criteria
        ]
        
        for pattern in complex_structures:
            if re.search(pattern, query_lower):
                complexity_score += 0.1
        
        return min(1.0, complexity_score)
    
    def get_legal_domain_indicators(self, query: str) -> Dict[str, float]:
        """Get legal domain relevance indicators."""
        query_lower = query.lower()
        indicators = {}
        
        for domain, terms in self.legal_patterns.items():
            relevance = 0.0
            for term in terms:
                if term in query_lower:
                    relevance += 1.0
            
            # Normalize by number of terms
            indicators[domain] = relevance / len(terms) if terms else 0.0
        
        return indicators


class AdaptiveRetrievalStrategy:
    """
    Adaptive retrieval strategy that adjusts based on query type and characteristics.
    """
    
    def __init__(self):
        self.classifier = QueryClassifier()
        
        # Retriever weight strategies by query type
        self.weight_strategies = {
            'criteria_based': {
                'dense': 0.4,
                'sparse': 0.4,
                'legal': 0.2,
                'boost_terms': ['establish', 'demonstrate', 'evidence', 'criteria', 'standards']
            },
            'evaluative': {
                'dense': 0.3,
                'sparse': 0.4,
                'legal': 0.3,
                'boost_terms': ['assess', 'evaluate', 'determine', 'consider', 'analyze']
            },
            'comparative': {
                'dense': 0.5,
                'sparse': 0.3,
                'legal': 0.2,
                'boost_terms': ['characteristics', 'features', 'distinguish', 'compare', 'versus']
            },
            'procedural': {
                'dense': 0.3,
                'sparse': 0.5,
                'legal': 0.2,
                'boost_terms': ['process', 'procedure', 'steps', 'method', 'how to']
            },
            'definitional': {
                'dense': 0.6,
                'sparse': 0.2,
                'legal': 0.2,
                'boost_terms': ['definition', 'meaning', 'what is', 'explain']
            },
            'evidential': {
                'dense': 0.3,
                'sparse': 0.3,
                'legal': 0.4,
                'boost_terms': ['evidence', 'documentation', 'proof', 'demonstrate']
            }
        }
        
        # Awards/recognition specific strategy
        self.awards_strategy = {
            'dense': 0.3,
            'sparse': 0.3,
            'legal': 0.4,
            'boost_terms': ['national', 'international', 'sustained', 'acclaim', 'prestigious', 'competitive']
        }
        
        # Judging specific strategy  
        self.judging_strategy = {
            'dense': 0.4,
            'sparse': 0.4,
            'legal': 0.2,
            'boost_terms': ['judge', 'judging', 'evaluation', 'panel', 'peer review', 'expertise']
        }
    
    def get_adaptive_strategy(self, query: str) -> Dict[str, Any]:
        """
        Get adaptive retrieval strategy based on query analysis.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with weights, boost terms, and strategy metadata
        """
        # Classify query
        classification = self.classifier.classify_query(query)
        
        # Get base strategy
        base_strategy = self.weight_strategies.get(
            classification.primary_type, 
            self.weight_strategies['definitional']  # Default
        )
        
        # Check for legal domain-specific adjustments
        legal_indicators = self.classifier.get_legal_domain_indicators(query)
        
        # Adjust strategy based on legal domain
        adjusted_strategy = base_strategy.copy()
        
        if legal_indicators.get('awards_recognition', 0) > 0.3:
            # Awards/recognition queries
            adjusted_strategy.update(self.awards_strategy)
            
        elif legal_indicators.get('judging_criteria', 0) > 0.3:
            # Judging criteria queries
            adjusted_strategy.update(self.judging_strategy)
            
        elif legal_indicators.get('regulatory', 0) > 0.3:
            # Regulatory/legal procedure queries
            adjusted_strategy['legal'] = min(0.5, adjusted_strategy['legal'] + 0.2)
            adjusted_strategy['dense'] = max(0.2, adjusted_strategy['dense'] - 0.1)
        
        # Dynamic k-value selection
        optimal_k = self._select_optimal_k(classification)
        
        return {
            'weights': {
                'dense': adjusted_strategy['dense'],
                'sparse': adjusted_strategy['sparse'], 
                'legal': adjusted_strategy['legal']
            },
            'boost_terms': adjusted_strategy['boost_terms'],
            'optimal_k': optimal_k,
            'classification': classification,
            'legal_indicators': legal_indicators,
            'strategy_type': self._determine_strategy_type(legal_indicators, classification)
        }
    
    def _select_optimal_k(self, classification: QueryClassification) -> int:
        """
        Dynamically select optimal number of documents to retrieve.
        
        Args:
            classification: Query classification result
            
        Returns:
            Optimal k value for retrieval
        """
        base_k = 5
        
        # Adjust based on query type
        type_multipliers = {
            'comparative': 1.5,      # Need more examples for comparison
            'criteria_based': 1.3,   # Need comprehensive coverage
            'evaluative': 1.2,       # Need multiple perspectives
            'procedural': 1.1,       # May need step-by-step details
            'evidential': 1.4,       # Need multiple evidence sources
            'definitional': 1.0      # Usually need focused definitions
        }
        
        multiplier = type_multipliers.get(classification.primary_type, 1.0)
        
        # Adjust for confidence (lower confidence = get more documents)
        confidence_adjustment = 1.0 + (0.5 - classification.confidence) * 0.4
        
        # Adjust for complexity (higher complexity = get more documents)  
        complexity_adjustment = 1.0 + classification.complexity * 0.3
        
        optimal_k = int(base_k * multiplier * confidence_adjustment * complexity_adjustment)
        
        # Ensure reasonable bounds
        return max(3, min(15, optimal_k))
    
    def _determine_strategy_type(self, legal_indicators: Dict[str, float], classification: QueryClassification) -> str:
        """Determine the overall strategy type being used."""
        max_legal_indicator = max(legal_indicators.values()) if legal_indicators else 0
        
        if max_legal_indicator > 0.5:
            # Strong legal domain match
            max_domain = max(legal_indicators.keys(), key=legal_indicators.get)
            return f"legal_domain_{max_domain}"
        elif classification.confidence > 0.7:
            # High confidence in query type
            return f"query_type_{classification.primary_type}"
        else:
            # Balanced approach for uncertain queries
            return "balanced_retrieval"


class QueryAwareRetriever:
    """
    Main query-aware retriever that combines classification and adaptive strategies.
    """
    
    def __init__(self, hybrid_retriever):
        self.hybrid_retriever = hybrid_retriever
        self.adaptive_strategy = AdaptiveRetrievalStrategy()
    
    def retrieve(self, query: str, base_k: int = 5) -> Tuple[List[Tuple[str, float, Dict]], Dict[str, Any]]:
        """
        Perform query-aware retrieval with adaptive strategy.
        
        Args:
            query: User query
            base_k: Base number of results (will be adjusted adaptively)
            
        Returns:
            Tuple of (retrieval_results, strategy_metadata)
        """
        # Get adaptive strategy
        strategy = self.adaptive_strategy.get_adaptive_strategy(query)
        
        # Use optimal k from strategy
        optimal_k = strategy['optimal_k']
        
        # Perform retrieval with adaptive weights
        results = self.hybrid_retriever.hybrid_retrieve(
            query=query,
            k=optimal_k,
            weights=strategy['weights']
        )
        
        # Apply boost terms if available
        if strategy['boost_terms']:
            results = self._apply_boost_terms(query, results, strategy['boost_terms'])
        
        # Limit to base_k if optimal_k was higher
        final_results = results[:base_k]
        
        # Add strategy metadata to results
        strategy_metadata = {
            'strategy_used': strategy['strategy_type'],
            'weights_applied': strategy['weights'],
            'boost_terms': strategy['boost_terms'],
            'optimal_k_selected': optimal_k,
            'query_classification': {
                'type': strategy['classification'].primary_type,
                'confidence': strategy['classification'].confidence,
                'complexity': strategy['classification'].complexity
            },
            'legal_indicators': strategy['legal_indicators'],
            'results_count': len(final_results)
        }
        
        logger.info(f"Query-aware retrieval: {strategy['strategy_type']} strategy, k={optimal_k}, {len(final_results)} results")
        
        return final_results, strategy_metadata
    
    def _apply_boost_terms(self, query: str, results: List[Tuple[str, float, Dict]], boost_terms: List[str]) -> List[Tuple[str, float, Dict]]:
        """Apply boost scoring based on boost terms."""
        boosted_results = []
        
        for doc_id, score, doc in results:
            boost_factor = 1.0
            doc_text = doc.get('text', '').lower()
            
            # Calculate boost based on boost term presence
            for term in boost_terms:
                if term.lower() in doc_text:
                    boost_factor += 0.1  # 10% boost per matching boost term
            
            boosted_score = score * boost_factor
            
            # Add boost metadata
            doc['boost_applied'] = boost_factor
            doc['original_score'] = score
            
            boosted_results.append((doc_id, boosted_score, doc))
        
        # Re-sort by boosted scores
        boosted_results.sort(key=lambda x: x[1], reverse=True)
        return boosted_results


def create_query_aware_retriever(hybrid_retriever) -> QueryAwareRetriever:
    """Factory function to create a query-aware retriever."""
    return QueryAwareRetriever(hybrid_retriever)


if __name__ == "__main__":
    # Test query classification and adaptive strategies
    classifier = QueryClassifier()
    adaptive_strategy = AdaptiveRetrievalStrategy()
    
    test_queries = [
        "How do recent AAO decisions evaluate an applicant's Participation as a Judge service criteria?",
        "What characteristics of national or international awards persuade the AAO that they constitute sustained acclaim?",
        "What is the definition of extraordinary ability?",
        "What evidence is required to demonstrate sustained national acclaim?",
        "Compare the criteria for judging participation versus awards recognition."
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        print("-" * 60)
        
        # Test classification
        classification = classifier.classify_query(query)
        print(f"Type: {classification.primary_type} (confidence: {classification.confidence:.3f})")
        print(f"Complexity: {classification.complexity:.3f}")
        
        # Test adaptive strategy
        strategy = adaptive_strategy.get_adaptive_strategy(query)
        print(f"Strategy: {strategy['strategy_type']}")
        print(f"Weights: {strategy['weights']}")
        print(f"Optimal K: {strategy['optimal_k']}")
        print(f"Boost terms: {strategy['boost_terms'][:3]}...")
        print()