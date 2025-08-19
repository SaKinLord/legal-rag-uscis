"""
Advanced multi-factor document ranking for legal RAG systems.
Implements Phase 1 enhancements from RAG_Enhancement_Strategy.md
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AdvancedDocumentRanker:
    """Multi-factor ranking algorithm for legal document retrieval."""
    
    def __init__(self):
        # Legal analysis indicators
        self.analysis_indicators = {
            'case_citations': [
                r'\b\d+\s+U\.S\.\s+\d+',
                r'\b\d+\s+F\.\d+d\s+\d+',
                r'\bMatter of [A-Z][a-z-]+',
                r'\b\d+\s+I&N\s+Dec\.\s+\d+'
            ],
            'legal_standards': [
                r'8\s+C\.F\.R\.\s+§\s+204\.5',
                r'8\s+U\.S\.C\.\s+§\s+1153',
                r'extraordinary ability',
                r'sustained national or international acclaim',
                r'small percentage.*top.*field'
            ],
            'analysis_sections': [
                r'analysis:?\s*$',
                r'discussion:?\s*$',
                r'findings?:?\s*$',
                r'conclusion:?\s*$',
                r'determination:?\s*$'
            ],
            'precedent_references': [
                r'see\s+Matter\s+of',
                r'citing\s+Matter\s+of',
                r'as\s+stated\s+in',
                r'consistent\s+with',
                r'in\s+accordance\s+with'
            ]
        }
        
        # Query-specific relevance patterns
        self.query_patterns = {
            'judging_criteria': {
                'primary': [r'judge', r'judging', r'evaluation', r'panel', r'peer\s+review'],
                'secondary': [r'participation', r'role', r'committee', r'selection', r'assessment']
            },
            'awards_criteria': {
                'primary': [r'award', r'recognition', r'prize', r'honor', r'acclaim'],
                'secondary': [r'national', r'international', r'prestigious', r'competitive', r'excellence']
            },
            'sustained_acclaim': {
                'primary': [r'sustained', r'acclaim', r'recognition', r'ongoing', r'continuous'],
                'secondary': [r'career', r'achievements', r'contributions', r'field', r'expertise']
            }
        }
    
    def rank_documents(self, 
                      query: str, 
                      candidates: List[Dict],
                      query_embedding: np.ndarray = None) -> List[Tuple[str, float, Dict]]:
        """
        Rank documents using multi-factor algorithm.
        
        Args:
            query: Original user query
            candidates: List of candidate documents with metadata
            query_embedding: Query embedding vector for semantic similarity
            
        Returns:
            List of (doc_id, final_score, metadata) tuples sorted by relevance
        """
        if not candidates:
            return []
        
        scored_docs = []
        query_type = self._classify_query_type(query)
        
        for candidate in candidates:
            doc_id = candidate.get('id', '')
            doc_text = candidate.get('text', '')
            doc_embedding = candidate.get('embedding')
            metadata = candidate.get('metadata', {})
            base_distance = candidate.get('distance', 1.0)
            
            # Factor 1: Semantic similarity (40%)
            semantic_score = max(0, 1 - base_distance) if base_distance is not None else 0.5
            
            # Factor 2: Legal analysis depth (25%)
            analysis_score = self._calculate_analysis_depth(doc_text)
            
            # Factor 3: Query-specific relevance (20%)
            relevance_score = self._calculate_query_relevance(query, query_type, doc_text)
            
            # Factor 4: Document authority (10%)
            authority_score = self._calculate_document_authority(metadata)
            
            # Factor 5: Recency bias (5%)
            recency_score = self._calculate_recency_score(metadata)
            
            # Combine scores with weights
            final_score = (
                semantic_score * 0.40 +
                analysis_score * 0.25 +
                relevance_score * 0.20 +
                authority_score * 0.10 +
                recency_score * 0.05
            )
            
            scoring_details = {
                'semantic_score': semantic_score,
                'analysis_score': analysis_score,
                'relevance_score': relevance_score,
                'authority_score': authority_score,
                'recency_score': recency_score,
                'final_score': final_score,
                'query_type': query_type
            }
            
            scored_docs.append((doc_id, final_score, {**metadata, 'scoring_details': scoring_details}))
        
        # Sort by final score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Ranked {len(scored_docs)} documents for query type: {query_type}")
        return scored_docs
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query type for targeted relevance scoring."""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['judge', 'judging', 'participation as judge']):
            return 'judging_criteria'
        elif any(term in query_lower for term in ['award', 'recognition', 'national', 'international']):
            return 'awards_criteria'
        elif any(term in query_lower for term in ['sustained', 'acclaim', 'ongoing']):
            return 'sustained_acclaim'
        else:
            return 'general'
    
    def _calculate_analysis_depth(self, doc_text: str) -> float:
        """Calculate legal analysis depth score (0-1)."""
        if not doc_text:
            return 0.0
        
        text_length = len(doc_text.split())
        if text_length == 0:
            return 0.0
        
        depth_score = 0.0
        
        # Check for case citations
        citation_count = 0
        for pattern in self.analysis_indicators['case_citations']:
            citation_count += len(re.findall(pattern, doc_text, re.IGNORECASE))
        depth_score += min(0.3, citation_count * 0.1)
        
        # Check for legal standards
        standards_count = 0
        for pattern in self.analysis_indicators['legal_standards']:
            standards_count += len(re.findall(pattern, doc_text, re.IGNORECASE))
        depth_score += min(0.3, standards_count * 0.15)
        
        # Check for analysis sections
        analysis_sections = 0
        for pattern in self.analysis_indicators['analysis_sections']:
            analysis_sections += len(re.findall(pattern, doc_text, re.IGNORECASE | re.MULTILINE))
        depth_score += min(0.2, analysis_sections * 0.1)
        
        # Check for precedent references
        precedent_refs = 0
        for pattern in self.analysis_indicators['precedent_references']:
            precedent_refs += len(re.findall(pattern, doc_text, re.IGNORECASE))
        depth_score += min(0.2, precedent_refs * 0.05)
        
        return min(1.0, depth_score)
    
    def _calculate_query_relevance(self, query: str, query_type: str, doc_text: str) -> float:
        """Calculate query-specific relevance score (0-1)."""
        if not doc_text or query_type == 'general':
            return 0.5  # Neutral score for general queries
        
        patterns = self.query_patterns.get(query_type, {})
        primary_patterns = patterns.get('primary', [])
        secondary_patterns = patterns.get('secondary', [])
        
        text_lower = doc_text.lower()
        text_length = len(doc_text.split())
        
        if text_length == 0:
            return 0.0
        
        # Count primary pattern matches
        primary_matches = 0
        for pattern in primary_patterns:
            primary_matches += len(re.findall(pattern, text_lower))
        
        # Count secondary pattern matches
        secondary_matches = 0
        for pattern in secondary_patterns:
            secondary_matches += len(re.findall(pattern, text_lower))
        
        # Calculate relevance density
        primary_density = primary_matches / text_length * 1000  # Per 1000 words
        secondary_density = secondary_matches / text_length * 1000
        
        # Weight primary matches more heavily
        relevance_score = (primary_density * 0.7 + secondary_density * 0.3)
        
        # Normalize to 0-1 range (cap at reasonable maximum)
        return min(1.0, relevance_score / 5.0)
    
    def _calculate_document_authority(self, metadata: Dict) -> float:
        """Calculate document authority score (0-1)."""
        authority_score = 0.5  # Base score
        
        # Check for AAO decision indicators
        doc_id = metadata.get('document_id', '').upper()
        case_name = metadata.get('case_name', '').upper()
        
        # AAO decisions typically have higher authority
        if 'AAO' in case_name or any(indicator in doc_id for indicator in ['AAO', 'ADMIN']):
            authority_score += 0.3
        
        # Recent decisions may have more authority
        pub_date = metadata.get('publication_date', '')
        if pub_date:
            try:
                # Extract year from publication date
                year_match = re.search(r'20\d{2}', pub_date)
                if year_match:
                    year = int(year_match.group())
                    current_year = datetime.now().year
                    if year >= current_year - 2:  # Last 2 years
                        authority_score += 0.2
                    elif year >= current_year - 5:  # Last 5 years
                        authority_score += 0.1
            except (ValueError, AttributeError):
                pass
        
        return min(1.0, authority_score)
    
    def _calculate_recency_score(self, metadata: Dict) -> float:
        """Calculate recency bias score (0-1)."""
        pub_date = metadata.get('publication_date', '')
        if not pub_date:
            return 0.5  # Neutral score for unknown dates
        
        try:
            # Extract year from publication date
            year_match = re.search(r'20\d{2}', pub_date)
            if not year_match:
                return 0.5
            
            year = int(year_match.group())
            current_year = datetime.now().year
            years_old = current_year - year
            
            # Score decreases with age
            if years_old <= 1:
                return 1.0
            elif years_old <= 3:
                return 0.8
            elif years_old <= 5:
                return 0.6
            elif years_old <= 10:
                return 0.4
            else:
                return 0.2
                
        except (ValueError, AttributeError):
            return 0.5
    
    def calculate_dynamic_threshold(self, 
                                  query: str, 
                                  scored_documents: List[Tuple[str, float, Dict]]) -> float:
        """
        Calculate dynamic relevance threshold based on query complexity and result quality.
        
        Args:
            query: Original user query
            scored_documents: List of scored documents
            
        Returns:
            Dynamic threshold value (0.3-0.8)
        """
        if not scored_documents:
            return 0.5
        
        # Base threshold
        base_threshold = 0.5
        
        # Calculate query complexity
        complexity_factor = self._calculate_query_complexity(query)
        complexity_adjustment = (complexity_factor - 0.5) * 0.2
        
        # Calculate result quality distribution
        scores = [score for _, score, _ in scored_documents]
        if len(scores) > 1:
            score_std = np.std(scores)
            quality_adjustment = min(0.15, score_std * 0.3)
        else:
            quality_adjustment = 0
        
        dynamic_threshold = base_threshold + complexity_adjustment - quality_adjustment
        
        # Ensure threshold is within reasonable bounds
        return max(0.3, min(0.8, dynamic_threshold))
    
    def _calculate_query_complexity(self, query: str) -> float:
        """Calculate query complexity score (0-1)."""
        complexity_score = 0.5  # Base complexity
        
        # Length factor
        word_count = len(query.split())
        if word_count > 15:
            complexity_score += 0.2
        elif word_count > 10:
            complexity_score += 0.1
        
        # Legal terminology density
        legal_terms = [
            'extraordinary ability', 'sustained acclaim', 'national recognition',
            'international recognition', 'criteria', 'requirements', 'AAO', 'USCIS'
        ]
        
        query_lower = query.lower()
        legal_term_count = sum(1 for term in legal_terms if term in query_lower)
        complexity_score += min(0.3, legal_term_count * 0.1)
        
        # Question complexity indicators
        complex_indicators = ['how do', 'what characteristics', 'in what way', 'to what extent']
        if any(indicator in query_lower for indicator in complex_indicators):
            complexity_score += 0.1
        
        return min(1.0, complexity_score)


def create_advanced_ranker() -> AdvancedDocumentRanker:
    """Factory function to create an advanced document ranker."""
    return AdvancedDocumentRanker()


if __name__ == "__main__":
    # Test the advanced ranker
    ranker = AdvancedDocumentRanker()
    
    # Test query complexity calculation
    test_queries = [
        "What is extraordinary ability?",
        "How do recent AAO decisions evaluate an applicant's Participation as a Judge service criteria?",
        "What characteristics of national or international awards persuade the AAO that they constitute sustained acclaim?"
    ]
    
    for query in test_queries:
        complexity = ranker._calculate_query_complexity(query)
        query_type = ranker._classify_query_type(query)
        print(f"Query: {query}")
        print(f"Complexity: {complexity:.3f}, Type: {query_type}")
        print("-" * 60)