"""
Legal Specialized Ranking System for Phase 3 domain specialization.
Integrates advanced legal document analysis, concept extraction, and authority analysis
for superior legal document ranking and retrieval.
"""

import math
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging

from src.legal_document_analyzer import create_legal_document_analyzer, DocumentStructureAnalysis
from src.legal_concept_matcher import create_legal_concept_matcher, ConceptExtractionResult
from src.legal_authority_analyzer import create_legal_authority_analyzer, PrecedentAnalysis

logger = logging.getLogger(__name__)


@dataclass
class LegalRankingFeatures:
    """Comprehensive legal ranking features for Phase 3."""
    document_structure: DocumentStructureAnalysis
    concept_extraction: ConceptExtractionResult
    precedent_analysis: PrecedentAnalysis
    legal_expertise_score: float
    domain_specialization_score: float
    query_legal_alignment: float


@dataclass
class LegalRankingResult:
    """Enhanced ranking result with legal specialization."""
    doc_id: str
    legal_ranking_score: float
    legal_features: LegalRankingFeatures
    ranking_explanation: Dict[str, float]
    confidence_score: float


class LegalSpecializedRanker:
    """
    Advanced legal document ranker implementing Phase 3 specialization.
    Combines deep legal analysis with sophisticated ranking algorithms.
    """
    
    def __init__(self):
        # Initialize Phase 3 analyzers
        self.doc_analyzer = create_legal_document_analyzer()
        self.concept_matcher = create_legal_concept_matcher()
        self.authority_analyzer = create_legal_authority_analyzer()
        
        # Phase 3 ranking weights - optimized for legal domain
        self.ranking_weights = {
            'semantic_similarity': 0.25,      # Reduced from 0.40 to make room for legal features
            'legal_analysis_depth': 0.20,     # Enhanced legal analysis
            'query_legal_relevance': 0.15,    # Legal concept alignment
            'authority_strength': 0.15,       # Precedent and regulatory authority
            'concept_density': 0.10,          # Legal concept density
            'document_expertise': 0.10,       # Legal expertise indicators
            'recency_bias': 0.05              # Temporal relevance
        }
        
        # Legal query type patterns for enhanced alignment
        self.legal_query_patterns = {
            'extraordinary_ability_analysis': {
                'patterns': ['extraordinary ability', 'sustained acclaim', 'top of field'],
                'boost_factor': 1.3,
                'relevant_concepts': ['extraordinary_ability_concepts']
            },
            'judging_criteria_analysis': {
                'patterns': ['participation as judge', 'judge of work', 'peer review'],
                'boost_factor': 1.2,
                'relevant_concepts': ['judging_concepts']
            },
            'awards_recognition_analysis': {
                'patterns': ['nationally recognized', 'internationally recognized', 'awards'],
                'boost_factor': 1.2,
                'relevant_concepts': ['awards_concepts']
            },
            'evidence_standards_analysis': {
                'patterns': ['burden of proof', 'preponderance', 'evidence'],
                'boost_factor': 1.1,
                'relevant_concepts': ['evidence_concepts', 'regulatory_concepts']
            },
            'procedural_analysis': {
                'patterns': ['appeal', 'determination', 'review'],
                'boost_factor': 1.0,
                'relevant_concepts': ['procedural_concepts']
            }
        }
    
    def rank_documents_legal_specialized(self, 
                                       query: str,
                                       candidates: List[Dict[str, Any]],
                                       query_embedding: Optional[Any] = None) -> List[LegalRankingResult]:
        """
        Perform legal specialized ranking with Phase 3 enhancements.
        
        Args:
            query: User query
            candidates: List of candidate documents
            query_embedding: Optional query embedding
            
        Returns:
            List of LegalRankingResult sorted by legal relevance
        """
        if not candidates:
            return []
        
        ranked_results = []
        query_type = self._classify_legal_query_type(query)
        
        logger.info(f"Legal specialized ranking for {len(candidates)} candidates, query type: {query_type}")
        
        for candidate in candidates:
            # Extract document data
            doc_id = candidate.get('id', '')
            doc_text = candidate.get('text', '')
            metadata = candidate.get('metadata', {})
            base_distance = candidate.get('distance', 1.0)
            
            # Perform comprehensive legal analysis
            legal_features = self._extract_legal_features(doc_text, metadata, query)
            
            # Calculate legal specialized ranking score
            legal_score = self._calculate_legal_ranking_score(
                query, query_type, legal_features, base_distance
            )
            
            # Generate ranking explanation
            explanation = self._generate_ranking_explanation(legal_features, query_type)
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(legal_features, query_type)
            
            result = LegalRankingResult(
                doc_id=doc_id,
                legal_ranking_score=legal_score,
                legal_features=legal_features,
                ranking_explanation=explanation,
                confidence_score=confidence
            )
            
            ranked_results.append(result)
        
        # Sort by legal ranking score
        ranked_results.sort(key=lambda x: x.legal_ranking_score, reverse=True)
        
        logger.info(f"Legal specialized ranking completed. Top score: {ranked_results[0].legal_ranking_score:.3f}" if ranked_results else "No results")
        
        return ranked_results
    
    def _extract_legal_features(self, doc_text: str, metadata: Dict, query: str) -> LegalRankingFeatures:
        """Extract comprehensive legal features for ranking."""
        # Document structure analysis
        doc_structure = self.doc_analyzer.analyze_document_structure(doc_text, metadata)
        
        # Legal concept extraction
        concept_extraction = self.concept_matcher.extract_legal_concepts(doc_text, metadata)
        
        # Precedent and authority analysis
        precedent_analysis = self.authority_analyzer.analyze_legal_authorities(doc_text, query)
        
        # Calculate derived scores
        legal_expertise_score = self._calculate_legal_expertise_score(
            doc_structure, concept_extraction, precedent_analysis
        )
        
        domain_specialization_score = self._calculate_domain_specialization_score(
            concept_extraction, precedent_analysis
        )
        
        query_legal_alignment = self._calculate_query_legal_alignment(
            query, concept_extraction, precedent_analysis
        )
        
        return LegalRankingFeatures(
            document_structure=doc_structure,
            concept_extraction=concept_extraction,
            precedent_analysis=precedent_analysis,
            legal_expertise_score=legal_expertise_score,
            domain_specialization_score=domain_specialization_score,
            query_legal_alignment=query_legal_alignment
        )
    
    def _calculate_legal_ranking_score(self, 
                                     query: str,
                                     query_type: str,
                                     features: LegalRankingFeatures,
                                     base_distance: float) -> float:
        """Calculate comprehensive legal ranking score."""
        
        # Base semantic similarity (converted from distance)
        semantic_score = max(0, 1 - base_distance)
        
        # Legal analysis depth score
        analysis_score = (
            features.document_structure.legal_complexity_score * 0.4 +
            features.document_structure.citation_density * 0.3 +
            features.document_structure.document_authority_score * 0.3
        )
        
        # Query-legal relevance score
        query_relevance_score = features.query_legal_alignment
        
        # Authority strength score
        authority_score = (
            features.precedent_analysis.precedent_strength_score * 0.5 +
            features.precedent_analysis.authority_diversity_score * 0.3 +
            features.precedent_analysis.recency_weighted_score * 0.2
        )
        
        # Concept density score
        concept_score = min(1.0, features.concept_extraction.concept_density_score / 10.0)
        
        # Document expertise score
        expertise_score = features.legal_expertise_score
        
        # Recency bias (enhanced with legal authority recency)
        recency_score = features.precedent_analysis.recency_weighted_score
        
        # Apply query type boost
        query_boost = self._get_query_type_boost(query, query_type, features)
        
        # Calculate weighted final score
        final_score = (
            semantic_score * self.ranking_weights['semantic_similarity'] +
            analysis_score * self.ranking_weights['legal_analysis_depth'] +
            query_relevance_score * self.ranking_weights['query_legal_relevance'] +
            authority_score * self.ranking_weights['authority_strength'] +
            concept_score * self.ranking_weights['concept_density'] +
            expertise_score * self.ranking_weights['document_expertise'] +
            recency_score * self.ranking_weights['recency_bias']
        )
        
        # Apply query type boost
        final_score *= query_boost
        
        return min(1.0, final_score)
    
    def _classify_legal_query_type(self, query: str) -> str:
        """Classify query into legal query types."""
        query_lower = query.lower()
        
        best_match = 'general'
        best_score = 0
        
        for query_type, type_data in self.legal_query_patterns.items():
            patterns = type_data['patterns']
            score = sum(1 for pattern in patterns if pattern in query_lower)
            
            if score > best_score:
                best_score = score
                best_match = query_type
        
        return best_match
    
    def _calculate_legal_expertise_score(self,
                                       doc_structure: DocumentStructureAnalysis,
                                       concept_extraction: ConceptExtractionResult,
                                       precedent_analysis: PrecedentAnalysis) -> float:
        """Calculate legal expertise score based on document features."""
        expertise_score = 0.0
        
        # Citation quality and quantity
        if doc_structure.case_citations:
            high_authority_citations = sum(
                1 for citation in doc_structure.case_citations
                if citation.confidence > 0.8
            )
            expertise_score += min(0.3, high_authority_citations * 0.1)
        
        # Regulatory citation sophistication
        if doc_structure.regulatory_citations:
            expertise_score += min(0.2, len(doc_structure.regulatory_citations) * 0.05)
        
        # Legal analysis depth
        if doc_structure.analysis_sections:
            expertise_score += min(0.2, len(doc_structure.analysis_sections) * 0.05)
        
        # Precedent sophistication
        if precedent_analysis.binding_authorities:
            expertise_score += min(0.15, len(precedent_analysis.binding_authorities) * 0.03)
        
        # Legal standards knowledge
        if doc_structure.legal_standards:
            expertise_score += min(0.15, len(doc_structure.legal_standards) * 0.02)
        
        return min(1.0, expertise_score)
    
    def _calculate_domain_specialization_score(self,
                                             concept_extraction: ConceptExtractionResult,
                                             precedent_analysis: PrecedentAnalysis) -> float:
        """Calculate immigration law domain specialization score."""
        specialization_score = 0.0
        
        # Extraordinary ability specialization
        ea_concepts = len(concept_extraction.extraordinary_ability_concepts)
        specialization_score += min(0.25, ea_concepts * 0.05)
        
        # Judging criteria specialization
        judging_concepts = len(concept_extraction.judging_concepts)
        specialization_score += min(0.25, judging_concepts * 0.05)
        
        # Awards/recognition specialization
        awards_concepts = len(concept_extraction.awards_concepts)
        specialization_score += min(0.20, awards_concepts * 0.04)
        
        # Evidence/regulatory specialization
        evidence_concepts = len(concept_extraction.evidence_concepts)
        regulatory_concepts = len(concept_extraction.regulatory_concepts)
        specialization_score += min(0.30, (evidence_concepts + regulatory_concepts) * 0.03)
        
        return min(1.0, specialization_score)
    
    def _calculate_query_legal_alignment(self,
                                       query: str,
                                       concept_extraction: ConceptExtractionResult,
                                       precedent_analysis: PrecedentAnalysis) -> float:
        """Calculate alignment between query and document legal content."""
        query_lower = query.lower()
        alignment_score = 0.0
        
        # Query term matching with legal concepts
        query_words = set(query_lower.split())
        
        # Check concept alignment
        all_concepts = (
            concept_extraction.extraordinary_ability_concepts +
            concept_extraction.judging_concepts +
            concept_extraction.awards_concepts +
            concept_extraction.evidence_concepts +
            concept_extraction.regulatory_concepts
        )
        
        for concept in all_concepts:
            concept_words = set(concept.matched_text.lower().split())
            overlap = len(query_words & concept_words)
            if overlap > 0:
                alignment_score += concept.confidence * (overlap / len(query_words))
        
        # Normalize by number of concepts
        if all_concepts:
            alignment_score = alignment_score / len(all_concepts)
        
        # Authority alignment
        for authority in precedent_analysis.binding_authorities + precedent_analysis.persuasive_authorities:
            if authority.subject_matter_relevance > 0.7:
                alignment_score += 0.1
        
        return min(1.0, alignment_score)
    
    def _get_query_type_boost(self, query: str, query_type: str, features: LegalRankingFeatures) -> float:
        """Get query type specific boost factor."""
        if query_type == 'general':
            return 1.0
        
        type_data = self.legal_query_patterns.get(query_type, {})
        base_boost = type_data.get('boost_factor', 1.0)
        relevant_concepts = type_data.get('relevant_concepts', [])
        
        # Check if document has relevant concepts for this query type
        concept_match_score = 0.0
        
        for concept_type in relevant_concepts:
            if concept_type == 'extraordinary_ability_concepts':
                concept_match_score += len(features.concept_extraction.extraordinary_ability_concepts) * 0.1
            elif concept_type == 'judging_concepts':
                concept_match_score += len(features.concept_extraction.judging_concepts) * 0.1
            elif concept_type == 'awards_concepts':
                concept_match_score += len(features.concept_extraction.awards_concepts) * 0.1
            elif concept_type == 'evidence_concepts':
                concept_match_score += len(features.concept_extraction.evidence_concepts) * 0.05
            elif concept_type == 'regulatory_concepts':
                concept_match_score += len(features.concept_extraction.regulatory_concepts) * 0.05
        
        # Apply boost based on concept match
        concept_boost = min(0.3, concept_match_score)
        return base_boost + concept_boost
    
    def _generate_ranking_explanation(self, features: LegalRankingFeatures, query_type: str) -> Dict[str, float]:
        """Generate explanation of ranking factors."""
        return {
            'legal_complexity': features.document_structure.legal_complexity_score,
            'citation_density': features.document_structure.citation_density,
            'document_authority': features.document_structure.document_authority_score,
            'precedent_strength': features.precedent_analysis.precedent_strength_score,
            'concept_density': features.concept_extraction.concept_density_score,
            'domain_relevance': features.concept_extraction.legal_domain_relevance,
            'expertise_level': features.legal_expertise_score,
            'query_alignment': features.query_legal_alignment,
            'query_type': query_type
        }
    
    def _calculate_confidence_score(self, features: LegalRankingFeatures, query_type: str) -> float:
        """Calculate confidence in the ranking result."""
        confidence_factors = [
            features.document_structure.legal_complexity_score,
            features.concept_extraction.legal_domain_relevance,
            features.precedent_analysis.precedent_strength_score,
            features.legal_expertise_score
        ]
        
        # Average confidence, weighted by number of strong indicators
        strong_indicators = sum(1 for factor in confidence_factors if factor > 0.7)
        base_confidence = sum(confidence_factors) / len(confidence_factors)
        
        # Boost confidence if multiple strong indicators
        if strong_indicators >= 3:
            base_confidence += 0.1
        elif strong_indicators >= 2:
            base_confidence += 0.05
        
        return min(1.0, base_confidence)


def create_legal_specialized_ranker() -> LegalSpecializedRanker:
    """Factory function to create a legal specialized ranker."""
    return LegalSpecializedRanker()


if __name__ == "__main__":
    # Test the legal specialized ranker
    ranker = create_legal_specialized_ranker()
    
    # Sample candidate documents for testing
    test_candidates = [
        {
            'id': 'doc_1',
            'text': '''
            The petitioner must demonstrate sustained national or international acclaim under 8 C.F.R. § 204.5(h)(3).
            Matter of Chawathe, 25 I&N Dec. 369 (AAO 2010) establishes the preponderance of evidence standard.
            The petitioner's participation as a judge on peer review panels demonstrates recognized expertise.
            See Kazarian v. USCIS, 596 F.3d 1115 (9th Cir. 2010) regarding the two-part analysis.
            ''',
            'metadata': {'document_id': 'test_doc_1'},
            'distance': 0.3
        },
        {
            'id': 'doc_2',
            'text': '''
            The Director determined that petitioner satisfied the awards criterion at 8 C.F.R. § 204.5(h)(3)(i).
            Evidence showed nationally recognized prizes for excellence in the field of endeavor.
            The final merits determination considers the totality of evidence under Kazarian.
            ''',
            'metadata': {'document_id': 'test_doc_2'},
            'distance': 0.4
        }
    ]
    
    test_query = "How do AAO decisions evaluate participation as a judge criteria?"
    
    print("Testing Legal Specialized Ranker")
    print("=" * 50)
    
    results = ranker.rank_documents_legal_specialized(test_query, test_candidates)
    
    for i, result in enumerate(results, 1):
        print(f"\nRank {i}: {result.doc_id}")
        print(f"Legal Ranking Score: {result.legal_ranking_score:.3f}")
        print(f"Confidence: {result.confidence_score:.3f}")
        print(f"Key Factors:")
        for factor, score in result.ranking_explanation.items():
            if isinstance(score, (int, float)) and score > 0.1:
                print(f"  {factor}: {score:.3f}")
        
        print(f"Legal Features Summary:")
        print(f"  Case Citations: {len(result.legal_features.document_structure.case_citations)}")
        print(f"  Regulatory Citations: {len(result.legal_features.document_structure.regulatory_citations)}")
        print(f"  EA Concepts: {len(result.legal_features.concept_extraction.extraordinary_ability_concepts)}")
        print(f"  Judging Concepts: {len(result.legal_features.concept_extraction.judging_concepts)}")
        print(f"  Binding Authorities: {len(result.legal_features.precedent_analysis.binding_authorities)}")