"""
Enhanced query processing for improved RAG retrieval performance.
"""

import re
from typing import List, Dict

class QueryEnhancer:
    """Enhances queries for better retrieval performance in legal RAG systems."""
    
    def __init__(self):
        # Legal concept hierarchies - implementing Phase 1 enhancement
        self.legal_concept_hierarchy = {
            'judging': {
                'parent_concepts': ['evaluation', 'assessment', 'review', 'examination'],
                'child_concepts': ['peer_review', 'expert_panel', 'selection_committee', 'editorial_board'],
                'related_procedures': ['criteria_evaluation', 'expert_determination', 'merit_assessment'],
                'outcome_indicators': ['qualified', 'recognized', 'distinguished', 'selected', 'appointed'],
                'activity_types': ['manuscript_review', 'grant_evaluation', 'conference_program', 'award_selection']
            },
            'awards': {
                'parent_concepts': ['recognition', 'achievement', 'honor', 'distinction'],
                'child_concepts': ['national_award', 'international_prize', 'professional_honor', 'academic_recognition'],
                'quality_indicators': ['prestigious', 'competitive', 'selective', 'exclusive', 'distinguished'],
                'scope_indicators': ['national', 'international', 'field-specific', 'cross-disciplinary'],
                'selection_criteria': ['merit_based', 'peer_nominated', 'expert_selected', 'competitive_process']
            },
            'sustained_acclaim': {
                'parent_concepts': ['continued_recognition', 'ongoing_acknowledgment', 'persistent_acclaim'],
                'temporal_indicators': ['sustained', 'continued', 'ongoing', 'persistent', 'enduring'],
                'evidence_types': ['citations', 'media_coverage', 'expert_testimony', 'industry_recognition'],
                'scope_qualifiers': ['national', 'international', 'field-wide', 'cross-sector']
            },
            'extraordinary_ability': {
                'defining_concepts': ['exceptional_skill', 'outstanding_achievement', 'superior_performance'],
                'evidence_categories': ['original_contributions', 'leadership_role', 'critical_role'],
                'comparison_standards': ['small_percentage', 'top_of_field', 'national_prominence'],
                'regulatory_framework': ['8_cfr_204_5_h_3', 'regulatory_criteria', 'evidentiary_standards']
            }
        }
        
        # Enhanced legal domain-specific term expansions
        self.legal_synonyms = {
            'extraordinary ability': ['exceptional ability', 'outstanding ability', 'exceptional skill'],
            'sustained acclaim': ['continued recognition', 'ongoing recognition', 'persistent acclaim'],
            'national awards': ['national recognition', 'national prizes', 'national honors'],
            'international awards': ['international recognition', 'international prizes', 'worldwide recognition'],
            'participation as judge': ['judging role', 'evaluation panel', 'review panel', 'assessment panel'],
            'criteria': ['requirements', 'standards', 'qualifications', 'conditions'],
            'evaluate': ['assess', 'review', 'consider', 'examine', 'analyze'],
            'applicant': ['petitioner', 'beneficiary', 'candidate']
        }
        
        # Common legal abbreviations and their expansions
        self.abbreviations = {
            'AAO': 'Administrative Appeals Office',
            'USCIS': 'United States Citizenship and Immigration Services',
            'I-140': 'Immigrant Petition for Alien Worker'
        }
    
    def enhance_query(self, query: str) -> Dict[str, str]:
        """
        Enhanced query processing with legal concept hierarchies.
        
        Returns:
            Dict with original query, enhanced query, and hierarchical expansions
        """
        original_query = query.strip()
        
        # Apply hierarchical concept expansion first
        hierarchical_enhanced = self._apply_concept_hierarchy_expansion(original_query)
        
        # Then apply traditional synonym expansion
        enhanced_query = self._expand_legal_terms(hierarchical_enhanced)
        enhanced_query = self._add_context_keywords(enhanced_query)
        
        return {
            'original': original_query,
            'enhanced': enhanced_query,
            'hierarchical_enhanced': hierarchical_enhanced,
            'search_terms': self._extract_key_terms(enhanced_query),
            'concept_expansions': self._get_concept_expansions(original_query)
        }
    
    def _expand_legal_terms(self, query: str) -> str:
        """Expand legal terms with synonyms for better matching."""
        enhanced = query.lower()
        
        # Expand legal synonyms
        for term, synonyms in self.legal_synonyms.items():
            if term in enhanced:
                # Add synonyms as alternative terms
                synonym_phrase = ' OR '.join([f'"{syn}"' for syn in synonyms])
                enhanced = enhanced.replace(term, f'{term} OR {synonym_phrase}')
        
        return enhanced
    
    def _add_context_keywords(self, query: str) -> str:
        """Add relevant context keywords based on query type."""
        context_keywords = []
        query_lower = query.lower()
        
        if 'judge' in query_lower or 'judging' in query_lower:
            context_keywords.extend(['peer review', 'evaluation panel', 'selection committee', 
                                   'participation judge', 'judging role', 'tribunal', 'review board'])
        
        if 'award' in query_lower:
            context_keywords.extend(['recognition', 'achievement', 'excellence', 'honor', 
                                   'prize', 'distinction', 'acclaim', 'prestigious'])
        
        if 'criteria' in query_lower or 'requirement' in query_lower:
            context_keywords.extend(['standard', 'qualification', 'evidence', 'benchmark',
                                   'requirement', 'condition', 'prerequisite'])
        
        if 'sustained acclaim' in query_lower:
            context_keywords.extend(['continuous recognition', 'ongoing acclaim', 'persistent recognition',
                                   'maintained excellence', 'consistent achievement'])
        
        if context_keywords:
            query = f"{query} {' '.join(context_keywords)}"
        
        return query
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms for focused search."""
        # Remove common stop words but keep legal-specific terms
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Extract meaningful terms (2+ characters, not stop words)
        terms = re.findall(r'\b\w{2,}\b', query.lower())
        key_terms = [term for term in terms if term not in stop_words]
        
        return key_terms[:10]  # Limit to top 10 terms
    
    def _apply_concept_hierarchy_expansion(self, query: str) -> str:
        """Apply legal concept hierarchy expansion to query."""
        query_lower = query.lower()
        expanded_terms = []
        
        # Identify relevant concept categories
        for concept_key, concept_data in self.legal_concept_hierarchy.items():
            if self._query_matches_concept(query_lower, concept_key):
                # Add hierarchical expansions
                for category, terms in concept_data.items():
                    if isinstance(terms, list):
                        # Add relevant terms from this category
                        relevant_terms = self._select_relevant_terms(query_lower, terms, category)
                        expanded_terms.extend(relevant_terms)
        
        if expanded_terms:
            # Combine original query with expanded terms
            expanded_query = f"{query} {' '.join(expanded_terms[:10])}"  # Limit expansion
            return expanded_query
        
        return query
    
    def _query_matches_concept(self, query_lower: str, concept_key: str) -> bool:
        """Check if query matches a concept category."""
        concept_indicators = {
            'judging': ['judge', 'judging', 'evaluation', 'panel', 'review'],
            'awards': ['award', 'recognition', 'prize', 'honor', 'acclaim'],
            'sustained_acclaim': ['sustained', 'acclaim', 'ongoing', 'continued'],
            'extraordinary_ability': ['extraordinary', 'exceptional', 'outstanding', 'ability']
        }
        
        indicators = concept_indicators.get(concept_key, [])
        return any(indicator in query_lower for indicator in indicators)
    
    def _select_relevant_terms(self, query_lower: str, terms: List[str], category: str) -> List[str]:
        """Select most relevant terms from a category based on query context."""
        # Weight terms based on category importance and query context
        weights = {
            'parent_concepts': 0.8,
            'child_concepts': 0.9,
            'related_procedures': 0.7,
            'outcome_indicators': 0.6,
            'quality_indicators': 0.8,
            'scope_indicators': 0.7,
            'temporal_indicators': 0.8,
            'evidence_types': 0.6
        }
        
        weight = weights.get(category, 0.5)
        
        # Select terms that don't overlap too much with existing query
        relevant_terms = []
        for term in terms:
            term_clean = term.replace('_', ' ')
            # Only add if not already substantially in query and weight is high enough
            if weight >= 0.7 and not any(word in query_lower for word in term_clean.split()):
                relevant_terms.append(term_clean)
        
        return relevant_terms[:3]  # Limit per category
    
    def _get_concept_expansions(self, query: str) -> Dict[str, List[str]]:
        """Get detailed concept expansions for analysis."""
        query_lower = query.lower()
        expansions = {}
        
        for concept_key, concept_data in self.legal_concept_hierarchy.items():
            if self._query_matches_concept(query_lower, concept_key):
                expansions[concept_key] = {}
                for category, terms in concept_data.items():
                    if isinstance(terms, list):
                        relevant = self._select_relevant_terms(query_lower, terms, category)
                        if relevant:
                            expansions[concept_key][category] = relevant
        
        return expansions
    
    def multi_round_query_expansion(self, original_query: str, max_rounds: int = 3) -> Dict[str, any]:
        """
        Multi-round query expansion as specified in RAG enhancement strategy.
        """
        expanded_queries = []
        
        # Round 1: Legal domain expansion with hierarchies
        round1_enhanced = self.enhance_query(original_query)
        legal_expanded = round1_enhanced['hierarchical_enhanced']
        expanded_queries.append(('legal_domain', legal_expanded, 1.0))
        
        # Round 2: Contextual expansion (simulated - in real implementation, 
        # this would use initial retrieval results)
        context_terms = self._extract_contextual_terms(original_query)
        if context_terms:
            context_expanded = f"{legal_expanded} {' '.join(context_terms)}"
            expanded_queries.append(('contextual', context_expanded, 0.8))
        
        # Round 3: Semantic expansion (simplified version)
        semantic_terms = self._get_semantic_neighbors(original_query)
        if semantic_terms:
            semantic_expanded = f"{legal_expanded} {' '.join(semantic_terms)}"
            expanded_queries.append(('semantic', semantic_expanded, 0.6))
        
        return {
            'original_query': original_query,
            'expanded_queries': expanded_queries,
            'concept_expansions': round1_enhanced['concept_expansions']
        }
    
    def _extract_contextual_terms(self, query: str) -> List[str]:
        """Extract contextual terms based on query type."""
        query_lower = query.lower()
        contextual_terms = []
        
        if 'judge' in query_lower:
            contextual_terms.extend(['expertise', 'qualifications', 'experience', 'credentials'])
        elif 'award' in query_lower:
            contextual_terms.extend(['significance', 'criteria', 'selection', 'prestige'])
        elif 'criteria' in query_lower:
            contextual_terms.extend(['evidence', 'documentation', 'standards', 'requirements'])
        
        return contextual_terms[:3]
    
    def _get_semantic_neighbors(self, query: str) -> List[str]:
        """Get semantic neighbors (simplified implementation)."""
        # In a full implementation, this would use embedding similarity
        semantic_mapping = {
            'evaluation': ['analysis', 'assessment', 'determination', 'consideration'],
            'recognition': ['acknowledgment', 'acclaim', 'distinction', 'honor'],
            'expertise': ['knowledge', 'skill', 'competence', 'proficiency'],
            'achievement': ['accomplishment', 'success', 'attainment', 'contribution']
        }
        
        query_lower = query.lower()
        neighbors = []
        
        for key, values in semantic_mapping.items():
            if key in query_lower:
                neighbors.extend(values[:2])  # Take top 2 neighbors
        
        return neighbors[:4]  # Limit total semantic neighbors
    
    def create_multi_query_variants(self, query: str) -> List[str]:
        """Create multiple query variants for comprehensive search."""
        variants = [query]
        
        # Create focused variants by extracting key phrases
        if 'how do' in query.lower():
            # Convert "How do..." to statement form
            focused = re.sub(r'how do .+ (evaluate|assess|consider)', r'\1', query, flags=re.IGNORECASE)
            variants.append(focused)
        
        if 'what characteristics' in query.lower():
            # Focus on characteristics
            variants.append(query.replace('What characteristics of', '').strip())
        
        # Add question-focused variant
        key_terms = self._extract_key_terms(query)
        if len(key_terms) >= 3:
            variants.append(' '.join(key_terms[:5]))
        
        return list(set(variants))  # Remove duplicates


def enhance_retrieval_query(query: str) -> Dict[str, any]:
    """
    Enhanced query processing for better retrieval with Phase 1 improvements.
    
    Args:
        query: Original user query
        
    Returns:
        Dict with enhanced query information including multi-round expansion
    """
    enhancer = QueryEnhancer()
    enhanced_info = enhancer.enhance_query(query)
    variants = enhancer.create_multi_query_variants(query)
    multi_round_expansion = enhancer.multi_round_query_expansion(query)
    
    return {
        'original_query': enhanced_info['original'],
        'enhanced_query': enhanced_info['enhanced'],
        'hierarchical_enhanced': enhanced_info['hierarchical_enhanced'],
        'search_terms': enhanced_info['search_terms'],
        'query_variants': variants,
        'concept_expansions': enhanced_info['concept_expansions'],
        'multi_round_expansion': multi_round_expansion
    }


if __name__ == "__main__":
    # Test query enhancement
    test_queries = [
        "How do recent AAO decisions evaluate an applicant's Participation as a Judge service criteria?",
        "What characteristics of national or international awards persuade the AAO that they constitute sustained acclaim?"
    ]
    
    enhancer = QueryEnhancer()
    
    for query in test_queries:
        print(f"Original: {query}")
        enhanced_info = enhance_retrieval_query(query)
        print(f"Enhanced: {enhanced_info['enhanced_query']}")
        print(f"Key terms: {enhanced_info['search_terms']}")
        print(f"Variants: {enhanced_info['query_variants']}")
        print("-" * 80)