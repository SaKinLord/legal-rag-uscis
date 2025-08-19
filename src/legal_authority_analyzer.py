"""
Legal Authority and Precedent Analysis for Phase 3 specialization.
Implements case law precedent analysis, regulatory citation weighting,
and authority scoring for enhanced legal document ranking.
"""

import re
import math
from datetime import datetime
from typing import Dict, List, Tuple, Any, Set, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class LegalAuthority:
    """Structured representation of legal authority."""
    authority_type: str
    citation: str
    authority_level: int  # 1=Supreme Court, 2=Circuit Court, 3=BIA, 4=AAO, 5=Other
    binding_weight: float
    persuasive_weight: float
    recency_factor: float
    subject_matter_relevance: float
    precedential_value: float


@dataclass
class PrecedentAnalysis:
    """Results of precedent analysis."""
    binding_authorities: List[LegalAuthority]
    persuasive_authorities: List[LegalAuthority]
    regulatory_authorities: List[LegalAuthority]
    precedent_strength_score: float
    authority_diversity_score: float
    recency_weighted_score: float


class LegalAuthorityAnalyzer:
    """
    Advanced legal authority and precedent analysis system.
    Implements sophisticated authority weighting and precedent evaluation.
    """
    
    def __init__(self):
        # Authority hierarchy with binding/persuasive weights
        self.authority_hierarchy = {
            'us_supreme_court': {
                'level': 1,
                'binding_weight': 1.0,
                'persuasive_weight': 1.0,
                'base_precedential_value': 1.0,
                'patterns': [
                    r'(\d+)\s+U\.S\.?\s+(\d+)',
                    r'(\d+)\s+S\.?\s?Ct\.?\s+(\d+)'
                ]
            },
            'circuit_court': {
                'level': 2,
                'binding_weight': 0.9,  # Binding in circuit
                'persuasive_weight': 0.8,  # Persuasive outside circuit
                'base_precedential_value': 0.85,
                'patterns': [
                    r'(\d+)\s+F\.(\d)d\s+(\d+)(?:\s+\(([^)]+)\s+(\d{4})\))?',
                    r'(\d+)\s+F\.?\s+Supp\.?\s+(\d)d\s+(\d+)'
                ]
            },
            'bia_decisions': {
                'level': 3,
                'binding_weight': 0.95,  # Binding for immigration matters
                'persuasive_weight': 0.9,
                'base_precedential_value': 0.9,
                'patterns': [
                    r'Matter\s+of\s+([A-Z][a-z\-\'\s]+),?\s+(\d+)\s+I&N\s+Dec\.?\s+(\d+)',
                    r'In\s+re:?\s+([A-Z][a-z\-\'\s]+),?\s+(\d+)\s+I&N\s+Dec\.?\s+(\d+)'
                ]
            },
            'aao_decisions': {
                'level': 4,
                'binding_weight': 0.6,  # Not binding but highly persuasive
                'persuasive_weight': 0.8,
                'base_precedential_value': 0.7,
                'patterns': [
                    r'Matter\s+of\s+([A-Z][a-z\-\'\s]+).*\(AAO\s+(\d{4})\)',
                    r'In\s+Re:?\s+(\d+)'
                ]
            },
            'district_court': {
                'level': 5,
                'binding_weight': 0.3,  # Limited binding effect
                'persuasive_weight': 0.5,
                'base_precedential_value': 0.5,
                'patterns': [
                    r'(\d+)\s+F\.?\s+Supp\.?\s+(\d)d\s+(\d+)(?:\s+\(([^)]+)\s+(\d{4})\))?'
                ]
            }
        }
        
        # Regulatory authority weights
        self.regulatory_authorities = {
            'cfr_primary': {
                'authority_level': 1,
                'binding_weight': 1.0,
                'precedential_value': 1.0,
                'patterns': [
                    r'8\s+C\.F\.R\.?\s+§?\s*(204\.5\(h\))',  # Primary EB-1A regulation
                    r'8\s+C\.F\.R\.?\s+§?\s*(204\.5)',       # General EB regulations
                ]
            },
            'usc_immigration': {
                'authority_level': 1,
                'binding_weight': 1.0,
                'precedential_value': 1.0,
                'patterns': [
                    r'8\s+U\.S\.C\.?\s+§?\s*(1153)',        # Immigration and Nationality Act
                    r'INA\s+§?\s*(203)'
                ]
            },
            'policy_manual': {
                'authority_level': 2,
                'binding_weight': 0.8,
                'precedential_value': 0.7,
                'patterns': [
                    r'(\d+)\s+USCIS\s+Policy\s+Manual\s+([A-Z])\.(\d+)',
                    r'USCIS\s+Policy\s+Manual\s+([A-Z])\.(\d+)'
                ]
            },
            'federal_register': {
                'authority_level': 3,
                'binding_weight': 0.6,
                'precedential_value': 0.6,
                'patterns': [
                    r'(\d+)\s+Fed\.?\s+Reg\.?\s+(\d+)'
                ]
            }
        }
        
        # Subject matter relevance indicators
        self.subject_matter_indicators = {
            'extraordinary_ability': [
                'extraordinary ability', 'sustained acclaim', 'national acclaim',
                'international acclaim', 'top of field', 'small percentage'
            ],
            'judging_participation': [
                'participation as judge', 'judge of work', 'peer review',
                'editorial board', 'selection committee'
            ],
            'awards_recognition': [
                'nationally recognized', 'internationally recognized',
                'prizes or awards', 'excellence in field'
            ],
            'evidence_standards': [
                'preponderance of evidence', 'burden of proof',
                'final merits determination', 'totality of evidence'
            ]
        }
        
        # Key precedent cases with enhanced analysis
        self.key_precedents = {
            'kazarian': {
                'full_citation': 'Kazarian v. USCIS, 596 F.3d 1115 (9th Cir. 2010)',
                'authority_level': 2,
                'precedential_value': 0.95,
                'subject_matters': ['extraordinary_ability', 'evidence_standards'],
                'key_holdings': [
                    'two-part analysis required',
                    'final merits determination',
                    'totality of evidence consideration'
                ]
            },
            'chawathe': {
                'full_citation': 'Matter of Chawathe, 25 I&N Dec. 369 (AAO 2010)',
                'authority_level': 4,
                'precedential_value': 0.85,
                'subject_matters': ['evidence_standards'],
                'key_holdings': [
                    'preponderance of evidence standard',
                    'petitioner burden of proof'
                ]
            },
            'visinscaia': {
                'full_citation': 'Visinscaia v. Beers, 4 F. Supp. 3d 126 (D.D.C. 2013)',
                'authority_level': 5,
                'precedential_value': 0.6,
                'subject_matters': ['extraordinary_ability'],
                'key_holdings': [
                    'two-part review process'
                ]
            }
        }
    
    def analyze_legal_authorities(self, text: str, query_context: Optional[str] = None) -> PrecedentAnalysis:
        """
        Analyze legal authorities and precedents in document text.
        
        Args:
            text: Document text to analyze
            query_context: Optional query context for relevance weighting
            
        Returns:
            PrecedentAnalysis with detailed authority analysis
        """
        # Extract case law authorities
        case_authorities = self._extract_case_authorities(text, query_context)
        
        # Extract regulatory authorities
        regulatory_authorities = self._extract_regulatory_authorities(text, query_context)
        
        # Categorize authorities
        binding_authorities = []
        persuasive_authorities = []
        
        for authority in case_authorities:
            if authority.binding_weight >= 0.8:
                binding_authorities.append(authority)
            else:
                persuasive_authorities.append(authority)
        
        # Calculate composite scores
        precedent_strength = self._calculate_precedent_strength(case_authorities, regulatory_authorities)
        authority_diversity = self._calculate_authority_diversity(case_authorities, regulatory_authorities)
        recency_weighted = self._calculate_recency_weighted_score(case_authorities)
        
        return PrecedentAnalysis(
            binding_authorities=binding_authorities,
            persuasive_authorities=persuasive_authorities,
            regulatory_authorities=regulatory_authorities,
            precedent_strength_score=precedent_strength,
            authority_diversity_score=authority_diversity,
            recency_weighted_score=recency_weighted
        )
    
    def _extract_case_authorities(self, text: str, query_context: Optional[str] = None) -> List[LegalAuthority]:
        """Extract and analyze case law authorities."""
        authorities = []
        
        for auth_type, auth_data in self.authority_hierarchy.items():
            patterns = auth_data['patterns']
            
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    # Calculate subject matter relevance
                    subject_relevance = self._calculate_subject_matter_relevance(
                        match, text, query_context
                    )
                    
                    # Calculate recency factor
                    recency_factor = self._extract_recency_factor(match, text)
                    
                    # Calculate precedential value
                    precedential_value = self._calculate_precedential_value(
                        match, auth_data, subject_relevance, recency_factor
                    )
                    
                    authority = LegalAuthority(
                        authority_type=auth_type,
                        citation=match.group(0),
                        authority_level=auth_data['level'],
                        binding_weight=auth_data['binding_weight'],
                        persuasive_weight=auth_data['persuasive_weight'],
                        recency_factor=recency_factor,
                        subject_matter_relevance=subject_relevance,
                        precedential_value=precedential_value
                    )
                    authorities.append(authority)
        
        # Check for key precedents
        authorities.extend(self._identify_key_precedents(text, query_context))
        
        return self._deduplicate_authorities(authorities)
    
    def _extract_regulatory_authorities(self, text: str, query_context: Optional[str] = None) -> List[LegalAuthority]:
        """Extract and analyze regulatory authorities."""
        authorities = []
        
        for auth_type, auth_data in self.regulatory_authorities.items():
            patterns = auth_data['patterns']
            
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    # Regulatory authorities are always current (high recency)
                    recency_factor = 1.0
                    
                    # Calculate subject matter relevance
                    subject_relevance = self._calculate_subject_matter_relevance(
                        match, text, query_context
                    )
                    
                    authority = LegalAuthority(
                        authority_type=auth_type,
                        citation=match.group(0),
                        authority_level=auth_data['authority_level'],
                        binding_weight=auth_data['binding_weight'],
                        persuasive_weight=auth_data['binding_weight'],  # Same for regulations
                        recency_factor=recency_factor,
                        subject_matter_relevance=subject_relevance,
                        precedential_value=auth_data['precedential_value']
                    )
                    authorities.append(authority)
        
        return self._deduplicate_authorities(authorities)
    
    def _calculate_subject_matter_relevance(self, 
                                          match: re.Match, 
                                          text: str, 
                                          query_context: Optional[str] = None) -> float:
        """Calculate subject matter relevance of an authority."""
        relevance_score = 0.5  # Base relevance
        
        # Get surrounding context
        start = max(0, match.start() - 200)
        end = min(len(text), match.end() + 200)
        context = text[start:end].lower()
        
        # Check against subject matter indicators
        max_relevance = 0.0
        for subject_matter, indicators in self.subject_matter_indicators.items():
            subject_score = 0.0
            for indicator in indicators:
                if indicator in context:
                    subject_score += 0.1
            
            # Normalize subject score
            subject_score = min(1.0, subject_score)
            max_relevance = max(max_relevance, subject_score)
        
        relevance_score += max_relevance * 0.4
        
        # Query context relevance
        if query_context:
            query_lower = query_context.lower()
            context_match_score = 0.0
            
            # Check for query terms in citation context
            query_words = set(query_lower.split())
            context_words = set(context.split())
            
            overlap = len(query_words & context_words)
            if len(query_words) > 0:
                context_match_score = overlap / len(query_words)
            
            relevance_score += context_match_score * 0.1
        
        return min(1.0, relevance_score)
    
    def _extract_recency_factor(self, match: re.Match, text: str) -> float:
        """Extract and calculate recency factor for a citation."""
        # Look for year in the citation or surrounding text
        citation_text = match.group(0)
        
        # Try to extract year from citation
        year_pattern = r'\((\d{4})\)'
        year_match = re.search(year_pattern, citation_text)
        
        if not year_match:
            # Look in surrounding context
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end]
            year_match = re.search(year_pattern, context)
        
        if year_match:
            try:
                year = int(year_match.group(1))
                current_year = datetime.now().year
                years_old = current_year - year
                
                # Calculate recency factor (1.0 for current year, decreasing with age)
                if years_old <= 0:
                    return 1.0
                elif years_old <= 2:
                    return 0.9
                elif years_old <= 5:
                    return 0.8
                elif years_old <= 10:
                    return 0.6
                elif years_old <= 20:
                    return 0.4
                else:
                    return 0.2
            except ValueError:
                pass
        
        return 0.7  # Default recency factor for unknown dates
    
    def _calculate_precedential_value(self, 
                                    match: re.Match,
                                    auth_data: Dict,
                                    subject_relevance: float,
                                    recency_factor: float) -> float:
        """Calculate overall precedential value."""
        base_value = auth_data['base_precedential_value']
        
        # Combine factors
        precedential_value = (
            base_value * 0.5 +
            subject_relevance * 0.3 +
            recency_factor * 0.2
        )
        
        return min(1.0, precedential_value)
    
    def _identify_key_precedents(self, text: str, query_context: Optional[str] = None) -> List[LegalAuthority]:
        """Identify key precedent cases with enhanced analysis."""
        authorities = []
        
        for precedent_key, precedent_data in self.key_precedents.items():
            # Create flexible patterns for key precedents
            case_name_pattern = precedent_key.title()
            
            # Look for various citation formats
            patterns = [
                rf'{case_name_pattern}\s+v\.?\s+[A-Z][a-z]+',
                rf'Matter\s+of\s+{case_name_pattern}',
                rf'{case_name_pattern}.*\d+\s+I&N\s+Dec',
                rf'{case_name_pattern}.*\d+\s+F\.\d+d'
            ]
            
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    # Calculate subject matter relevance for key precedents
                    subject_relevance = 0.8  # Base high relevance for key cases
                    
                    # Check if query context matches precedent subject matters
                    if query_context:
                        query_lower = query_context.lower()
                        for subject_matter in precedent_data['subject_matters']:
                            indicators = self.subject_matter_indicators.get(subject_matter, [])
                            for indicator in indicators:
                                if indicator in query_lower:
                                    subject_relevance = min(1.0, subject_relevance + 0.1)
                    
                    authority = LegalAuthority(
                        authority_type=f'key_precedent_{precedent_key}',
                        citation=precedent_data['full_citation'],
                        authority_level=precedent_data['authority_level'],
                        binding_weight=0.9,  # Key precedents have high weight
                        persuasive_weight=0.95,
                        recency_factor=0.8,  # Established precedents retain value
                        subject_matter_relevance=subject_relevance,
                        precedential_value=precedent_data['precedential_value']
                    )
                    authorities.append(authority)
        
        return authorities
    
    def _deduplicate_authorities(self, authorities: List[LegalAuthority]) -> List[LegalAuthority]:
        """Remove duplicate authorities based on citation similarity."""
        seen_citations = set()
        unique_authorities = []
        
        for authority in authorities:
            # Create normalized citation key
            citation_key = re.sub(r'\s+', ' ', authority.citation.lower().strip())
            citation_key = re.sub(r'[^\w\s]', '', citation_key)
            
            if citation_key not in seen_citations:
                seen_citations.add(citation_key)
                unique_authorities.append(authority)
        
        return unique_authorities
    
    def _calculate_precedent_strength(self, 
                                    case_authorities: List[LegalAuthority],
                                    regulatory_authorities: List[LegalAuthority]) -> float:
        """Calculate overall precedent strength score."""
        if not case_authorities and not regulatory_authorities:
            return 0.0
        
        total_strength = 0.0
        total_weight = 0.0
        
        # Weight case authorities
        for authority in case_authorities:
            strength_contribution = (
                authority.precedential_value * 0.4 +
                authority.binding_weight * 0.3 +
                authority.subject_matter_relevance * 0.2 +
                authority.recency_factor * 0.1
            )
            
            # Higher level authorities get more weight
            authority_weight = 1.0 / authority.authority_level
            total_strength += strength_contribution * authority_weight
            total_weight += authority_weight
        
        # Weight regulatory authorities
        for authority in regulatory_authorities:
            strength_contribution = authority.binding_weight * authority.subject_matter_relevance
            authority_weight = 1.0 / authority.authority_level
            total_strength += strength_contribution * authority_weight
            total_weight += authority_weight
        
        return total_strength / total_weight if total_weight > 0 else 0.0
    
    def _calculate_authority_diversity(self, 
                                     case_authorities: List[LegalAuthority],
                                     regulatory_authorities: List[LegalAuthority]) -> float:
        """Calculate authority diversity score."""
        authority_types = set()
        authority_levels = set()
        
        for authority in case_authorities + regulatory_authorities:
            authority_types.add(authority.authority_type)
            authority_levels.add(authority.authority_level)
        
        # Diversity based on variety of authority types and levels
        type_diversity = min(1.0, len(authority_types) / 6.0)  # Up to 6 major types
        level_diversity = min(1.0, len(authority_levels) / 5.0)  # Up to 5 levels
        
        return (type_diversity + level_diversity) / 2.0
    
    def _calculate_recency_weighted_score(self, case_authorities: List[LegalAuthority]) -> float:
        """Calculate recency-weighted authority score."""
        if not case_authorities:
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for authority in case_authorities:
            weight = authority.precedential_value * authority.recency_factor
            weighted_sum += authority.recency_factor * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0


def create_legal_authority_analyzer() -> LegalAuthorityAnalyzer:
    """Factory function to create a legal authority analyzer."""
    return LegalAuthorityAnalyzer()


if __name__ == "__main__":
    # Test the legal authority analyzer
    analyzer = create_legal_authority_analyzer()
    
    # Sample legal text for testing
    test_text = """
    The petitioner bears the burden of proof to demonstrate eligibility by a preponderance of the evidence.
    Matter of Chawathe, 25 I&N Dec. 369, 375-76 (AAO 2010). We review the questions in this matter
    de novo. See Kazarian v. USCIS, 596 F.3d 1115 (9th Cir. 2010) (discussing a two-part review).
    
    Under 8 C.F.R. § 204.5(h)(3)(iv), evidence of the alien's participation as a judge of the work of others
    is required. This regulation implements the statutory requirements found in 8 U.S.C. § 1153(b)(1)(A).
    
    The USCIS Policy Manual F.2(B)(1) provides additional guidance on evaluating nationally recognized awards.
    See also Visinscaia v. Beers, 4 F. Supp. 3d 126, 131-32 (D.D.C. 2013).
    """
    
    query_context = "How do AAO decisions evaluate participation as a judge criteria?"
    
    print("Testing Legal Authority Analyzer")
    print("=" * 50)
    
    analysis = analyzer.analyze_legal_authorities(test_text, query_context)
    
    print(f"Binding Authorities: {len(analysis.binding_authorities)}")
    for authority in analysis.binding_authorities:
        print(f"  - {authority.authority_type}: {authority.citation} (value: {authority.precedential_value:.3f})")
    
    print(f"\nPersuasive Authorities: {len(analysis.persuasive_authorities)}")
    for authority in analysis.persuasive_authorities:
        print(f"  - {authority.authority_type}: {authority.citation} (value: {authority.precedential_value:.3f})")
    
    print(f"\nRegulatory Authorities: {len(analysis.regulatory_authorities)}")
    for authority in analysis.regulatory_authorities:
        print(f"  - {authority.authority_type}: {authority.citation} (weight: {authority.binding_weight:.3f})")
    
    print(f"\nPrecedent Strength Score: {analysis.precedent_strength_score:.3f}")
    print(f"Authority Diversity Score: {analysis.authority_diversity_score:.3f}")
    print(f"Recency Weighted Score: {analysis.recency_weighted_score:.3f}")