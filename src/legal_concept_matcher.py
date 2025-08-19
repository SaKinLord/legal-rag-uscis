"""
Advanced Legal Concept Extraction and Matching for Phase 3 specialization.
Implements sophisticated legal concept recognition, regulatory pattern matching,
and domain-specific semantic understanding for enhanced legal RAG systems.
"""

import re
import math
from typing import Dict, List, Tuple, Any, Set, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class LegalConcept:
    """Structured representation of a legal concept match."""
    concept_type: str
    concept_name: str
    matched_text: str
    confidence: float
    context: str
    position: int
    regulatory_basis: Optional[str] = None
    precedent_support: Optional[str] = None


@dataclass
class ConceptExtractionResult:
    """Results of legal concept extraction."""
    extraordinary_ability_concepts: List[LegalConcept]
    judging_concepts: List[LegalConcept]
    awards_concepts: List[LegalConcept]
    evidence_concepts: List[LegalConcept]
    regulatory_concepts: List[LegalConcept]
    procedural_concepts: List[LegalConcept]
    concept_density_score: float
    legal_domain_relevance: float


class LegalConceptMatcher:
    """
    Advanced legal concept extraction and matching system.
    Implements Phase 3 domain specialization for enhanced legal understanding.
    """
    
    def __init__(self):
        # Extraordinary Ability Concepts - Enhanced with regulatory specificity
        self.extraordinary_ability_concepts = {
            'sustained_acclaim': {
                'patterns': [
                    r'sustained\s+national\s+or\s+international\s+acclaim',
                    r'sustained\s+(?:national|international)\s+acclaim',
                    r'ongoing\s+(?:national|international)\s+recognition',
                    r'continued\s+acclaim\s+in\s+(?:the\s+)?field',
                    r'persistent\s+recognition.*field\s+of\s+endeavor'
                ],
                'regulatory_basis': '8 C.F.R. § 204.5(h)(2)',
                'weight': 1.0,
                'context_keywords': ['acclaim', 'recognition', 'field', 'endeavor', 'achievement']
            },
            'top_percentage': {
                'patterns': [
                    r'small\s+percentage\s+at\s+the\s+(?:very\s+)?top\s+of\s+(?:the\s+)?field',
                    r'among\s+the\s+(?:very\s+)?(?:small\s+)?(?:few|top)\s+(?:percent|percentage)',
                    r'(?:very\s+)?top\s+of\s+(?:his|her|their)\s+field\s+of\s+endeavor',
                    r'distinguished\s+(?:above|beyond)\s+others\s+in\s+(?:the\s+)?field',
                    r'risen\s+to\s+the\s+(?:very\s+)?top\s+of\s+(?:the\s+)?field'
                ],
                'regulatory_basis': '8 C.F.R. § 204.5(h)(2)',
                'weight': 1.0,
                'context_keywords': ['top', 'field', 'percentage', 'distinguished', 'exceptional']
            },
            'recognition_level': {
                'patterns': [
                    r'recognition\s+of\s+achievements?\s+in\s+the\s+field',
                    r'acknowledged\s+(?:expertise|excellence)\s+in',
                    r'widely\s+recognized\s+(?:as|for)',
                    r'national\s+or\s+international\s+standing',
                    r'established\s+reputation\s+in\s+(?:the\s+)?field'
                ],
                'regulatory_basis': '8 C.F.R. § 204.5(h)(2)',
                'weight': 0.8,
                'context_keywords': ['recognition', 'achievements', 'expertise', 'reputation']
            }
        }
        
        # Judging Concepts - Enhanced with participation specificity
        self.judging_concepts = {
            'participation_as_judge': {
                'patterns': [
                    r'participation.*(?:as\s+)?(?:a\s+)?judge\s+of\s+(?:the\s+)?work\s+of\s+others',
                    r'served\s+as\s+(?:a\s+)?judge\s+(?:of|for)',
                    r'judging\s+(?:the\s+)?work\s+of\s+others',
                    r'participation.*(?:on\s+)?(?:a\s+)?(?:review|evaluation|selection)\s+panel',
                    r'panel\s+member.*(?:reviewing|evaluating|judging)',
                    r'editorial\s+board.*(?:review|evaluation)'
                ],
                'regulatory_basis': '8 C.F.R. § 204.5(h)(3)(iv)',
                'weight': 1.0,
                'context_keywords': ['judge', 'participation', 'panel', 'review', 'evaluation']
            },
            'peer_review': {
                'patterns': [
                    r'peer\s+review(?:er)?',
                    r'manuscript\s+review(?:er)?',
                    r'journal\s+(?:article\s+)?review(?:er)?',
                    r'grant\s+(?:application\s+)?review(?:er)?',
                    r'conference\s+(?:paper\s+)?review(?:er)?',
                    r'abstract\s+review(?:er)?'
                ],
                'regulatory_basis': '8 C.F.R. § 204.5(h)(3)(iv)',
                'weight': 0.9,
                'context_keywords': ['peer', 'review', 'manuscript', 'journal', 'grant']
            },
            'selection_committee': {
                'patterns': [
                    r'selection\s+committee',
                    r'evaluation\s+committee',
                    r'review\s+committee',
                    r'admissions\s+committee',
                    r'hiring\s+committee',
                    r'award\s+(?:selection\s+)?committee'
                ],
                'regulatory_basis': '8 C.F.R. § 204.5(h)(3)(iv)',
                'weight': 0.8,
                'context_keywords': ['committee', 'selection', 'evaluation', 'admissions']
            },
            'expertise_requirements': {
                'patterns': [
                    r'qualified\s+to\s+judge',
                    r'expertise.*(?:in\s+)?(?:the\s+)?(?:same|allied)\s+field',
                    r'recognized\s+expert\s+in',
                    r'professional\s+standing.*field',
                    r'competence\s+to\s+evaluate'
                ],
                'regulatory_basis': '8 C.F.R. § 204.5(h)(3)(iv)',
                'weight': 0.7,
                'context_keywords': ['qualified', 'expertise', 'expert', 'competence', 'field']
            }
        }
        
        # Awards and Recognition Concepts - Enhanced with national/international distinction
        self.awards_concepts = {
            'national_recognition': {
                'patterns': [
                    r'nationally\s+recognized\s+(?:prize|award|honor)',
                    r'national\s+(?:level\s+)?(?:prize|award|honor)',
                    r'(?:prize|award|honor).*national\s+(?:scope|level|significance)',
                    r'national\s+professional\s+(?:recognition|honor)',
                    r'country-wide\s+(?:recognition|award)'
                ],
                'regulatory_basis': '8 C.F.R. § 204.5(h)(3)(i)',
                'weight': 1.0,
                'context_keywords': ['national', 'recognized', 'award', 'prize', 'honor']
            },
            'international_recognition': {
                'patterns': [
                    r'internationally\s+recognized\s+(?:prize|award|honor)',
                    r'international\s+(?:level\s+)?(?:prize|award|honor)',
                    r'(?:prize|award|honor).*international\s+(?:scope|level|significance)',
                    r'global\s+(?:recognition|award|honor)',
                    r'worldwide\s+(?:recognition|acclaim)'
                ],
                'regulatory_basis': '8 C.F.R. § 204.5(h)(3)(i)',
                'weight': 1.0,
                'context_keywords': ['international', 'global', 'worldwide', 'award', 'recognition']
            },
            'excellence_indicators': {
                'patterns': [
                    r'excellence\s+in\s+(?:the\s+)?field\s+of\s+endeavor',
                    r'outstanding\s+achievement\s+in',
                    r'exceptional\s+performance\s+in',
                    r'distinguished\s+accomplishment',
                    r'merit-based\s+recognition'
                ],
                'regulatory_basis': '8 C.F.R. § 204.5(h)(3)(i)',
                'weight': 0.8,
                'context_keywords': ['excellence', 'outstanding', 'exceptional', 'distinguished', 'merit']
            },
            'competitive_selection': {
                'patterns': [
                    r'competitive\s+(?:selection|process)',
                    r'rigorous\s+(?:selection|evaluation)',
                    r'selective\s+(?:award|recognition)',
                    r'limited\s+number\s+of\s+recipients',
                    r'exclusive\s+(?:recognition|honor)'
                ],
                'regulatory_basis': 'USCIS Policy Manual',
                'weight': 0.7,
                'context_keywords': ['competitive', 'selective', 'rigorous', 'exclusive', 'limited']
            }
        }
        
        # Evidence and Documentation Concepts
        self.evidence_concepts = {
            'burden_of_proof': {
                'patterns': [
                    r'burden\s+of\s+proof',
                    r'petitioner\s+(?:must\s+)?(?:demonstrate|establish|prove|show)',
                    r'responsibility\s+to\s+(?:demonstrate|establish|prove)',
                    r'obligation\s+to\s+(?:demonstrate|establish|prove)'
                ],
                'regulatory_basis': 'General Immigration Law',
                'weight': 0.6,
                'context_keywords': ['burden', 'demonstrate', 'establish', 'prove', 'responsibility']
            },
            'preponderance_standard': {
                'patterns': [
                    r'preponderance\s+of\s+(?:the\s+)?evidence',
                    r'more\s+likely\s+than\s+not',
                    r'probably\s+true',
                    r'greater\s+weight\s+of\s+(?:the\s+)?evidence'
                ],
                'regulatory_basis': 'Matter of Chawathe',
                'weight': 0.8,
                'context_keywords': ['preponderance', 'evidence', 'likely', 'probable', 'weight']
            },
            'documentation_requirements': {
                'patterns': [
                    r'documentation\s+of',
                    r'evidence\s+of',
                    r'proof\s+of',
                    r'record\s+(?:must\s+)?(?:contain|include|demonstrate)',
                    r'submitted\s+evidence'
                ],
                'regulatory_basis': '8 C.F.R. § 204.5',
                'weight': 0.5,
                'context_keywords': ['documentation', 'evidence', 'proof', 'record', 'submitted']
            }
        }
        
        # Regulatory and Procedural Concepts
        self.regulatory_concepts = {
            'regulatory_criteria': {
                'patterns': [
                    r'regulatory\s+criteria?\s+(?:at|under)\s+8\s+C\.F\.R',
                    r'criteria?\s+(?:at|under)\s+8\s+C\.F\.R\.?\s+§?\s*204\.5\(h\)\(3\)',
                    r'evidentiary\s+criteria?\s+(?:at|under)',
                    r'alternate\s+regulatory\s+criteria?'
                ],
                'regulatory_basis': '8 C.F.R. § 204.5(h)(3)',
                'weight': 1.0,
                'context_keywords': ['regulatory', 'criteria', 'evidentiary', 'alternate']
            },
            'final_merits_determination': {
                'patterns': [
                    r'final\s+merits\s+determination',
                    r'totality\s+of\s+(?:the\s+)?(?:material|evidence)',
                    r'two-part\s+(?:analysis|review)',
                    r'Kazarian\s+(?:analysis|test|review)'
                ],
                'regulatory_basis': 'Kazarian v. USCIS',
                'weight': 0.9,
                'context_keywords': ['final', 'merits', 'totality', 'Kazarian', 'two-part']
            },
            'major_award_alternative': {
                'patterns': [
                    r'major,?\s+internationally\s+recognized\s+award',
                    r'one-time\s+achievement',
                    r'Nobel\s+Prize',
                    r'Olympic\s+(?:Gold\s+)?Medal',
                    r'Academy\s+Award'
                ],
                'regulatory_basis': '8 C.F.R. § 204.5(h)(3)',
                'weight': 1.0,
                'context_keywords': ['major', 'internationally', 'recognized', 'award', 'achievement']
            }
        }
        
        # Procedural Concepts
        self.procedural_concepts = {
            'appeal_procedures': {
                'patterns': [
                    r'(?:on\s+)?appeal',
                    r'appellate\s+(?:review|decision)',
                    r'Administrative\s+Appeals\s+Office',
                    r'AAO\s+(?:decision|determination|review)',
                    r'de\s+novo\s+review'
                ],
                'regulatory_basis': '8 C.F.R. § 103.3',
                'weight': 0.6,
                'context_keywords': ['appeal', 'appellate', 'AAO', 'review', 'novo']
            },
            'director_determination': {
                'patterns': [
                    r'(?:the\s+)?Director\s+(?:determined|concluded|found)',
                    r'initial\s+(?:decision|determination)',
                    r'Service\s+Center\s+(?:decision|determination)',
                    r'denial\s+(?:decision|determination)'
                ],
                'regulatory_basis': 'Administrative Process',
                'weight': 0.5,
                'context_keywords': ['Director', 'determined', 'initial', 'decision', 'denial']
            }
        }
    
    def extract_legal_concepts(self, text: str, metadata: Optional[Dict] = None) -> ConceptExtractionResult:
        """
        Extract legal concepts from document text with enhanced domain understanding.
        
        Args:
            text: Document text to analyze
            metadata: Optional document metadata
            
        Returns:
            ConceptExtractionResult with categorized legal concepts
        """
        # Extract concepts by category
        ea_concepts = self._extract_concepts_by_category(text, self.extraordinary_ability_concepts, 'extraordinary_ability')
        judging_concepts = self._extract_concepts_by_category(text, self.judging_concepts, 'judging')
        awards_concepts = self._extract_concepts_by_category(text, self.awards_concepts, 'awards')
        evidence_concepts = self._extract_concepts_by_category(text, self.evidence_concepts, 'evidence')
        regulatory_concepts = self._extract_concepts_by_category(text, self.regulatory_concepts, 'regulatory')
        procedural_concepts = self._extract_concepts_by_category(text, self.procedural_concepts, 'procedural')
        
        # Calculate concept density
        concept_density = self._calculate_concept_density(
            text, ea_concepts + judging_concepts + awards_concepts + evidence_concepts + regulatory_concepts + procedural_concepts
        )
        
        # Calculate legal domain relevance
        domain_relevance = self._calculate_legal_domain_relevance(
            ea_concepts, judging_concepts, awards_concepts, evidence_concepts, regulatory_concepts
        )
        
        return ConceptExtractionResult(
            extraordinary_ability_concepts=ea_concepts,
            judging_concepts=judging_concepts,
            awards_concepts=awards_concepts,
            evidence_concepts=evidence_concepts,
            regulatory_concepts=regulatory_concepts,
            procedural_concepts=procedural_concepts,
            concept_density_score=concept_density,
            legal_domain_relevance=domain_relevance
        )
    
    def _extract_concepts_by_category(self, text: str, concept_category: Dict, category_name: str) -> List[LegalConcept]:
        """Extract concepts from a specific category."""
        concepts = []
        
        for concept_type, concept_data in concept_category.items():
            patterns = concept_data['patterns']
            weight = concept_data.get('weight', 1.0)
            regulatory_basis = concept_data.get('regulatory_basis')
            context_keywords = concept_data.get('context_keywords', [])
            
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                
                for match in matches:
                    # Calculate confidence based on pattern specificity and context
                    confidence = self._calculate_concept_confidence(
                        match, text, weight, context_keywords
                    )
                    
                    # Extract surrounding context
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 100)
                    context = text[start:end].strip()
                    
                    concept = LegalConcept(
                        concept_type=concept_type,
                        concept_name=f"{category_name}_{concept_type}",
                        matched_text=match.group(0),
                        confidence=confidence,
                        context=context,
                        position=match.start(),
                        regulatory_basis=regulatory_basis
                    )
                    concepts.append(concept)
        
        # Remove duplicates and sort by confidence
        unique_concepts = self._deduplicate_concepts(concepts)
        return sorted(unique_concepts, key=lambda x: x.confidence, reverse=True)
    
    def _calculate_concept_confidence(self, 
                                    match: re.Match, 
                                    text: str, 
                                    base_weight: float,
                                    context_keywords: List[str]) -> float:
        """Calculate confidence score for a concept match."""
        confidence = base_weight * 0.5  # Base confidence from pattern weight
        
        # Context keyword boost
        start = max(0, match.start() - 200)
        end = min(len(text), match.end() + 200)
        surrounding_text = text[start:end].lower()
        
        keyword_matches = sum(1 for keyword in context_keywords if keyword in surrounding_text)
        context_boost = min(0.3, keyword_matches * 0.05)
        confidence += context_boost
        
        # Pattern specificity boost
        pattern_length = len(match.group(0))
        if pattern_length > 20:
            confidence += 0.1
        elif pattern_length > 10:
            confidence += 0.05
        
        # Regulatory citation proximity boost
        regulatory_patterns = [
            r'8\s+C\.F\.R\.?\s+§?\s*204\.5',
            r'8\s+U\.S\.C\.?\s+§?\s*1153',
            r'Matter\s+of\s+[A-Z][a-z\-\']+',
            r'USCIS\s+Policy\s+Manual'
        ]
        
        for reg_pattern in regulatory_patterns:
            if re.search(reg_pattern, surrounding_text, re.IGNORECASE):
                confidence += 0.1
                break
        
        return min(1.0, confidence)
    
    def _deduplicate_concepts(self, concepts: List[LegalConcept]) -> List[LegalConcept]:
        """Remove duplicate concepts based on position and text similarity."""
        seen_positions = set()
        unique_concepts = []
        
        for concept in concepts:
            # Create a key based on position range and concept type
            position_range = (concept.position // 50) * 50  # Group by 50-character ranges
            key = (concept.concept_type, position_range)
            
            if key not in seen_positions:
                seen_positions.add(key)
                unique_concepts.append(concept)
        
        return unique_concepts
    
    def _calculate_concept_density(self, text: str, concepts: List[LegalConcept]) -> float:
        """Calculate concept density per 1000 words."""
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
        
        # Weight concepts by confidence
        weighted_concept_count = sum(concept.confidence for concept in concepts)
        return (weighted_concept_count / word_count) * 1000
    
    def _calculate_legal_domain_relevance(self,
                                        ea_concepts: List[LegalConcept],
                                        judging_concepts: List[LegalConcept],
                                        awards_concepts: List[LegalConcept],
                                        evidence_concepts: List[LegalConcept],
                                        regulatory_concepts: List[LegalConcept]) -> float:
        """Calculate overall legal domain relevance score."""
        relevance_score = 0.0
        
        # Weight different concept categories
        weights = {
            'extraordinary_ability': 0.3,
            'judging': 0.2,
            'awards': 0.2,
            'evidence': 0.15,
            'regulatory': 0.15
        }
        
        concept_categories = [
            (ea_concepts, weights['extraordinary_ability']),
            (judging_concepts, weights['judging']),
            (awards_concepts, weights['awards']),
            (evidence_concepts, weights['evidence']),
            (regulatory_concepts, weights['regulatory'])
        ]
        
        for concepts, weight in concept_categories:
            if concepts:
                # Calculate average confidence for this category
                avg_confidence = sum(c.confidence for c in concepts) / len(concepts)
                # Factor in the number of concepts (up to a maximum)
                concept_count_factor = min(1.0, len(concepts) / 5.0)
                category_score = avg_confidence * concept_count_factor * weight
                relevance_score += category_score
        
        return min(1.0, relevance_score)
    
    def get_concept_summary(self, extraction_result: ConceptExtractionResult) -> Dict[str, Any]:
        """Get a summary of extracted concepts for analysis."""
        summary = {
            'concept_counts': {
                'extraordinary_ability': len(extraction_result.extraordinary_ability_concepts),
                'judging': len(extraction_result.judging_concepts),
                'awards': len(extraction_result.awards_concepts),
                'evidence': len(extraction_result.evidence_concepts),
                'regulatory': len(extraction_result.regulatory_concepts),
                'procedural': len(extraction_result.procedural_concepts)
            },
            'top_concepts': {},
            'regulatory_basis_coverage': set(),
            'concept_density': extraction_result.concept_density_score,
            'domain_relevance': extraction_result.legal_domain_relevance
        }
        
        # Get top concepts by category
        all_concept_lists = [
            ('extraordinary_ability', extraction_result.extraordinary_ability_concepts),
            ('judging', extraction_result.judging_concepts),
            ('awards', extraction_result.awards_concepts),
            ('evidence', extraction_result.evidence_concepts),
            ('regulatory', extraction_result.regulatory_concepts),
            ('procedural', extraction_result.procedural_concepts)
        ]
        
        for category_name, concepts in all_concept_lists:
            if concepts:
                top_concept = max(concepts, key=lambda x: x.confidence)
                summary['top_concepts'][category_name] = {
                    'concept_type': top_concept.concept_type,
                    'matched_text': top_concept.matched_text,
                    'confidence': top_concept.confidence
                }
                
                # Collect regulatory basis coverage
                for concept in concepts:
                    if concept.regulatory_basis:
                        summary['regulatory_basis_coverage'].add(concept.regulatory_basis)
        
        summary['regulatory_basis_coverage'] = list(summary['regulatory_basis_coverage'])
        return summary


def create_legal_concept_matcher() -> LegalConceptMatcher:
    """Factory function to create a legal concept matcher."""
    return LegalConceptMatcher()


if __name__ == "__main__":
    # Test the legal concept matcher
    matcher = create_legal_concept_matcher()
    
    # Sample legal text for testing
    test_text = """
    The petitioner must demonstrate sustained national or international acclaim and that the individual
    is among the small percentage at the very top of the field of endeavor. Under 8 C.F.R. § 204.5(h)(3)(iv),
    evidence of the alien's participation, either individually or on a panel, as a judge of the work of others
    in the same or an allied field in which classification is sought.
    
    The petitioner bears the burden of proof to demonstrate eligibility by a preponderance of the evidence.
    Matter of Chawathe, 25 I&N Dec. 369, 375-76 (AAO 2010). The Director determined that the petitioner
    satisfied the criteria for nationally recognized prizes or awards for excellence in the field of endeavor.
    
    Evidence shows participation as a peer reviewer for manuscripts submitted to professional journals
    and service on grant evaluation committees, demonstrating recognized expertise in the field.
    """
    
    print("Testing Legal Concept Matcher")
    print("=" * 50)
    
    extraction = matcher.extract_legal_concepts(test_text)
    summary = matcher.get_concept_summary(extraction)
    
    print(f"Concept Counts:")
    for category, count in summary['concept_counts'].items():
        print(f"  {category}: {count}")
    
    print(f"\nTop Concepts:")
    for category, concept_info in summary['top_concepts'].items():
        print(f"  {category}: {concept_info['matched_text']} (confidence: {concept_info['confidence']:.3f})")
    
    print(f"\nRegulatory Basis Coverage:")
    for basis in summary['regulatory_basis_coverage']:
        print(f"  - {basis}")
    
    print(f"\nConcept Density: {summary['concept_density']:.2f} per 1000 words")
    print(f"Domain Relevance: {summary['domain_relevance']:.3f}")