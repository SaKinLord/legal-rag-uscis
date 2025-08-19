"""
Advanced Legal Document Analysis for Phase 3 specialization.
Implements sophisticated legal document structure analysis, case law recognition,
and regulatory pattern matching for enhanced legal RAG systems.
"""

import re
import math
from typing import Dict, List, Tuple, Any, Set, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class LegalCitation:
    """Structured representation of a legal citation."""
    citation_type: str
    full_text: str
    case_name: Optional[str] = None
    volume: Optional[str] = None
    reporter: Optional[str] = None
    page: Optional[str] = None
    year: Optional[str] = None
    court: Optional[str] = None
    confidence: float = 0.0


@dataclass
class DocumentStructureAnalysis:
    """Results of legal document structure analysis."""
    case_citations: List[LegalCitation]
    regulatory_citations: List[LegalCitation]
    analysis_sections: List[Dict[str, Any]]
    legal_standards: List[Dict[str, Any]]
    precedent_references: List[Dict[str, Any]]
    document_authority_score: float
    legal_complexity_score: float
    citation_density: float


class LegalDocumentAnalyzer:
    """
    Advanced legal document analysis implementing Phase 3 enhancements.
    Provides deep legal document understanding for improved RAG retrieval.
    """
    
    def __init__(self):
        # Enhanced case citation patterns
        self.case_citation_patterns = {
            'us_supreme_court': [
                r'(\d+)\s+U\.S\.?\s+(\d+)(?:\s+\((\d{4})\))?',
                r'(\d+)\s+S\.?\s?Ct\.?\s+(\d+)(?:\s+\((\d{4})\))?'
            ],
            'federal_circuit': [
                r'(\d+)\s+F\.(\d)d\s+(\d+)(?:\s+\(([^)]+)\s+(\d{4})\))?',
                r'(\d+)\s+F\.?\s+Supp\.?\s+(\d)d\s+(\d+)(?:\s+\(([^)]+)\s+(\d{4})\))?'
            ],
            'bia_decisions': [
                r'Matter\s+of\s+([A-Z][a-z\-\'\s]+),?\s+(\d+)\s+I&N\s+Dec\.?\s+(\d+)(?:\s+\(([^)]+)\s+(\d{4})\))?',
                r'In\s+re:?\s+([A-Z][a-z\-\'\s]+),?\s+(\d+)\s+I&N\s+Dec\.?\s+(\d+)(?:\s+\(([^)]+)\s+(\d{4})\))?'
            ],
            'aao_decisions': [
                r'Matter\s+of\s+([A-Z][a-z\-\'\s]+),?\s+(\d+)\s+I&N\s+Dec\.?\s+(\d+)(?:\s+\((AAO)\s+(\d{4})\))?',
                r'In\s+Re:?\s+(\d+)',  # AAO case numbers
                r'Case\s+No\.?\s+([A-Z0-9\-]+)'
            ]
        }
        
        # Regulatory citation patterns
        self.regulatory_patterns = {
            'cfr_citations': [
                r'8\s+C\.F\.R\.?\s+§?\s*(\d+)\.(\d+)(?:\(([a-z0-9]+)\))?(?:\(([a-z0-9]+)\))?',
                r'8\s+U\.S\.C\.?\s+§?\s*(\d+)(?:\(([a-z0-9]+)\))?',
                r'INA\s+§?\s*(\d+)(?:\(([a-z])\))?'
            ],
            'policy_manual': [
                r'(\d+)\s+USCIS\s+Policy\s+Manual\s+([A-Z])\.(\d+)(?:\(([A-Z])\))?',
                r'USCIS\s+Policy\s+Manual\s+([A-Z])\.(\d+)(?:\(([A-Z])\))?(?:\((\d+)\))?'
            ],
            'federal_register': [
                r'(\d+)\s+Fed\.?\s+Reg\.?\s+(\d+)(?:\s+\(([^)]+)\s+(\d{1,2}),?\s+(\d{4})\))?'
            ]
        }
        
        # Legal analysis section patterns
        self.analysis_section_patterns = {
            'analysis_headers': [
                r'^(?:II\.?\s*)?ANALYSIS\s*$',
                r'^(?:III\.?\s*)?DISCUSSION\s*$',
                r'^(?:IV\.?\s*)?FINDINGS?\s*$',
                r'^(?:V\.?\s*)?CONCLUSION\s*$',
                r'^(?:VI\.?\s*)?DETERMINATION\s*$',
                r'^(?:A\.?\s*)?Legal\s+Standard\s*$',
                r'^(?:B\.?\s*)?Application\s+of\s+Law\s*$'
            ],
            'criterion_analysis': [
                r'criterion\s+at\s+8\s+C\.F\.R\.?\s+§?\s*204\.5\(h\)\(3\)\([ivx]+\)',
                r'regulatory\s+criteria?\s+under\s+8\s+C\.F\.R\.?\s+§?\s*204\.5',
                r'evidentiary\s+criteria?\s+at\s+8\s+C\.F\.R\.?'
            ],
            'burden_of_proof': [
                r'burden\s+of\s+proof',
                r'preponderance\s+of\s+(?:the\s+)?evidence',
                r'petitioner\s+(?:must\s+)?(?:demonstrate|establish|prove)',
                r'standard\s+of\s+(?:proof|evidence)'
            ]
        }
        
        # Legal standards and tests
        self.legal_standards = {
            'extraordinary_ability_standard': [
                r'sustained\s+national\s+or\s+international\s+acclaim',
                r'small\s+percentage\s+at\s+the\s+(?:very\s+)?top\s+of\s+(?:the\s+)?field',
                r'recognition\s+of\s+achievements?\s+in\s+the\s+field',
                r'one-time\s+achievement.*major.*internationally\s+recognized\s+award'
            ],
            'kazarian_test': [
                r'Kazarian\s+v\.?\s+USCIS',
                r'two-part\s+(?:analysis|review)',
                r'final\s+merits\s+determination',
                r'totality\s+of\s+(?:the\s+)?(?:material|evidence)'
            ],
            'preponderance_standard': [
                r'preponderance\s+of\s+(?:the\s+)?evidence',
                r'probably\s+true',
                r'more\s+likely\s+than\s+not',
                r'burden\s+of\s+proof.*preponderance'
            ]
        }
        
        # Precedent reference patterns
        self.precedent_patterns = {
            'citing_patterns': [
                r'(?:see|citing|accord|cf\.?)\s+Matter\s+of\s+([A-Z][a-z\-\'\s]+)',
                r'(?:see|citing|accord|cf\.?)\s+([A-Z][a-z\-\'\s]+)\s+v\.?\s+([A-Z][a-z\-\'\s]+)',
                r'as\s+(?:stated|held|determined|found)\s+in\s+Matter\s+of\s+([A-Z][a-z\-\'\s]+)',
                r'consistent\s+with\s+(?:Matter\s+of\s+)?([A-Z][a-z\-\'\s]+)',
                r'in\s+accordance\s+with\s+(?:Matter\s+of\s+)?([A-Z][a-z\-\'\s]+)'
            ],
            'distinguishing_patterns': [
                r'(?:distinguish|unlike|however|but\s+see)\s+Matter\s+of\s+([A-Z][a-z\-\'\s]+)',
                r'(?:distinguish|unlike)\s+([A-Z][a-z\-\'\s]+)\s+v\.?\s+([A-Z][a-z\-\'\s]+)'
            ]
        }
    
    def analyze_document_structure(self, doc_text: str, metadata: Optional[Dict] = None) -> DocumentStructureAnalysis:
        """
        Perform comprehensive legal document structure analysis.
        
        Args:
            doc_text: Document text to analyze
            metadata: Optional document metadata
            
        Returns:
            DocumentStructureAnalysis with detailed legal structure information
        """
        # Extract case citations
        case_citations = self._extract_case_citations(doc_text)
        
        # Extract regulatory citations
        regulatory_citations = self._extract_regulatory_citations(doc_text)
        
        # Identify analysis sections
        analysis_sections = self._identify_analysis_sections(doc_text)
        
        # Extract legal standards
        legal_standards = self._extract_legal_standards(doc_text)
        
        # Find precedent references
        precedent_references = self._extract_precedent_references(doc_text)
        
        # Calculate document authority score
        authority_score = self._calculate_document_authority(
            case_citations, regulatory_citations, precedent_references, metadata
        )
        
        # Calculate legal complexity score
        complexity_score = self._calculate_legal_complexity(
            case_citations, regulatory_citations, analysis_sections, legal_standards
        )
        
        # Calculate citation density
        citation_density = self._calculate_citation_density(
            doc_text, case_citations, regulatory_citations
        )
        
        return DocumentStructureAnalysis(
            case_citations=case_citations,
            regulatory_citations=regulatory_citations,
            analysis_sections=analysis_sections,
            legal_standards=legal_standards,
            precedent_references=precedent_references,
            document_authority_score=authority_score,
            legal_complexity_score=complexity_score,
            citation_density=citation_density
        )
    
    def _extract_case_citations(self, text: str) -> List[LegalCitation]:
        """Extract case law citations from text."""
        citations = []
        
        for citation_type, patterns in self.case_citation_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                
                for match in matches:
                    citation = self._parse_case_citation(match, citation_type)
                    if citation:
                        citations.append(citation)
        
        # Remove duplicates and sort by confidence
        unique_citations = self._deduplicate_citations(citations)
        return sorted(unique_citations, key=lambda x: x.confidence, reverse=True)
    
    def _extract_regulatory_citations(self, text: str) -> List[LegalCitation]:
        """Extract regulatory citations from text."""
        citations = []
        
        for citation_type, patterns in self.regulatory_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    citation = self._parse_regulatory_citation(match, citation_type)
                    if citation:
                        citations.append(citation)
        
        return self._deduplicate_citations(citations)
    
    def _identify_analysis_sections(self, text: str) -> List[Dict[str, Any]]:
        """Identify legal analysis sections in the document."""
        sections = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            for section_type, patterns in self.analysis_section_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, line_stripped, re.IGNORECASE):
                        # Extract content following the header
                        content_start = i + 1
                        content_lines = []
                        
                        # Collect content until next major section or end
                        for j in range(content_start, min(content_start + 20, len(lines))):
                            if j < len(lines):
                                next_line = lines[j].strip()
                                if (next_line and 
                                    not re.search(r'^[IVX]+\.', next_line) and
                                    not re.search(r'^[A-E]\.', next_line)):
                                    content_lines.append(next_line)
                                elif next_line and re.search(r'^[IVX]+\.', next_line):
                                    break
                        
                        section = {
                            'type': section_type,
                            'header': line_stripped,
                            'line_number': i,
                            'content_preview': ' '.join(content_lines[:3]),
                            'pattern_matched': pattern
                        }
                        sections.append(section)
        
        return sections
    
    def _extract_legal_standards(self, text: str) -> List[Dict[str, Any]]:
        """Extract legal standards and tests mentioned in the document."""
        standards = []
        
        for standard_type, patterns in self.legal_standards.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    # Get surrounding context
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 100)
                    context = text[start:end].strip()
                    
                    standard = {
                        'type': standard_type,
                        'matched_text': match.group(0),
                        'pattern': pattern,
                        'context': context,
                        'position': match.start()
                    }
                    standards.append(standard)
        
        return standards
    
    def _extract_precedent_references(self, text: str) -> List[Dict[str, Any]]:
        """Extract precedent references and their types."""
        references = []
        
        for ref_type, patterns in self.precedent_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    # Extract case name from the match
                    case_name = None
                    if match.groups():
                        case_name = match.group(1) if match.group(1) else None
                    
                    # Get surrounding context
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end].strip()
                    
                    reference = {
                        'type': ref_type,
                        'case_name': case_name,
                        'matched_text': match.group(0),
                        'context': context,
                        'position': match.start()
                    }
                    references.append(reference)
        
        return references
    
    def _parse_case_citation(self, match: re.Match, citation_type: str) -> Optional[LegalCitation]:
        """Parse a case citation match into a structured citation."""
        try:
            groups = match.groups()
            full_text = match.group(0)
            
            if citation_type == 'us_supreme_court':
                return LegalCitation(
                    citation_type=citation_type,
                    full_text=full_text,
                    volume=groups[0] if groups[0] else None,
                    page=groups[1] if groups[1] else None,
                    year=groups[2] if len(groups) > 2 and groups[2] else None,
                    reporter='U.S.',
                    confidence=0.95
                )
            
            elif citation_type == 'federal_circuit':
                return LegalCitation(
                    citation_type=citation_type,
                    full_text=full_text,
                    volume=groups[0] if groups[0] else None,
                    reporter=f"F.{groups[1]}d" if len(groups) > 1 and groups[1] else "F.",
                    page=groups[2] if len(groups) > 2 and groups[2] else None,
                    court=groups[3] if len(groups) > 3 and groups[3] else None,
                    year=groups[4] if len(groups) > 4 and groups[4] else None,
                    confidence=0.90
                )
            
            elif citation_type == 'bia_decisions':
                return LegalCitation(
                    citation_type=citation_type,
                    full_text=full_text,
                    case_name=groups[0] if groups[0] else None,
                    volume=groups[1] if len(groups) > 1 and groups[1] else None,
                    page=groups[2] if len(groups) > 2 and groups[2] else None,
                    court=groups[3] if len(groups) > 3 and groups[3] else 'BIA',
                    year=groups[4] if len(groups) > 4 and groups[4] else None,
                    reporter='I&N Dec.',
                    confidence=0.92
                )
            
            elif citation_type == 'aao_decisions':
                return LegalCitation(
                    citation_type=citation_type,
                    full_text=full_text,
                    case_name=groups[0] if groups and groups[0] else None,
                    court='AAO',
                    confidence=0.85
                )
            
        except Exception as e:
            logger.warning(f"Error parsing case citation: {e}")
        
        return None
    
    def _parse_regulatory_citation(self, match: re.Match, citation_type: str) -> Optional[LegalCitation]:
        """Parse a regulatory citation match."""
        try:
            groups = match.groups()
            full_text = match.group(0)
            
            return LegalCitation(
                citation_type=citation_type,
                full_text=full_text,
                volume=groups[0] if groups and groups[0] else None,
                page=groups[1] if len(groups) > 1 and groups[1] else None,
                confidence=0.88
            )
            
        except Exception as e:
            logger.warning(f"Error parsing regulatory citation: {e}")
        
        return None
    
    def _deduplicate_citations(self, citations: List[LegalCitation]) -> List[LegalCitation]:
        """Remove duplicate citations based on full text similarity."""
        seen = set()
        unique_citations = []
        
        for citation in citations:
            # Create a normalized key for deduplication
            key = citation.full_text.lower().strip()
            key = re.sub(r'\s+', ' ', key)  # Normalize whitespace
            
            if key not in seen:
                seen.add(key)
                unique_citations.append(citation)
        
        return unique_citations
    
    def _calculate_document_authority(self, 
                                   case_citations: List[LegalCitation],
                                   regulatory_citations: List[LegalCitation],
                                   precedent_references: List[Dict[str, Any]],
                                   metadata: Optional[Dict] = None) -> float:
        """Calculate document authority score based on citations and references."""
        authority_score = 0.0
        
        # Base authority from citation types
        for citation in case_citations:
            if citation.citation_type == 'us_supreme_court':
                authority_score += 0.15
            elif citation.citation_type == 'federal_circuit':
                authority_score += 0.10
            elif citation.citation_type == 'bia_decisions':
                authority_score += 0.12
            elif citation.citation_type == 'aao_decisions':
                authority_score += 0.08
        
        # Authority from regulatory citations
        for citation in regulatory_citations:
            if citation.citation_type == 'cfr_citations':
                authority_score += 0.10
            elif citation.citation_type == 'policy_manual':
                authority_score += 0.08
        
        # Authority from precedent references
        citing_refs = sum(1 for ref in precedent_references if ref['type'] == 'citing_patterns')
        authority_score += min(0.20, citing_refs * 0.05)
        
        # Metadata-based authority
        if metadata:
            doc_id = metadata.get('document_id', '')
            if 'AAO' in doc_id.upper():
                authority_score += 0.10
        
        return min(1.0, authority_score)
    
    def _calculate_legal_complexity(self,
                                  case_citations: List[LegalCitation],
                                  regulatory_citations: List[LegalCitation],
                                  analysis_sections: List[Dict[str, Any]],
                                  legal_standards: List[Dict[str, Any]]) -> float:
        """Calculate legal complexity score."""
        complexity_score = 0.0
        
        # Citation complexity
        complexity_score += min(0.30, len(case_citations) * 0.05)
        complexity_score += min(0.20, len(regulatory_citations) * 0.03)
        
        # Analysis depth
        complexity_score += min(0.25, len(analysis_sections) * 0.08)
        
        # Legal standards complexity
        complexity_score += min(0.25, len(legal_standards) * 0.05)
        
        return min(1.0, complexity_score)
    
    def _calculate_citation_density(self,
                                  text: str,
                                  case_citations: List[LegalCitation],
                                  regulatory_citations: List[LegalCitation]) -> float:
        """Calculate citation density per 1000 words."""
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
        
        total_citations = len(case_citations) + len(regulatory_citations)
        return (total_citations / word_count) * 1000


def create_legal_document_analyzer() -> LegalDocumentAnalyzer:
    """Factory function to create a legal document analyzer."""
    return LegalDocumentAnalyzer()


if __name__ == "__main__":
    # Test the legal document analyzer
    analyzer = create_legal_document_analyzer()
    
    # Sample legal text for testing
    test_text = """
    II. ANALYSIS
    
    The Petitioner bears the burden of proof to demonstrate eligibility by a preponderance of the evidence.
    Matter of Chawathe, 25 I&N Dec. 369, 375-76 (AAO 2010). We review the questions in this matter
    de novo. Matter of Christo's, Inc., 26 I&N Dec. 537, 537 n.2 (AAO 2015).
    
    Under 8 C.F.R. § 204.5(h)(3)(iv), evidence of the alien's participation, either individually or on a panel,
    as a judge of the work of others in the same or an allied field in which classification is sought.
    See Kazarian v. USCIS, 596 F.3d 1115 (9th Cir. 2010) (discussing a two-part review where the
    documentation is first counted and then, if fulfilling the required number of criteria, considered in
    the context of a final merits determination).
    
    The petitioner must demonstrate sustained national or international acclaim and that the individual
    is among the small percentage at the very top of the field of endeavor.
    """
    
    print("Testing Legal Document Analyzer")
    print("=" * 50)
    
    analysis = analyzer.analyze_document_structure(test_text)
    
    print(f"Case Citations Found: {len(analysis.case_citations)}")
    for citation in analysis.case_citations:
        print(f"  - {citation.citation_type}: {citation.full_text}")
    
    print(f"\nRegulatory Citations Found: {len(analysis.regulatory_citations)}")
    for citation in analysis.regulatory_citations:
        print(f"  - {citation.citation_type}: {citation.full_text}")
    
    print(f"\nAnalysis Sections Found: {len(analysis.analysis_sections)}")
    for section in analysis.analysis_sections:
        print(f"  - {section['type']}: {section['header']}")
    
    print(f"\nLegal Standards Found: {len(analysis.legal_standards)}")
    for standard in analysis.legal_standards:
        print(f"  - {standard['type']}: {standard['matched_text'][:50]}...")
    
    print(f"\nDocument Authority Score: {analysis.document_authority_score:.3f}")
    print(f"Legal Complexity Score: {analysis.legal_complexity_score:.3f}")
    print(f"Citation Density: {analysis.citation_density:.2f} per 1000 words")