# src/query_preprocessor.py

import re
from typing import List, Dict, Set, Tuple
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import spacy
from collections import defaultdict

# --- NLTK Resource Download Logic ---
nltk_data_path_displayed_once = False

def ensure_nltk_resource(resource_human_name: str, download_id: str, check_file_path: str):
    """
    Checks if a specific file for an NLTK resource exists, and downloads if not.
    """
    global nltk_data_path_displayed_once
    try:
        nltk.data.find(check_file_path)
        # print(f"NLTK resource '{resource_human_name}' found ({check_file_path}).")
    except LookupError:
        if not nltk_data_path_displayed_once:
            print(f"NLTK will search for data in the following paths: {nltk.data.path}")
            nltk_data_path_displayed_once = True
        print(f"NLTK resource '{resource_human_name}' (file: {check_file_path}) not found. Attempting to download '{download_id}'...")
        try:
            nltk.download(download_id, quiet=False) # Changed quiet to False for better download visibility
            # Verify after download
            nltk.data.find(check_file_path)
            print(f"NLTK resource '{download_id}' downloaded successfully and verified.")
        except Exception as e:
            print(f"Error downloading or verifying NLTK resource '{download_id}': {e}")
            print(f"Please try manually: import nltk; nltk.download('{download_id}')")
            raise

# Define required NLTK resources and a key file to check for each
# Removed 'punkt_tab' as it should be part of 'punkt'
nltk_resources = [
    ("tokenizers/punkt", "punkt", "tokenizers/punkt/PY3/english.pickle"),
    # MODIFIED LINE for the POS tagger:
    (
        "taggers/averaged_perceptron_tagger_eng",  # Human-readable name (can be anything descriptive)
        "averaged_perceptron_tagger_eng",          # Download ID suggested by the NLTK error
        "taggers/averaged_perceptron_tagger_eng/"  # Path to check (the directory itself)
    ),
    ("corpora/wordnet", "wordnet", "corpora/wordnet/lexnames"),
]

for name, dl_id, file_path in nltk_resources:
    ensure_nltk_resource(name, dl_id, file_path)
# --- End of NLTK Resource Download Logic ---


# Load spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy 'en_core_web_sm' model not found. Downloading...")
    import subprocess
    try:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm")
        print("spaCy 'en_core_web_sm' model downloaded successfully.")
    except Exception as e:
        print(f"Error downloading spaCy model: {e}")
        print("Please try manually: python -m spacy download en_core_web_sm")
        raise


class LegalQueryPreprocessor:
    """Preprocesses queries for legal RAG system with expansion and normalization."""
    
    def __init__(self):
        # Legal term mappings for normalization
        self.legal_abbreviations = {
            "aao": "administrative appeals office",
            "uscis": "united states citizenship and immigration services",
            "i-140": "form i-140 immigrant petition for alien workers",
            "eb1": "employment based first preference",
            "eb-1": "employment based first preference",
            "cfr": "code of federal regulations",
            "usc": "united states code",
            "ina": "immigration and nationality act",
            "rfe": "request for evidence",
            "ea": "extraordinary ability",
            "nio": "national interest waiver"
        }
        
        # Legal synonyms and related terms
        self.legal_synonyms = {
            "extraordinary ability": ["exceptional ability", "outstanding ability", "remarkable talent"],
            "sustained acclaim": ["continuous recognition", "ongoing recognition", "persistent acclaim"],
            "judge": ["adjudicator", "reviewer", "evaluator", "peer reviewer"],
            "participation": ["involvement", "engagement", "contribution", "role"],
            "award": ["prize", "honor", "accolade", "recognition", "achievement"],
            "national": ["nationwide", "countrywide", "domestic"],
            "international": ["global", "worldwide", "multinational"],
            "significance": ["importance", "impact", "influence", "consequence"],
            "criterion": ["criteria", "requirement", "standard", "benchmark"],
            "evidence": ["proof", "documentation", "support", "substantiation"]
        }
        
        # Common legal phrases that should be kept together
        self.legal_phrases = {
            "sustained national or international acclaim",
            "extraordinary ability",
            "major significance",
            "leading or critical role",
            "original contributions",
            "peer review",
            "scholarly articles",
            "professional associations",
            "high salary",
            "commercial success"
        }
        
        # Regex patterns for legal citations
        self.citation_patterns = [
            r'\b8\s*C\.?F\.?R\.?\s*§?\s*\d+\.?\d*',  # 8 CFR § 204.5
            r'\b\d+\s*U\.?S\.?C\.?\s*§?\s*\d+',       # 8 USC § 1153
            r'Matter\s+of\s+[A-Z][a-z]+',              # Matter of Chawathe
            r'I-\d{3}[A-Z]?',                          # I-140
            r'EB-?\d',                                  # EB-1
        ]
        
    def preprocess_query(self, query: str) -> Dict[str, any]:
        """Main preprocessing function that orchestrates all enhancements."""
        original_query = query
        
        # Step 1: Clean and normalize the query
        normalized_query = self._normalize_query(query)
        
        # Step 2: Extract legal entities and citations
        entities = self._extract_legal_entities(normalized_query)
        
        # Step 3: Expand query with synonyms and related terms
        expanded_terms = self._expand_query(normalized_query)
        
        # Step 4: Extract key concepts for targeted search
        key_concepts = self._extract_key_concepts(normalized_query)
        
        # Step 5: Generate multiple query variations
        query_variations = self._generate_query_variations(normalized_query, expanded_terms)
        
        # Step 6: Identify query intent
        query_intent = self._identify_query_intent(normalized_query)
        
        return {
            'original_query': original_query,
            'normalized_query': normalized_query,
            'entities': entities,
            'expanded_terms': expanded_terms,
            'key_concepts': key_concepts,
            'query_variations': query_variations,
            'query_intent': query_intent,
            'search_queries': self._create_search_queries(normalized_query, expanded_terms, key_concepts)
        }
    
    def _normalize_query(self, query: str) -> str:
        """Normalize the query by expanding abbreviations and standardizing format."""
        # Convert to lowercase for processing
        normalized = query.lower()
        
        # Preserve legal citations by temporarily replacing them
        citation_placeholders = {}
        for i, pattern in enumerate(self.citation_patterns):
            matches = re.finditer(pattern, normalized, re.IGNORECASE)
            for match in matches:
                placeholder = f"__CITATION_{i}_{len(citation_placeholders)}__"
                citation_placeholders[placeholder] = match.group()
                normalized = normalized.replace(match.group().lower(), placeholder)
        
        # Expand abbreviations
        for abbr, full_form in self.legal_abbreviations.items():
            normalized = re.sub(r'\b' + abbr + r'\b', full_form, normalized)
        
        # Restore citations
        for placeholder, citation in citation_placeholders.items():
            normalized = normalized.replace(placeholder, citation)
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def _extract_legal_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract legal entities, citations, and important terms from the query."""
        entities = {
            'citations': [],
            'forms': [],
            'organizations': [],
            'legal_concepts': [],
            'criteria': []
        }
        
        # Extract citations
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['citations'].extend(matches)
        
        # Extract form references
        form_pattern = r'\b(?:form\s+)?I-\d{3}[A-Z]?\b'
        entities['forms'] = re.findall(form_pattern, text, re.IGNORECASE)
        
        # Extract criteria references
        criteria_keywords = [
            'criterion', 'criteria', 'requirement', 'standard',
            'participation as a judge', 'scholarly articles', 'original contributions',
            'leading role', 'critical role', 'high salary', 'awards', 'memberships'
        ]
        for keyword in criteria_keywords:
            if keyword in text.lower():
                entities['criteria'].append(keyword)
        
        # Use spaCy for organization extraction
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "ORG":
                entities['organizations'].append(ent.text)
        
        # Extract legal concepts
        legal_concepts = [
            'extraordinary ability', 'sustained acclaim', 'major significance',
            'national recognition', 'international recognition', 'peer review'
        ]
        for concept in legal_concepts:
            if concept in text.lower():
                entities['legal_concepts'].append(concept)
        
        return entities
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related legal terms."""
        expanded_terms = set()
        tokens = word_tokenize(query.lower()) # This uses NLTK's default English tokenizer (Punkt)
        
        # Add original tokens
        expanded_terms.update(tokens)
        
        # Expand using legal synonyms
        for term, synonyms in self.legal_synonyms.items():
            if term in query.lower():
                expanded_terms.update(synonyms)
        
        # Expand individual words using WordNet
        pos_tags = pos_tag(tokens) # This uses NLTK's default English POS tagger (PerceptronTagger)
        for word, pos in pos_tags:
            # Only expand nouns and verbs
            if pos.startswith('NN') or pos.startswith('VB'):
                synsets = wordnet.synsets(word)
                for synset in synsets[:3]:  # Limit to top 3 synsets
                    for lemma in synset.lemmas()[:3]:  # Limit to top 3 lemmas
                        expanded_terms.add(lemma.name().replace('_', ' '))
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        expanded_terms = {term for term in expanded_terms if term not in stop_words}
        
        return list(expanded_terms)
    
    def _extract_key_concepts(self, query: str) -> List[str]:
        """Extract key legal concepts from the query."""
        key_concepts = []
        
        # Check for legal phrases
        for phrase in self.legal_phrases:
            if phrase in query.lower():
                key_concepts.append(phrase)
        
        # Extract noun phrases using spaCy
        doc = nlp(query)
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 2:  # Multi-word phrases
                key_concepts.append(chunk.text.lower())
        
        # Extract specific legal criteria mentions
        criteria_patterns = [
            r'participation\s+as\s+a\s+judge',
            r'judging\s+.*\s+work',
            r'peer\s+review',
            r'original\s+contributions?',
            r'major\s+significance',
            r'leading\s+(?:or\s+critical\s+)?role',
            r'high\s+salary',
            r'commercial\s+success'
        ]
        
        for pattern in criteria_patterns:
            matches = re.findall(pattern, query.lower())
            key_concepts.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_concepts = []
        for concept in key_concepts:
            if concept not in seen:
                seen.add(concept)
                unique_concepts.append(concept)
        
        return unique_concepts
    
    def _generate_query_variations(self, query: str, expanded_terms: List[str]) -> List[str]:
        """Generate multiple query variations for better retrieval."""
        variations = {query}  # Original query, use set for auto-deduplication

        # Variation 1: Query with key expanded terms
        key_expansions = [term for term in expanded_terms if len(term.strip()) > 3][:5]
        if key_expansions:
            variations.add(f"{query} {' '.join(key_expansions)}")
        
        # Variation 2: Question form
        if not query.strip().endswith('?'):
            variations.add(f"{query}?")
        
        # Variation 3: Focus on criteria
        if "criterion" in query.lower() or "criteria" in query.lower():
            variations.add(f"8 CFR 204.5(h)(3) {query}")
        
        # Variation 4: Focus on AAO decisions
        if "aao" not in query.lower() and "administrative appeals office" not in query.lower():
            variations.add(f"AAO decisions {query}")
        
        # Variation 5: Rephrase as "how does/do"
        if query.lower().startswith(('what', 'when', 'where', 'why')):
            prefix_to_replace = query.lower().split()[0] # what, when, etc.
            if "aao" not in query.lower() and "administrative appeals office" not in query.lower():
                 rephrased = f"how does AAO evaluate {query[len(prefix_to_replace):].strip()}"
            else: 
                 rephrased = f"how does it evaluate {query[len(prefix_to_replace):].strip()}"
            variations.add(rephrased)
        
        return list(variations)
    
    def _identify_query_intent(self, query: str) -> str:
        """Identify the intent behind the query."""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['leading role', 'critical role']):
            return 'criterion_role'
        elif any(term in query_lower for term in ['judge', 'judging']):
            return 'criterion_judge'
        elif 'peer review' in query_lower: 
            if any(term in query_lower for term in ['contribution', 'significance']):
                return 'criterion_contributions' # Peer review as evidence for contributions
            if any(term in query_lower for term in ['publication', 'article', 'scholarly']):
                 return 'criterion_publications' # Peer review for scholarly articles
            return 'concept_peer_review' # General peer review concept

        elif any(term in query_lower for term in ['award', 'prize']):
            return 'criterion_awards'
        
        elif any(term in query_lower for term in ['publication', 'article', 'scholarly']):
            return 'criterion_publications'
        elif any(term in query_lower for term in ['contribution', 'significance']): # 'major significance' is a key phrase
            return 'criterion_contributions'
        elif any(term in query_lower for term in ['salary', 'remuneration', 'compensation']):
            return 'criterion_salary'
        elif any(term in query_lower for term in ['membership', 'association']):
            return 'criterion_membership'
        elif any(term in query_lower for term in ['media', 'press', 'published material']):
            return 'criterion_media'
        
        # General concepts (check after specific criteria to avoid premature matching)
        elif 'sustained acclaim' in query_lower or 'national recognition' in query_lower or 'international recognition' in query_lower :
             # 'recognition' alone might be too broad, so couple it with national/international or sustained acclaim
            return 'concept_acclaim'
        elif 'extraordinary ability' in query_lower:
            return 'concept_extraordinary_ability'
        
        # Procedural or definitional questions
        elif any(term in query_lower for term in ['how to', 'process', 'procedure', 'apply', 'requirement', 'what is', 'define']):
            return 'procedural_definitional'
        
        return 'general_inquiry'
    
    def _create_search_queries(self, normalized_query: str, expanded_terms: List[str], 
                             key_concepts: List[str]) -> List[Tuple[str, float]]:
        """Create weighted search queries for retrieval."""
        search_queries_dict = {} 

        def add_query(query_str, weight):
            query_str = query_str.strip()
            if not query_str: return # Avoid empty queries
            # If query already exists, update if new weight is higher
            if query_str not in search_queries_dict or weight > search_queries_dict[query_str]:
                search_queries_dict[query_str] = weight

        # 1. Original normalized query (highest weight)
        add_query(normalized_query, 1.0)
        
        # 2. Key concepts query (high weight)
        if key_concepts:
            # Prioritize longer, more specific key concepts
            sorted_key_concepts = sorted(list(set(key_concepts)), key=len, reverse=True)
            concept_query = ' '.join(sorted_key_concepts[:3]) # Top 3 unique, longest concepts
            add_query(concept_query, 0.9)
        
        # 3. Query with some relevant expanded terms (medium-high weight)
        #    Terms that are expansions but not already in the normalized query
        additional_expansions = [
            term for term in expanded_terms 
            if term not in normalized_query.split() and len(term.strip()) > 4 # Min length for meaningful expansion
        ][:3] # Limit to a few strong expansions

        if additional_expansions:
            combined_terms = set(normalized_query.split())
            combined_terms.update(additional_expansions)
            expanded_query_str = ' '.join(list(combined_terms))
            add_query(expanded_query_str, 0.75) 
        
        # 4. Broader expansion (medium weight) - using more of the expanded terms
        if expanded_terms:
            # Select longer expanded terms, ensure they are somewhat distinct from normalized query
            broad_expansion_terms = [term for term in expanded_terms if len(term.strip()) > 4 and term not in normalized_query][:5]
            if broad_expansion_terms:
                add_query(' '.join(broad_expansion_terms), 0.6)

        # 5. Citation-focused query if citations present (medium-high weight)
        temp_entities = self._extract_legal_entities(normalized_query) # Use normalized query
        citations = list(set(temp_entities.get('citations', []))) # Unique citations
        
        if citations:
            citation_only_query = ' '.join(citations)
            add_query(citation_only_query, 0.8) 
            # Combine normalized query with citations
            add_query(f"{normalized_query} {' '.join(citations)}", 0.85)
        
        # Sort final queries by weight, descending
        sorted_queries = sorted(search_queries_dict.items(), key=lambda item: item[1], reverse=True)
        
        return sorted_queries


# Example usage and testing (if run directly)
if __name__ == "__main__":
    preprocessor = LegalQueryPreprocessor()
    
    test_queries = [
        "How do recent AAO decisions evaluate an applicant's Participation as a Judge service criteria?",
        "What characteristics of national or international awards persuade the AAO that they constitute sustained acclaim?",
        "EB-1 extraordinary ability requirements under 8 CFR 204.5",
        "What is major significance in original contributions?"
    ]
    
    for query in test_queries:
        print(f"\n--- Testing Query ---")
        print(f"Original: {query}")
        result = preprocessor.preprocess_query(query)
        print(f"Normalized: {result['normalized_query']}")
        print(f"Intent: {result['query_intent']}")
        print(f"Key Concepts (Top 3): {result['key_concepts'][:3]}") 
        print(f"Search Queries (Top 2 with weights): {result['search_queries'][:2]}") 
        # print(f"All Search Queries: {result['search_queries']}") # Can be verbose
        # print(f"Entities: {result['entities']}") # Can be verbose
        print("-" * 80)