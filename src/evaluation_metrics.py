# src/evaluation_metrics.py

import os
import json
import time
import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
import google.generativeai as genai
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@dataclass
class RetrievalMetrics:
    """Metrics for retrieval quality evaluation."""
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    f1_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    map_score: float = 0.0
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    retrieval_time: float = 0.0
    semantic_similarity: float = 0.0


@dataclass
class AnswerMetrics:
    """Metrics for answer quality evaluation."""
    bleu_score: float = 0.0
    rouge_scores: Dict[str, float] = field(default_factory=dict)
    semantic_similarity: float = 0.0
    citation_accuracy: float = 0.0
    citation_completeness: float = 0.0
    answer_relevance: float = 0.0
    factual_accuracy: float = 0.0
    coherence_score: float = 0.0
    generation_time: float = 0.0


@dataclass
class QueryResult:
    """Complete result for a single query evaluation."""
    query: str
    ground_truth_docs: List[str]
    retrieved_docs: List[str]
    generated_answer: str
    expected_answer: Optional[str]
    retrieval_metrics: RetrievalMetrics
    answer_metrics: AnswerMetrics
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class RAGEvaluator:
    """Comprehensive evaluation system for RAG pipeline."""
    
    def __init__(self, 
                 collection: chromadb.Collection,
                 embedding_function,
                 llm_model_name: str = "gemini-1.5-flash-latest",
                 api_key: Optional[str] = None):
        self.collection = collection
        self.embedding_function = embedding_function
        self.llm_model_name = llm_model_name
        
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(llm_model_name)
        else:
            self.model = None
        
        # Cache for embeddings
        self.embedding_cache = {}
        
    def evaluate_query(self,
                      query: str,
                      ground_truth_docs: List[str],
                      expected_answer: Optional[str] = None,
                      k_values: List[int] = [3, 5, 10]) -> QueryResult:
        """Evaluate a single query through the RAG pipeline."""
        start_time = time.time()
        retrieval_results = self.collection.query(
            query_texts=[query],
            n_results=max(k_values),
            include=['documents', 'metadatas', 'distances', 'embeddings']
        )
        retrieval_time = time.time() - start_time
        
        retrieved_docs = []
        retrieved_texts = []
        retrieved_embeddings = []
        
        _metadatas_for_query0 = None
        if retrieval_results.get('metadatas') and len(retrieval_results['metadatas']) > 0:
            _metadatas_for_query0 = retrieval_results['metadatas'][0]
        
        _documents_for_query0 = None
        if retrieval_results.get('documents') and len(retrieval_results['documents']) > 0:
            _documents_for_query0 = retrieval_results['documents'][0]

        _embeddings_container_for_query0 = None 
        _embeddings_payload = retrieval_results.get('embeddings')
        if _embeddings_payload and len(_embeddings_payload) > 0:
            _embeddings_container_for_query0 = _embeddings_payload[0]

        if _metadatas_for_query0:
            for i, meta in enumerate(_metadatas_for_query0):
                if meta and isinstance(meta, dict) and 'document_id' in meta:
                    retrieved_docs.append(meta['document_id'])
                else:
                    retrieved_docs.append(f"unknown_doc_id_at_index_{i}")

                current_doc_text = ""
                if _documents_for_query0 and i < len(_documents_for_query0) and _documents_for_query0[i] is not None:
                    current_doc_text = _documents_for_query0[i]
                retrieved_texts.append(current_doc_text)

                doc_i_embedding_candidate = None
                if _embeddings_container_for_query0 is not None:
                    if isinstance(_embeddings_container_for_query0, list):
                        if i < len(_embeddings_container_for_query0):
                            doc_i_embedding_candidate = _embeddings_container_for_query0[i]
                    elif isinstance(_embeddings_container_for_query0, np.ndarray):
                        if _embeddings_container_for_query0.ndim == 2 and i < _embeddings_container_for_query0.shape[0]:
                            doc_i_embedding_candidate = _embeddings_container_for_query0[i]
                        elif _embeddings_container_for_query0.ndim == 1 and i == 0:
                            doc_i_embedding_candidate = _embeddings_container_for_query0
                
                if doc_i_embedding_candidate is not None:
                    try:
                        valid_np_embedding = np.array(doc_i_embedding_candidate, dtype=np.float32)
                        retrieved_embeddings.append(valid_np_embedding)
                    except ValueError as e:
                        print(f"Warning: Could not convert embedding to np.array for doc index {i}. Error: {e}")
                        pass 
        
        query_emb_for_metrics = self._get_embedding(query)
        retrieval_metrics = self._calculate_retrieval_metrics(
            retrieved_docs=retrieved_docs,
            ground_truth_docs=ground_truth_docs,
            query_embedding=query_emb_for_metrics,
            retrieved_embeddings=retrieved_embeddings,
            k_values=k_values,
            retrieval_time=retrieval_time
        )
        
        answer_metrics = AnswerMetrics()
        generated_answer = ""
        
        if self.model and retrieved_texts:
            start_time_gen = time.time()
            context_texts_for_generation = retrieved_texts[:min(len(retrieved_texts), 5)]
            if context_texts_for_generation:
                generated_answer = self._generate_answer(query, context_texts_for_generation)
            generation_time = time.time() - start_time_gen
            
            answer_metrics = self._calculate_answer_metrics(
                generated_answer=generated_answer,
                expected_answer=expected_answer,
                query=query,
                retrieved_texts=context_texts_for_generation, 
                generation_time=generation_time
            )
        
        return QueryResult(
            query=query,
            ground_truth_docs=ground_truth_docs,
            retrieved_docs=retrieved_docs,
            generated_answer=generated_answer,
            expected_answer=expected_answer,
            retrieval_metrics=retrieval_metrics,
            answer_metrics=answer_metrics
        )
    
    def _calculate_retrieval_metrics(self,
                                   retrieved_docs: List[str],
                                   ground_truth_docs: List[str],
                                   query_embedding: np.ndarray,
                                   retrieved_embeddings: List[np.ndarray],
                                   k_values: List[int],
                                   retrieval_time: float) -> RetrievalMetrics:
        metrics = RetrievalMetrics(retrieval_time=float(retrieval_time)) # Ensure time is float
        ground_truth_set = set(ground_truth_docs)
        
        for k in k_values:
            metrics.precision_at_k[k] = 0.0
            metrics.recall_at_k[k] = 0.0
            metrics.f1_at_k[k] = 0.0
            metrics.ndcg_at_k[k] = 0.0

            retrieved_at_k = retrieved_docs[:k]
            retrieved_set_k = set(retrieved_at_k)
            
            if k > 0:
                metrics.precision_at_k[k] = float(len(retrieved_set_k & ground_truth_set) / k)
            
            if ground_truth_set:
                metrics.recall_at_k[k] = float(len(retrieved_set_k & ground_truth_set) / len(ground_truth_set))
            
            current_precision_k = metrics.precision_at_k[k]
            current_recall_k = metrics.recall_at_k[k]
            
            if current_precision_k + current_recall_k > 0:
                metrics.f1_at_k[k] = float(2 * (current_precision_k * current_recall_k) / \
                                    (current_precision_k + current_recall_k))
            
            metrics.ndcg_at_k[k] = float(self._calculate_ndcg(retrieved_at_k, ground_truth_set, k))
        
        for i, doc in enumerate(retrieved_docs):
            if doc in ground_truth_set:
                metrics.mrr = float(1 / (i + 1))
                break
        
        metrics.map_score = float(self._calculate_map(retrieved_docs, ground_truth_set))
        
        if retrieved_embeddings:
            num_embeddings_for_sim = min(len(retrieved_embeddings), 5)
            if num_embeddings_for_sim > 0:
                similarities = [
                    cosine_similarity([query_embedding], [emb])[0][0]
                    for emb in retrieved_embeddings[:num_embeddings_for_sim]
                ]
                metrics.semantic_similarity = float(np.mean(similarities)) # CAST
        
        return metrics
    
    def _calculate_ndcg(self, retrieved_docs: List[str], ground_truth_set: Set[str], k: int) -> float:
        def dcg(relevances: List[float], k_val: int) -> float:
            return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances[:k_val]))
        
        relevances = [1.0 if doc in ground_truth_set else 0.0 for doc in retrieved_docs] # Use float
        ideal_relevances = sorted([1.0 if doc in ground_truth_set else 0.0 for doc in retrieved_docs], reverse=True)

        dcg_score = dcg(relevances, k)
        idcg_score = dcg(ideal_relevances, k)
        
        return float(dcg_score / idcg_score) if idcg_score > 0 else 0.0 # CAST
    
    def _calculate_map(self, retrieved_docs: List[str], ground_truth_set: Set[str]) -> float:
        if not ground_truth_set:
            return 0.0
        
        precisions = []
        relevant_count = 0
        
        for i, doc in enumerate(retrieved_docs):
            if doc in ground_truth_set:
                relevant_count += 1
                precisions.append(float(relevant_count / (i + 1))) # Ensure float
        
        return float(sum(precisions) / len(ground_truth_set)) if precisions else 0.0 # CAST
    
    def _calculate_answer_metrics(self,
                                generated_answer: str,
                                expected_answer: Optional[str],
                                query: str,
                                retrieved_texts: List[str],
                                generation_time: float) -> AnswerMetrics:
        metrics = AnswerMetrics(generation_time=float(generation_time)) # Ensure time is float
        
        if expected_answer:
            reference_tokens = word_tokenize(expected_answer.lower())
            candidate_tokens = word_tokenize(generated_answer.lower())
            if not candidate_tokens:
                metrics.bleu_score = 0.0
            else:
                metrics.bleu_score = float(sentence_bleu([reference_tokens], candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25))) # CAST
            
            metrics.rouge_scores = self._calculate_rouge_scores(generated_answer, expected_answer) # rouge scores are already floats
            
            if generated_answer.strip() and expected_answer.strip():
                gen_embedding = self._get_embedding(generated_answer)
                exp_embedding = self._get_embedding(expected_answer)
                metrics.semantic_similarity = float(cosine_similarity([gen_embedding], [exp_embedding])[0][0]) # CAST
            else:
                metrics.semantic_similarity = 0.0
        
        citation_results = self._evaluate_citations(generated_answer, retrieved_texts) # citation results are floats
        metrics.citation_accuracy = citation_results['accuracy']
        metrics.citation_completeness = citation_results['completeness']
        
        if generated_answer.strip() and query.strip():
            query_embedding = self._get_embedding(query)
            answer_embedding = self._get_embedding(generated_answer)
            metrics.answer_relevance = float(cosine_similarity([query_embedding], [answer_embedding])[0][0]) # CAST
        else:
            metrics.answer_relevance = 0.0
        
        if self.model and generated_answer.strip() and retrieved_texts:
            metrics.factual_accuracy = float(self._evaluate_factual_accuracy( # CAST (already returns float, but good practice)
                generated_answer, retrieved_texts
            ))
        else:
            metrics.factual_accuracy = 0.0
        
        if generated_answer.strip():
            metrics.coherence_score = self._evaluate_coherence(generated_answer) # Already returns float
        else:
            metrics.coherence_score = 0.0
        
        return metrics
    
    def _calculate_rouge_scores(self, generated: str, reference: str) -> Dict[str, float]:
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            return {
                'rouge1_f1': 0.0,
                'rouge2_f1': 0.0,
                'rougeL_f1': 0.0
            }

        if not generated.strip() or not reference.strip():
            return {
                'rouge1_f1': 0.0,
                'rouge2_f1': 0.0,
                'rougeL_f1': 0.0
            }

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, generated)
        
        return { # These fmeasure values are typically Python floats
            'rouge1_f1': float(scores['rouge1'].fmeasure),
            'rouge2_f1': float(scores['rouge2'].fmeasure),
            'rougeL_f1': float(scores['rougeL'].fmeasure)
        }
    
    def _evaluate_citations(self, answer: str, source_texts: List[str]) -> Dict[str, float]:
        """
        Updated citation evaluation that handles Context references from evaluation prompts.
        Handles both:
        1. Full format: [In Re: 36061766, 2025-02-03, FEB032025_02B2203, Chunk ID: c2b612b5]
        2. Context format: [Context 1], [Context 1, Context 3, Context 5]
        """
        # Multiple citation patterns to try
        citation_patterns = [
            # Context reference patterns (what the evaluation actually generates)
            (r'\[Context (\d+(?:,\s*Context \d+)*)\]', 'context_multi'),  # [Context 1, Context 3]
            (r'\[Context (\d+)\]', 'context_single'),  # [Context 1]
            # Full format: [Case Name, Date, Document ID, Chunk ID: chunk_value]
            (r'\[([^,]+),\s*([^,]+),\s*([^,]+),\s*Chunk ID:\s*([^\]]+)\]', 'full_format'),
            # Any square brackets as fallback
            (r'\[([^\]]+)\]', 'generic'),
        ]
        
        citations_extracted = []
        citation_details = []
        pattern_used = None
        
        # Try each pattern
        for pattern, pattern_type in citation_patterns:
            matches = re.findall(pattern, answer)
            if matches:
                pattern_used = pattern_type
                
                if pattern_type == 'context_multi':
                    # Handle [Context 1, Context 3, Context 5]
                    for match in matches:
                        # Extract all context numbers
                        context_nums = re.findall(r'\d+', match)
                        for num in context_nums:
                            citations_extracted.append({
                                'type': 'context',
                                'context_num': int(num),
                                'full': f'Context {num}'
                            })
                elif pattern_type == 'context_single':
                    # Handle [Context 1]
                    for match in matches:
                        citations_extracted.append({
                            'type': 'context',
                            'context_num': int(match),
                            'full': f'Context {match}'
                        })
                elif pattern_type == 'full_format':
                    # Handle full citation format
                    for match in matches:
                        case_name = match[0].strip()
                        date = match[1].strip()
                        doc_id = match[2].strip()
                        chunk_id = match[3].strip()
                        
                        citations_extracted.append({
                            'type': 'full',
                            'full': f"{case_name}, {date}, {doc_id}, Chunk ID: {chunk_id}",
                            'case': case_name,
                            'date': date,
                            'doc_id': doc_id,
                            'chunk_id': chunk_id
                        })
                else:
                    # Generic bracket content
                    for match in matches:
                        citations_extracted.append({'type': 'generic', 'full': match})
                
                if citations_extracted:
                    break  # Found citations, stop trying patterns
        
        if not citations_extracted:
            # Check if the answer appears to make factual claims without citations
            fact_indicators = ['requires', 'must', 'needs', 'evaluated', 'determined', 'constitutes', 
                             'defined as', 'considers', 'is not considered', 'in contrast', 'demonstrates',
                             'is', 'are', 'was', 'were', 'shows', 'indicates', 'states']
            
            # More sophisticated check for factual content
            sentences = nltk.sent_tokenize(answer)
            has_factual_sentences = False
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                # Check if sentence has fact indicators and is not a question
                if any(indicator in sentence_lower for indicator in fact_indicators) and not sentence.strip().endswith('?'):
                    # Check if it's a substantial statement (more than 5 words)
                    if len(word_tokenize(sentence)) > 5:
                        has_factual_sentences = True
                        break
            
            if has_factual_sentences:
                # Has factual claims but no citations
                return {'accuracy': 0.0, 'completeness': 0.0}
            else:
                # No clear factual claims or very short answer
                return {'accuracy': 1.0, 'completeness': 1.0}
        
        # Validate citations against source texts
        valid_citations = 0
        total_citations = len(citations_extracted)
        
        for citation in citations_extracted:
            citation_found = False
            
            if citation['type'] == 'context':
                # For Context references, check if the context number is valid
                context_num = citation['context_num']
                # Context numbers are 1-based, source_texts are 0-based
                if 1 <= context_num <= len(source_texts):
                    citation_found = True
            
            elif citation['type'] == 'full':
                # For full citations, check if components match source texts
                doc_id = citation.get('doc_id', '')
                chunk_id = citation.get('chunk_id', '')
                case_name = citation.get('case', '')
                
                # Check if citation matches any source
                for source in source_texts:
                    matches = [
                        # Direct document ID match
                        doc_id in source,
                        # Case pattern match
                        re.search(rf'{re.escape(doc_id)}', source, re.IGNORECASE) is not None,
                        # Chunk ID might appear in the source
                        chunk_id and chunk_id in source,
                        # Case name might appear
                        case_name and case_name in source,
                        # Try partial matches for document ID components
                        any(part in source for part in doc_id.split('_') if len(part) > 4),
                    ]
                    
                    if any(matches):
                        citation_found = True
                        break
            
            else:
                # For generic citations, check if they appear in sources
                citation_text = citation.get('full', '')
                for source in source_texts:
                    if citation_text.lower() in source.lower():
                        citation_found = True
                        break
            
            if citation_found:
                valid_citations += 1
        
        accuracy = float(valid_citations / total_citations) if total_citations > 0 else 0.0
        
        # Evaluate completeness - are factual statements properly cited?
        sentences = nltk.sent_tokenize(answer)
        fact_keywords = ['is', 'are', 'was', 'were', 'states', 'shows', 'indicates', 
                        'requires', 'must', 'needs', 'should', 'evaluated', 'determined',
                        'constitutes', 'consists', 'includes', 'defined', 'means', 
                        'considers', 'considered', 'sufficient', 'not considered', 'demonstrate']
        
        # Identify sentences that likely contain facts needing citations
        fact_sentences = []
        for s in sentences:
            s_lower = s.lower()
            # Check if sentence has fact keywords and is substantial
            if any(kw in s_lower for kw in fact_keywords) and len(word_tokenize(s)) > 5:
                # Also check it's not a question or hypothetical
                if not s.strip().endswith('?') and 'if ' not in s_lower[:10]:
                    fact_sentences.append(s)
        
        if not fact_sentences:
            completeness = 1.0  # No factual claims to cite
        else:
            cited_facts_count = 0
            # Check if each fact sentence has a citation
            for fact_sent in fact_sentences:
                # Check for any form of citation in or near the sentence
                has_citation = False
                
                # Check for citations within the sentence
                for pattern, _ in citation_patterns:
                    if re.search(pattern, fact_sent):
                        has_citation = True
                        break
                
                if not has_citation:
                    # Check if citation appears at end of previous or next sentence
                    sentence_index = sentences.index(fact_sent) if fact_sent in sentences else -1
                    
                    # Check previous sentence
                    if sentence_index > 0:
                        prev_sentence = sentences[sentence_index - 1]
                        for pattern, _ in citation_patterns:
                            if re.search(pattern, prev_sentence[-50:]):  # Check end of previous sentence
                                has_citation = True
                                break
                    
                    # Check next sentence
                    if not has_citation and sentence_index < len(sentences) - 1:
                        next_sentence = sentences[sentence_index + 1]
                        for pattern, _ in citation_patterns:
                            if re.search(pattern, next_sentence[:50]):  # Check beginning of next sentence
                                has_citation = True
                                break
                
                if has_citation:
                    cited_facts_count += 1
            
            completeness = float(cited_facts_count / len(fact_sentences))
        
        return {
            'accuracy': accuracy,
            'completeness': completeness,
            'details': {
                'total_citations': total_citations,
                'valid_citations': valid_citations,
                'fact_sentences': len(fact_sentences),
                'cited_facts': cited_facts_count if 'cited_facts_count' in locals() else 0,
                'pattern_used': pattern_used
            }
        }
    
    def _evaluate_factual_accuracy(self, answer: str, source_texts: List[str]) -> float:
        if not self.model: return 0.5

        context_for_eval = "\n".join(source_texts[:3])
        
        prompt = f"""
        You are an impartial judge. Evaluate the factual accuracy of the "Generated Answer" based *only* on the provided "Source Texts".
        Ignore any information not present in the Source Texts.
        
        Source Texts:
        {context_for_eval}
        
        Generated Answer:
        {answer}
        
        Consider each statement in the "Generated Answer".
        - If all statements in the "Generated Answer" are directly and verifiably supported by the "Source Texts", rate it 1.0.
        - If some statements are supported but others are not, or if statements contradict the "Source Texts", rate it between 0.0 and 1.0 based on the proportion and severity of inaccuracies.
        - If the "Generated Answer" contains significant factual errors or fabrications not supported by the "Source Texts", rate it 0.0.
        - If the "Generated Answer" is too vague or makes no verifiable claims based on the sources, it may also receive a lower score.

        Respond with a single floating-point number between 0.0 and 1.0. Do not provide any explanation, just the number.
        Factual Accuracy Score:
        """
        
        try:
            response = self.model.generate_content(prompt)
            match = re.search(r"(\d\.\d+)", response.text)
            if match:
                score = float(match.group(1)) # Already float
                return max(0.0, min(1.0, score))
            else:
                return 0.5 
        except Exception:
            return 0.5
    
    def _evaluate_coherence(self, text: str) -> float:
        sentences = nltk.sent_tokenize(text)
        sentences = [s.strip() for s in sentences if s.strip() and len(word_tokenize(s)) > 3]
        
        if len(sentences) < 2:
            return 1.0
        
        similarities = []
        for i in range(len(sentences) - 1):
            if sentences[i].strip() and sentences[i+1].strip():
                emb1 = self._get_embedding(sentences[i])
                emb2 = self._get_embedding(sentences[i + 1])
                similarity = cosine_similarity([emb1], [emb2])[0][0]
                similarities.append(similarity) # similarity is np.float32
        
        return float(np.mean(similarities)) if similarities else 0.0 # CAST np.mean output
    
    def _get_embedding(self, text: str) -> np.ndarray:
        normalized_text = text.strip().lower()
        if normalized_text in self.embedding_cache:
            return self.embedding_cache[normalized_text]
        
        if not normalized_text:
            try:
                sample_emb = self.embedding_function(["sample text"])[0]
                embedding = np.zeros_like(sample_emb, dtype=np.float32)
            except: 
                embedding = np.zeros(384, dtype=np.float32) # Default, ensure dtype
            self.embedding_cache[normalized_text] = embedding
            return embedding

        embedding_list = self.embedding_function([text])
        # Ensure the embedding is a np.ndarray of np.float32
        if embedding_list and embedding_list[0] is not None:
            embedding = np.array(embedding_list[0], dtype=np.float32)
        else: # Fallback if embedding function returns None or empty
             try:
                sample_emb = self.embedding_function(["sample text"])[0]
                embedding = np.zeros_like(sample_emb, dtype=np.float32)
             except:
                embedding = np.zeros(384, dtype=np.float32)

        self.embedding_cache[normalized_text] = embedding
        return embedding
    
    def _generate_answer(self, query: str, contexts: List[str]) -> str:
        if not self.model: return "LLM model not available."
        if not contexts: return "No context provided to generate an answer."

        context_str = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""
        You are a helpful assistant. Answer the following query based *only* on the provided contexts.
        If the contexts do not provide enough information to answer the query, state that the information is not available in the provided contexts.
        Cite the source context number (e.g., [Context 1]) for each piece of information you use from the contexts.
        
        Query: {query}
        
        Contexts:
        {context_str}
        
        Answer:
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def evaluate_test_set(self, test_queries: List[Dict]) -> Dict:
        results = []
        
        for test_case in tqdm(test_queries, desc="Evaluating queries"):
            result = self.evaluate_query(
                query=test_case['query'],
                ground_truth_docs=test_case.get('relevant_documents', []),
                expected_answer=test_case.get('expected_answer')
            )
            results.append(result)
        
        aggregated = self._aggregate_results(results)
        
        return {
            'individual_results': results,
            'aggregated_metrics': aggregated,
            'evaluation_timestamp': datetime.now().isoformat()
        }
    
    def _aggregate_results(self, results: List[QueryResult]) -> Dict:
        retrieval_metrics_aggr = defaultdict(list)
        answer_metrics_aggr = defaultdict(list)
        
        for result in results:
            for k_val in result.retrieval_metrics.precision_at_k:
                retrieval_metrics_aggr[f'precision_at_{k_val}'].append(
                    result.retrieval_metrics.precision_at_k[k_val]
                )
                retrieval_metrics_aggr[f'recall_at_{k_val}'].append(
                    result.retrieval_metrics.recall_at_k[k_val]
                )
                retrieval_metrics_aggr[f'f1_at_{k_val}'].append(
                    result.retrieval_metrics.f1_at_k[k_val]
                )
                retrieval_metrics_aggr[f'ndcg_at_{k_val}'].append(
                    result.retrieval_metrics.ndcg_at_k[k_val]
                )
            
            retrieval_metrics_aggr['mrr'].append(result.retrieval_metrics.mrr)
            retrieval_metrics_aggr['map'].append(result.retrieval_metrics.map_score)
            retrieval_metrics_aggr['retrieval_time'].append(result.retrieval_metrics.retrieval_time)
            if result.retrieval_metrics.semantic_similarity is not None:
                 retrieval_metrics_aggr['semantic_similarity'].append(
                    result.retrieval_metrics.semantic_similarity
                )
            
            if result.answer_metrics.generation_time > 0:
                answer_metrics_aggr['bleu_score'].append(result.answer_metrics.bleu_score)
                if result.answer_metrics.semantic_similarity is not None:
                    answer_metrics_aggr['semantic_similarity'].append(
                        result.answer_metrics.semantic_similarity
                    )
                answer_metrics_aggr['citation_accuracy'].append(
                    result.answer_metrics.citation_accuracy
                )
                answer_metrics_aggr['citation_completeness'].append(
                    result.answer_metrics.citation_completeness
                )
                answer_metrics_aggr['answer_relevance'].append(
                    result.answer_metrics.answer_relevance
                )
                answer_metrics_aggr['factual_accuracy'].append(
                    result.answer_metrics.factual_accuracy
                )
                answer_metrics_aggr['coherence_score'].append(
                    result.answer_metrics.coherence_score
                )
                answer_metrics_aggr['generation_time'].append(
                    result.answer_metrics.generation_time
                )
                
                for rouge_metric, score in result.answer_metrics.rouge_scores.items():
                    answer_metrics_aggr[rouge_metric].append(score)
        
        aggregated = {
            'retrieval': {},
            'answer': {},
            'overall': {}
        }
        
        for metric, values in retrieval_metrics_aggr.items():
            if values:
                aggregated['retrieval'][f'mean_{metric}'] = float(np.mean(values)) # CAST
                aggregated['retrieval'][f'std_{metric}'] = float(np.std(values))   # CAST
            else:
                aggregated['retrieval'][f'mean_{metric}'] = 0.0
                aggregated['retrieval'][f'std_{metric}'] = 0.0

        for metric, values in answer_metrics_aggr.items():
            if values:
                aggregated['answer'][f'mean_{metric}'] = float(np.mean(values)) # CAST
                aggregated['answer'][f'std_{metric}'] = float(np.std(values))   # CAST
            else:
                aggregated['answer'][f'mean_{metric}'] = 0.0
                aggregated['answer'][f'std_{metric}'] = 0.0
        
        rag_score_k = 5 
        mean_f1_at_k_key = f'mean_f1_at_{rag_score_k}'
        
        f1_for_rag_score = aggregated['retrieval'].get(mean_f1_at_k_key, 0.0)
        relevance_for_rag_score = aggregated['answer'].get('mean_answer_relevance', 0.0)
        citation_acc_for_rag_score = aggregated['answer'].get('mean_citation_accuracy', 0.0)

        if f1_for_rag_score > 0 or relevance_for_rag_score > 0 or citation_acc_for_rag_score > 0 :
            rag_score_val = (
                f1_for_rag_score * 0.4 +
                relevance_for_rag_score * 0.3 +
                citation_acc_for_rag_score * 0.3
            )
            aggregated['overall']['rag_score'] = float(rag_score_val) # CAST
        else:
            aggregated['overall']['rag_score'] = 0.0
        
        return aggregated

    def visualize_results(self, evaluation_results: Dict, output_dir: str = "evaluation_results"):
        """Create comprehensive visualizations of evaluation results."""
        os.makedirs(output_dir, exist_ok=True)
        
        aggregated = evaluation_results['aggregated_metrics']
        
        # 1. Retrieval Metrics Bar Chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7)) 
        
        k_values_plot = []
        precisions = []
        recalls = []
        f1_scores = []
        
        present_k_values = set()
        for key in aggregated['retrieval']:
            if key.startswith('mean_precision_at_'):
                try:
                    k = int(key.split('_')[-1])
                    present_k_values.add(k)
                except ValueError:
                    continue 
        
        sorted_k_values = sorted(list(present_k_values))

        for k_val_plot in sorted_k_values:
            if f'mean_precision_at_{k_val_plot}' in aggregated['retrieval']:
                k_values_plot.append(k_val_plot)
                precisions.append(aggregated['retrieval'].get(f'mean_precision_at_{k_val_plot}', 0.0))
                recalls.append(aggregated['retrieval'].get(f'mean_recall_at_{k_val_plot}', 0.0))
                f1_scores.append(aggregated['retrieval'].get(f'mean_f1_at_{k_val_plot}', 0.0))
        
        if k_values_plot: 
            x = np.arange(len(k_values_plot))
            width = 0.25
            
            ax1.bar(x - width, precisions, width, label='Precision', color='skyblue')
            ax1.bar(x, recalls, width, label='Recall', color='lightgreen')
            ax1.bar(x + width, f1_scores, width, label='F1', color='coral')
            
            ax1.set_xlabel('k')
            ax1.set_ylabel('Score')
            ax1.set_title('Retrieval Metrics at Different k Values')
            ax1.set_xticks(x)
            ax1.set_xticklabels(k_values_plot)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1.05) 
        else:
            ax1.text(0.5, 0.5, "No k-based retrieval metrics found.", ha='center', va='center')
            ax1.set_title('Retrieval Metrics at Different k Values')

        answer_metrics_plot_data = [
            ('BLEU', aggregated['answer'].get('mean_bleu_score', 0.0)),
            ('Rouge1-F1', aggregated['answer'].get('mean_rouge1_f1', 0.0)), 
            ('Semantic Sim', aggregated['answer'].get('mean_semantic_similarity', 0.0)),
            ('Citation Acc', aggregated['answer'].get('mean_citation_accuracy', 0.0)),
            ('Answer Rel', aggregated['answer'].get('mean_answer_relevance', 0.0)),
            ('Factual Acc', aggregated['answer'].get('mean_factual_accuracy', 0.0)),
            ('Coherence', aggregated['answer'].get('mean_coherence_score', 0.0))
        ]
        
        metrics_names = [m[0] for m in answer_metrics_plot_data]
        metrics_values = [m[1] for m in answer_metrics_plot_data] # These are now Python floats
        
        if any(metrics_values): 
            ax2.barh(metrics_names, metrics_values, color='gold')
            ax2.set_xlabel('Score')
            ax2.set_title('Mean Answer Quality Metrics')
            ax2.set_xlim(0, 1)
            ax2.grid(True, alpha=0.3, axis='x')
            for index, value in enumerate(metrics_values): 
                ax2.text(value + 0.01, index, f"{value:.2f}")
        else:
            ax2.text(0.5, 0.5, "No answer quality metrics found.", ha='center', va='center')
            ax2.set_title('Mean Answer Quality Metrics')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_overview.png'), dpi=300)
        plt.close()
        
        if 'individual_results' in evaluation_results and evaluation_results['individual_results']:
            results_list = evaluation_results['individual_results'] 
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 12)) 
            fig.suptitle("Performance Distributions", fontsize=16)
            
            mrr_values = [r.retrieval_metrics.mrr for r in results_list if r.retrieval_metrics.mrr is not None]
            if mrr_values:
                sns.histplot(mrr_values, bins=10, kde=True, color='purple', ax=axes[0, 0])
                axes[0, 0].set_title('MRR Distribution')
                axes[0, 0].set_xlabel('MRR')
                axes[0, 0].set_ylabel('Frequency')
            else:
                axes[0,0].text(0.5, 0.5, "No MRR data.", ha='center', va='center')

            retrieval_times = [r.retrieval_metrics.retrieval_time for r in results_list if r.retrieval_metrics.retrieval_time is not None]
            if retrieval_times:
                sns.histplot(retrieval_times, bins=10, kde=True, color='orange', ax=axes[0, 1])
                axes[0, 1].set_title('Retrieval Time (s) Distribution')
                axes[0, 1].set_xlabel('Time (seconds)')
            else:
                axes[0,1].text(0.5, 0.5, "No retrieval time data.", ha='center', va='center')

            citation_acc = [r.answer_metrics.citation_accuracy for r in results_list 
                          if r.answer_metrics.generation_time > 0 and r.answer_metrics.citation_accuracy is not None]
            if citation_acc:
                sns.histplot(citation_acc, bins=10, kde=True, color='green', ax=axes[1, 0])
                axes[1, 0].set_title('Citation Accuracy Distribution')
                axes[1, 0].set_xlabel('Accuracy')
                axes[1, 0].set_xlim(0,1)
            else:
                axes[1,0].text(0.5, 0.5, "No citation accuracy data.", ha='center', va='center')
            
            answer_rel = [r.answer_metrics.answer_relevance for r in results_list 
                         if r.answer_metrics.generation_time > 0 and r.answer_metrics.answer_relevance is not None]
            if answer_rel:
                sns.histplot(answer_rel, bins=10, kde=True, color='red', ax=axes[1, 1])
                axes[1, 1].set_title('Answer Relevance Distribution')
                axes[1, 1].set_xlabel('Relevance Score')
                axes[1, 1].set_xlim(0,1)
            else:
                axes[1,1].text(0.5, 0.5, "No answer relevance data.", ha='center', va='center')
            
            plt.tight_layout(rect=[0, 0, 1, 0.96]) 
            plt.savefig(os.path.join(output_dir, 'performance_distributions.png'), dpi=300)
            plt.close()
        
        self._create_error_analysis(evaluation_results, output_dir)
        print(f"Visualizations saved to {output_dir}/")
    
    def _create_error_analysis(self, evaluation_results: Dict, output_dir: str):
        if 'individual_results' not in evaluation_results or not evaluation_results['individual_results']:
            print("No individual results for error analysis.")
            return
        
        results_list_err = evaluation_results['individual_results'] 
        
        retrieval_failures = []
        citation_failures = []
        relevance_failures = []
        
        RETRIEVAL_F1_THRESHOLD = 0.5
        CITATION_ACC_THRESHOLD = 0.5
        ANSWER_REL_THRESHOLD = 0.5
        F1_K_FOR_FAILURE = 5

        for result in results_list_err:
            if result.retrieval_metrics.f1_at_k.get(F1_K_FOR_FAILURE, 1.0) < RETRIEVAL_F1_THRESHOLD:
                retrieval_failures.append({
                    'query': result.query,
                    f'f1_score_at_{F1_K_FOR_FAILURE}': result.retrieval_metrics.f1_at_k.get(F1_K_FOR_FAILURE, 0.0),
                    'retrieved_top_k': result.retrieved_docs[:F1_K_FOR_FAILURE],
                    'ground_truth': result.ground_truth_docs
                })
            
            if result.answer_metrics.generation_time > 0 and result.answer_metrics.citation_accuracy < CITATION_ACC_THRESHOLD:
                citation_failures.append({
                    'query': result.query,
                    'accuracy': result.answer_metrics.citation_accuracy,
                    'answer': result.generated_answer[:200] + "..." 
                })
            
            if result.answer_metrics.generation_time > 0 and result.answer_metrics.answer_relevance < ANSWER_REL_THRESHOLD:
                relevance_failures.append({
                    'query': result.query,
                    'relevance': result.answer_metrics.answer_relevance,
                    'answer': result.generated_answer[:200] + "..." 
                })
        
        error_summary = {
            'total_queries': len(results_list_err),
            'retrieval_failure_threshold': f'F1@{F1_K_FOR_FAILURE} < {RETRIEVAL_F1_THRESHOLD}',
            'citation_failure_threshold': f'Accuracy < {CITATION_ACC_THRESHOLD}',
            'relevance_failure_threshold': f'Relevance < {ANSWER_REL_THRESHOLD}',
            'retrieval_failures_count': len(retrieval_failures),
            'citation_failures_count': len(citation_failures),
            'relevance_failures_count': len(relevance_failures),
            'failure_examples': { 
                'retrieval': retrieval_failures[:3],
                'citation': citation_failures[:3],
                'relevance': relevance_failures[:3]
            }
        }
        
        error_analysis_path = os.path.join(output_dir, 'error_analysis.json')
        with open(error_analysis_path, 'w') as f:
            json.dump(error_summary, f, indent=2) # This dump should be fine as values are primitive or lists/dicts of primitives
        print(f"Error analysis report saved to {error_analysis_path}")

        plt.figure(figsize=(8, 8))
        failure_types = ['Retrieval Failures', 'Citation Failures', 'Relevance Failures']
        failure_counts = [
            len(retrieval_failures),
            len(citation_failures),
            len(relevance_failures)
        ]
        
        if sum(failure_counts) > 0:
            plt.pie(failure_counts, labels=failure_types, autopct='%1.1f%%', 
                   colors=['#ff9999', '#66b3ff', '#99ff99'], startangle=90)
            plt.title('Distribution of Query Failure Types')
            plt.savefig(os.path.join(output_dir, 'failure_distribution.png'), dpi=300)
        else:
            print("No failures detected to plot in pie chart.")
        plt.close()
    
    def generate_evaluation_report(self, 
                                 evaluation_results: Dict,
                                 output_file: str = "evaluation_report.json") -> Dict:
        aggregated = evaluation_results['aggregated_metrics'] # These values are now Python floats
        
        report = {
            'evaluation_summary': {
                'timestamp': evaluation_results.get('evaluation_timestamp', datetime.now().isoformat()),
                'total_queries': len(evaluation_results.get('individual_results', [])),
                'overall_rag_score': aggregated.get('overall', {}).get('rag_score', 0.0)
            },
            'retrieval_performance': {
                'best_k_by_f1': self._find_best_k(aggregated.get('retrieval', {})),
                'mean_mrr': aggregated.get('retrieval', {}).get('mean_mrr', 0.0),
                'mean_map': aggregated.get('retrieval', {}).get('mean_map', 0.0),
                'avg_retrieval_time_sec': aggregated.get('retrieval', {}).get('mean_retrieval_time', 0.0)
            },
            'answer_quality': {
                'mean_bleu': aggregated.get('answer', {}).get('mean_bleu_score', 0.0),
                'mean_rouge1_f1': aggregated.get('answer', {}).get('mean_rouge1_f1', 0.0),
                'mean_rouge2_f1': aggregated.get('answer', {}).get('mean_rouge2_f1', 0.0),
                'mean_rougeL_f1': aggregated.get('answer', {}).get('mean_rougeL_f1', 0.0),
                'mean_citation_accuracy': aggregated.get('answer', {}).get('mean_citation_accuracy', 0.0),
                'mean_citation_completeness': aggregated.get('answer', {}).get('mean_citation_completeness', 0.0),
                'mean_answer_relevance': aggregated.get('answer', {}).get('mean_answer_relevance', 0.0),
                'mean_factual_accuracy': aggregated.get('answer', {}).get('mean_factual_accuracy', 0.0),
                'mean_coherence_score': aggregated.get('answer', {}).get('mean_coherence_score', 0.0),
                'avg_generation_time_sec': aggregated.get('answer', {}).get('mean_generation_time', 0.0)
            },
            'recommendations': self._generate_evaluation_recommendations(aggregated)
        }
        
        if 'retrieval' in aggregated:
            for k_val_report in self._get_k_values_from_metrics(aggregated['retrieval']):
                report['retrieval_performance'][f'precision_at_{k_val_report}'] = aggregated['retrieval'].get(f'mean_precision_at_{k_val_report}', 0.0)
                report['retrieval_performance'][f'recall_at_{k_val_report}'] = aggregated['retrieval'].get(f'mean_recall_at_{k_val_report}', 0.0)
                report['retrieval_performance'][f'f1_at_{k_val_report}'] = aggregated['retrieval'].get(f'mean_f1_at_{k_val_report}', 0.0)
                report['retrieval_performance'][f'ndcg_at_{k_val_report}'] = aggregated['retrieval'].get(f'mean_ndcg_at_{k_val_report}', 0.0)

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2) # This should now work
        
        print(f"Evaluation report saved to {output_file}")
        return report

    def _get_k_values_from_metrics(self, retrieval_metrics_dict: Dict) -> List[int]:
        k_s = set()
        for key in retrieval_metrics_dict:
            if key.startswith('mean_f1_at_') or key.startswith('mean_precision_at_'):
                try:
                    k_s.add(int(key.split('_')[-1]))
                except ValueError:
                    continue
        return sorted(list(k_s))

    def _find_best_k(self, retrieval_metrics_dict: Dict) -> int:
        best_k_val = 0 
        best_f1 = -1.0 
        
        found_k_values = self._get_k_values_from_metrics(retrieval_metrics_dict)
        if not found_k_values: return 5 

        for k_val_find in found_k_values:
            f1_score = retrieval_metrics_dict.get(f'mean_f1_at_{k_val_find}', 0.0)
            if f1_score > best_f1:
                best_f1 = f1_score
                best_k_val = k_val_find
        
        return best_k_val if best_k_val > 0 else (found_k_values[0] if found_k_values else 5)
    
    def _generate_evaluation_recommendations(self, aggregated: Dict) -> List[str]:
        recommendations = []
        ret_metrics = aggregated.get('retrieval', {})
        ans_metrics = aggregated.get('answer', {})

        if ret_metrics.get('mean_mrr', 0.0) < 0.5:
            recommendations.append(
                "Low Mean Reciprocal Rank (MRR) suggests relevant documents are often not ranked at the very top. "
                "Consider improving query understanding, embedding model fine-tuning, or re-ranking strategies."
            )
        
        best_k_f1 = self._find_best_k(ret_metrics)
        if ret_metrics.get(f'mean_f1_at_{best_k_f1}', 0.0) < 0.6:
             recommendations.append(
                f"Overall retrieval F1 score (e.g., F1@{best_k_f1}={ret_metrics.get(f'mean_f1_at_{best_k_f1}', 0.0):.2f}) is modest. "
                "Focus on improving both precision and recall. This might involve better chunking, metadata usage, or hybrid search."
            )

        if ret_metrics.get('mean_retrieval_time', 0.0) > 1.0: 
            recommendations.append(
                f"Average retrieval time ({ret_metrics.get('mean_retrieval_time', 0.0):.2f}s) is high. "
                "Optimize vector database queries, consider embedding caching, or explore more efficient indexing."
            )
        
        if ans_metrics.get('mean_bleu_score', 0.0) < 0.3 and ans_metrics.get('mean_rougeL_f1', 0.0) < 0.4 :
            recommendations.append(
                "Low BLEU/ROUGE scores indicate generated answers differ significantly from expected answers in wording/structure. "
                "Review prompt engineering for answer formatting, or if expected answers are gold standards, this might be acceptable if semantic meaning is preserved."
            )

        if ans_metrics.get('mean_citation_accuracy', 0.0) < 0.7:
            recommendations.append(
                f"Citation accuracy ({ans_metrics.get('mean_citation_accuracy', 0.0):.2f}) is suboptimal. "
                "Refine citation extraction logic in generated answers and ensure LLM is prompted clearly to cite specific sources."
            )
        
        if ans_metrics.get('mean_answer_relevance', 0.0) < 0.7:
            recommendations.append(
                f"Answer relevance to the query ({ans_metrics.get('mean_answer_relevance', 0.0):.2f}) could be improved. "
                "Ensure prompts strongly guide the LLM to stay on topic and directly address the query using provided context."
            )

        if ans_metrics.get('mean_factual_accuracy', 0.0) < 0.75 and self.model: 
            recommendations.append(
                f"Factual accuracy ({ans_metrics.get('mean_factual_accuracy', 0.0):.2f}) based on LLM judge is not ideal. "
                "This may indicate hallucination or misinterpretation of source documents. Strengthen grounding in prompts or use more robust context."
            )
        
        if ans_metrics.get('mean_coherence_score', 0.0) < 0.6:
            recommendations.append(
                f"Low coherence scores ({ans_metrics.get('mean_coherence_score', 0.0):.2f}) suggest answers may lack logical flow. "
                "Review context ordering, prompt structure for summarization, or LLM's ability to synthesize information smoothly."
            )
        
        if not recommendations:
            recommendations.append("Overall performance looks reasonable. Continue monitoring and consider A/B testing for further improvements.")

        return recommendations

if __name__ == "__main__":
    GEMINI_API_KEY_EXAMPLE = os.environ.get("GEMINI_API_KEY") 
    if not GEMINI_API_KEY_EXAMPLE:
        print("Warning: GEMINI_API_KEY not found. LLM-dependent metrics will not be calculated.")

    try:
        client = chromadb.PersistentClient(path="../vector_db") 
        embedding_function_instance = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-small-en-v1.5" 
        )
        collection_instance = client.get_collection(name="aao_decisions", embedding_function=embedding_function_instance)
    except Exception as e:
        print(f"Error initializing ChromaDB or getting collection: {e}")
        print("Please ensure ChromaDB is set up and the collection 'aao_decisions' exists with the correct embedding function.")
        collection_instance = None

    if collection_instance:
        evaluator = RAGEvaluator(
            collection=collection_instance,
            embedding_function=embedding_function_instance, 
            api_key=GEMINI_API_KEY_EXAMPLE
        )
        
        test_queries_example = [
            {
                'query': "How do recent AAO decisions evaluate an applicant's Participation as a Judge service criteria?",
                'relevant_documents': ['February_03__2025_FEB032025_01B2203', 'February_13__2025_FEB132025_03B2203'],
                'expected_answer': "AAO evaluates participation as a judge by examining the nature of the judging role, the reputation of the organization, and whether it required expertise in the field."
            },
            {
                'query': "What characteristics of national or international awards persuade the AAO that they constitute sustained acclaim?",
                'relevant_documents': ['February_13__2025_FEB132025_03B2203', 'February_24__2025_FEB242025_01B2203'],
                'expected_answer': "AAO looks for awards that are nationally or internationally recognized, competitive in nature, based on excellence in the field, and indicative of sustained acclaim."
            }
        ]
        
        print("Running example evaluation...")
        results_example = evaluator.evaluate_test_set(test_queries_example)
        
        output_dir_example = "example_evaluation_output"
        report_example = evaluator.generate_evaluation_report(results_example, os.path.join(output_dir_example, "evaluation_report.json"))
        evaluator.visualize_results(results_example, output_dir_example)
        
        print("\nExample Evaluation Summary:")
        if results_example['aggregated_metrics'].get('overall'):
            print(f"Overall RAG Score: {results_example['aggregated_metrics']['overall'].get('rag_score', 0.0):.3f}")
        if results_example['aggregated_metrics'].get('retrieval'):
            print(f"Mean F1@5: {results_example['aggregated_metrics']['retrieval'].get('mean_f1_at_5', 0.0):.3f}")
        if results_example['aggregated_metrics'].get('answer'):
            print(f"Mean Answer Relevance: {results_example['aggregated_metrics']['answer'].get('mean_answer_relevance', 0.0):.3f}")
    else:
        print("Skipping example evaluation run due to ChromaDB/collection initialization issues.")