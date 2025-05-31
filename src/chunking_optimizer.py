# src/chunking_optimizer.py

import os
import json
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

@dataclass
class ChunkingConfig:
    """Configuration for text chunking experiments."""
    chunk_size: int
    chunk_overlap: int
    separators: List[str] = None
    
    def __post_init__(self):
        if self.separators is None:
            # Legal document-specific separators
            self.separators = [
                "\n\n\n",  # Multiple line breaks (section breaks)
                "\n\n",    # Paragraph breaks
                "\n",      # Line breaks
                ". ",      # Sentence endings
                "; ",      # Legal clause separators
                ", ",      # Comma separators
                " "        # Word boundaries
            ]


class ChunkingOptimizer:
    """Optimizes chunking strategy for legal documents."""
    
    def __init__(self, embedding_model_name: str = "BAAI/bge-small-en-v1.5"):
        self.embedding_model_name = embedding_model_name
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        
        # Test configurations
        self.test_configs = [
            ChunkingConfig(chunk_size=500, chunk_overlap=50),
            ChunkingConfig(chunk_size=500, chunk_overlap=100),
            ChunkingConfig(chunk_size=500, chunk_overlap=150),
            ChunkingConfig(chunk_size=750, chunk_overlap=75),
            ChunkingConfig(chunk_size=750, chunk_overlap=150),
            ChunkingConfig(chunk_size=750, chunk_overlap=225),
            ChunkingConfig(chunk_size=1000, chunk_overlap=100),
            ChunkingConfig(chunk_size=1000, chunk_overlap=200),
            ChunkingConfig(chunk_size=1000, chunk_overlap=300),
            ChunkingConfig(chunk_size=1250, chunk_overlap=125),
            ChunkingConfig(chunk_size=1250, chunk_overlap=250),
            ChunkingConfig(chunk_size=1250, chunk_overlap=375),
            ChunkingConfig(chunk_size=1500, chunk_overlap=150),
            ChunkingConfig(chunk_size=1500, chunk_overlap=300),
            ChunkingConfig(chunk_size=1500, chunk_overlap=450),
        ]
        
        # Metrics storage (not actively used in current methods, but kept for potential future use)
        self.metrics = defaultdict(dict) 
        
    def evaluate_chunking_strategies(self, 
                                   test_documents: List[Dict],
                                   test_queries: List[Dict[str, str]]) -> Dict:
        """Evaluate different chunking strategies on test documents and queries."""
        results = {}
        
        if not test_documents:
            print("Warning: No test documents provided to evaluate_chunking_strategies.")
            # return results # Or raise error

        if not test_queries:
            print("Warning: No test queries provided to evaluate_chunking_strategies.")
            # return results # Or raise error

        for config in tqdm(self.test_configs, desc="Testing chunking configurations"):
            config_key = f"size_{config.chunk_size}_overlap_{config.chunk_overlap}"
            print(f"\nEvaluating: {config_key}")
            
            client = chromadb.EphemeralClient() 
            collection = client.create_collection(
                name=f"test_{config_key}",
                embedding_function=self.embedding_function
            )
            
            chunk_stats = self._chunk_and_store_documents(
                documents=test_documents,
                collection=collection,
                config=config
            )
            
            retrieval_metrics = self._evaluate_retrieval(
                collection=collection,
                test_queries=test_queries,
                config=config # config is not used by _evaluate_retrieval but passed for consistency
            )
            
            num_chunks_in_collection = collection.count()
            coherence_sample_size = min(100, num_chunks_in_collection)

            coherence_score = self._calculate_semantic_coherence(
                collection=collection,
                sample_size=coherence_sample_size
            )
            
            results[config_key] = {
                'config': {
                    'chunk_size': config.chunk_size,
                    'chunk_overlap': config.chunk_overlap
                },
                'chunk_stats': chunk_stats,
                'retrieval_metrics': retrieval_metrics,
                'coherence_score': coherence_score,
                'efficiency_score': self._calculate_efficiency_score(chunk_stats)
            }
            
            client.delete_collection(name=f"test_{config_key}")
        
        return results
    
    def _chunk_and_store_documents(self, 
                                  documents: List[Dict],
                                  collection: chromadb.Collection,
                                  config: ChunkingConfig) -> Dict:
        """Chunk documents and store in collection, returning statistics."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
            add_start_index=True
        )
        
        total_chunks = 0
        chunk_lengths = []
        overlap_ratios = []
        
        for doc_idx, doc in enumerate(documents):
            text = doc.get('cleaned_text', '')
            doc_id = doc.get('document_id', f'unknown_doc_{doc_idx}') # Fallback doc_id

            if not text:
                continue
                
            doc_metadata_base = {
                'document_id': doc_id,
                'case_name': doc.get('case_name', 'Unknown'),
                'publication_date': doc.get('publication_date_on_website', 'Unknown')
            }
            # Langchain's create_documents expects a list of texts and a parallel list of metadatas
            chunks = text_splitter.create_documents(
                [text], 
                metadatas=[doc_metadata_base] 
            )
            
            batch_ids = []
            batch_documents = []
            batch_metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                batch_ids.append(chunk_id)
                batch_documents.append(chunk.page_content)
                
                metadata = chunk.metadata.copy() 
                metadata['chunk_index'] = i 
                metadata['chunk_length'] = len(chunk.page_content)
                batch_metadatas.append(metadata)
                
                chunk_lengths.append(len(chunk.page_content))
                
                if i > 0:
                    prev_chunk_content = chunks[i-1].page_content
                    curr_chunk_content = chunk.page_content
                    overlap = self._calculate_overlap(prev_chunk_content, curr_chunk_content)
                    overlap_ratios.append(overlap / len(curr_chunk_content) if len(curr_chunk_content) > 0 else 0)
            
            if batch_ids:
                collection.add(
                    ids=batch_ids,
                    documents=batch_documents,
                    metadatas=batch_metadatas
                )
                total_chunks += len(batch_ids)
        
        return {
            'total_chunks': total_chunks,
            'avg_chunk_length': float(np.mean(chunk_lengths)) if chunk_lengths else 0.0,
            'std_chunk_length': float(np.std(chunk_lengths)) if chunk_lengths else 0.0,
            'min_chunk_length': float(min(chunk_lengths)) if chunk_lengths else 0.0,
            'max_chunk_length': float(max(chunk_lengths)) if chunk_lengths else 0.0,
            'avg_overlap_ratio': float(np.mean(overlap_ratios)) if overlap_ratios else 0.0
        }
    
    def _calculate_overlap(self, text1: str, text2: str) -> int:
        """Calculate the overlapping characters between end of text1 and start of text2."""
        max_possible_overlap = min(len(text1), len(text2))
        for length in range(max_possible_overlap, 0, -1):
            if text1.endswith(text2[:length]):
                return length
        return 0
    
    def _evaluate_retrieval(self,
                          collection: chromadb.Collection,
                          test_queries: List[Dict[str, str]],
                          config: ChunkingConfig, # Not used, but kept for signature consistency
                          k_values: List[int] = [3, 5, 10]) -> Dict:
        """Evaluate retrieval performance for different k values."""
        metrics = {f'precision_at_{k}': [] for k in k_values}
        metrics.update({f'recall_at_{k}': [] for k in k_values})
        metrics['mrr'] = []
        metrics['retrieval_times'] = []
        
        for query_data in test_queries:
            query = query_data['query']
            relevant_docs_for_query = query_data.get('relevant_documents', [])
            
            if not relevant_docs_for_query:
                continue
            
            start_time = time.time()
            results = collection.query(
                query_texts=[query],
                n_results=max(k_values),
                include=['metadatas', 'distances']
            )
            retrieval_time = time.time() - start_time
            metrics['retrieval_times'].append(retrieval_time)
            
            retrieved_doc_ids_ordered = []
            if results['metadatas'] and results['metadatas'][0]:
                retrieved_doc_ids_ordered = [meta['document_id'] for meta in results['metadatas'][0] if 'document_id' in meta]
            
            unique_retrieved_doc_ids_ordered = list(dict.fromkeys(retrieved_doc_ids_ordered))

            for k in k_values:
                # Use unique doc IDs up to k *retrieved items* (not k unique docs)
                # This means if top k chunks are from same doc, it's one unique doc for P/R
                retrieved_at_k_unique_docs = list(dict.fromkeys(retrieved_doc_ids_ordered[:k]))
                
                relevant_retrieved_count = len(set(retrieved_at_k_unique_docs) & set(relevant_docs_for_query))
                
                precision_k = relevant_retrieved_count / len(retrieved_at_k_unique_docs) if len(retrieved_at_k_unique_docs) > 0 else 0.0
                metrics[f'precision_at_{k}'].append(precision_k)
                
                recall_k = relevant_retrieved_count / len(relevant_docs_for_query) if len(relevant_docs_for_query) > 0 else 0.0
                metrics[f'recall_at_{k}'].append(recall_k)
            
            reciprocal_rank = 0.0
            for i, doc_id in enumerate(unique_retrieved_doc_ids_ordered): # Iterate unique IDs in order of first appearance
                if doc_id in relevant_docs_for_query:
                    reciprocal_rank = 1.0 / (i + 1)
                    break
            metrics['mrr'].append(reciprocal_rank)
        
        avg_metrics = {}
        for metric_name, values_list in metrics.items():
            if values_list:
                avg_val = float(np.mean(values_list)) # Cast to float
                avg_metrics[f'avg_{metric_name}'] = avg_val
                if metric_name == 'retrieval_times':
                    avg_metrics[f'std_{metric_name}'] = float(np.std(values_list)) # Cast to float
            else: # Default to 0 if no data (e.g., all queries skipped)
                avg_metrics[f'avg_{metric_name}'] = 0.0
                if metric_name == 'retrieval_times':
                    avg_metrics[f'std_{metric_name}'] = 0.0
        
        return avg_metrics
    
    def _calculate_semantic_coherence(self,
                                    collection: chromadb.Collection,
                                    sample_size: int = 100) -> float:
        """Calculate semantic coherence of chunks."""
        if sample_size == 0: # No items to sample
            return 0.0
            
        try:
            all_data = collection.get(limit=sample_size, include=['embeddings', 'metadatas'])
        except Exception: # Broad exception if collection.get fails for any reason
            return 0.0

        retrieved_embeddings = all_data.get('embeddings')
        retrieved_metadatas = all_data.get('metadatas')

        if retrieved_embeddings is None or len(retrieved_embeddings) == 0 or \
           retrieved_metadatas is None or len(retrieved_metadatas) == 0:
            return 0.0
        
        if len(retrieved_embeddings) != len(retrieved_metadatas):
            return 0.0 # Data inconsistency

        coherence_scores = []
        doc_chunks = defaultdict(list)

        for i, metadata in enumerate(retrieved_metadatas):
            doc_id = metadata.get('document_id')
            chunk_index = metadata.get('chunk_index') 
            
            if doc_id is None or chunk_index is None:
                continue 
            doc_chunks[doc_id].append((chunk_index, retrieved_embeddings[i]))
        
        for doc_id, current_doc_embeddings_with_indices in doc_chunks.items():
            if len(current_doc_embeddings_with_indices) < 2:
                continue

            current_doc_embeddings_with_indices.sort(key=lambda x: x[0])
            
            for i in range(len(current_doc_embeddings_with_indices) - 1):
                embedding1_data = current_doc_embeddings_with_indices[i][1]
                embedding2_data = current_doc_embeddings_with_indices[i+1][1]

                embedding1 = np.array(embedding1_data, dtype=np.float32)
                embedding2 = np.array(embedding2_data, dtype=np.float32)
                
                norm_e1 = np.linalg.norm(embedding1)
                norm_e2 = np.linalg.norm(embedding2)

                if norm_e1 == 0 or norm_e2 == 0:
                    similarity = 0.0 
                else:
                    # Ensure embeddings are 1D for dot product
                    if embedding1.ndim > 1: embedding1 = embedding1.squeeze()
                    if embedding2.ndim > 1: embedding2 = embedding2.squeeze()
                    
                    # Check again after squeeze, if still not 1D, something is wrong with embedding shape
                    if embedding1.ndim > 1 or embedding2.ndim > 1 or embedding1.shape != embedding2.shape:
                        similarity = 0.0 # Skip if shapes are incompatible for dot product
                    else:
                        similarity = np.dot(embedding1, embedding2) / (norm_e1 * norm_e2)
                
                coherence_scores.append(similarity)
        
        return float(np.mean(coherence_scores)) if coherence_scores else 0.0
    
    def _calculate_efficiency_score(self, chunk_stats: Dict) -> float:
        """Calculate efficiency score based on chunk statistics."""
        avg_chunk_len = chunk_stats.get('avg_chunk_length', 0.0)
        std_chunk_len = chunk_stats.get('std_chunk_length', 0.0)
        avg_overlap_ratio = chunk_stats.get('avg_overlap_ratio', 0.0)

        if avg_chunk_len == 0:
            return 0.0

        ideal_length = 1000.0
        length_penalty = abs(avg_chunk_len - ideal_length) / ideal_length
        
        consistency_score = 1.0 / (1.0 + (std_chunk_len / avg_chunk_len if avg_chunk_len > 0 else 1.0))
        
        ideal_overlap_ratio = 0.15
        overlap_penalty = abs(avg_overlap_ratio - ideal_overlap_ratio) / ideal_overlap_ratio if ideal_overlap_ratio > 0 else abs(avg_overlap_ratio)
        
        # Weights can be tuned. Max score for each component is 1.
        # (1 - penalty) gives score where 0 penalty = 1 score.
        score_len = max(0, 1.0 - length_penalty)
        score_consistency = consistency_score # Already in a good range
        score_overlap = max(0, 1.0 - overlap_penalty)

        efficiency = (score_len * 0.4) + \
                     (score_consistency * 0.3) + \
                     (score_overlap * 0.3)
        
        return max(0.0, min(1.0, efficiency))
    
    def find_optimal_configuration(self, results: Dict) -> Tuple[Optional[str], Optional[Dict]]:
        """Find the optimal chunking configuration based on all metrics."""
        if not results:
            return None, None
            
        scores = {}
        
        for config_key, metrics_data in results.items():
            retrieval_metrics = metrics_data.get('retrieval_metrics', {})
            coherence_val = metrics_data.get('coherence_score', 0.0)
            efficiency_val = metrics_data.get('efficiency_score', 0.0)

            precision_5_val = retrieval_metrics.get('avg_precision_at_5', 0.0)
            recall_5_val = retrieval_metrics.get('avg_recall_at_5', 0.0)
            mrr_val = retrieval_metrics.get('avg_mrr', 0.0)
            avg_retrieval_time = retrieval_metrics.get('avg_retrieval_times', 1.0) # Default to a high time if missing

            # Weights for each component score
            w_retrieval = 0.6
            w_coherence = 0.2
            w_efficiency = 0.2
            
            # Retrieval sub-scores weights
            w_p5 = 0.4
            w_r5 = 0.3
            w_mrr = 0.3
            
            retrieval_score_component = (precision_5_val * w_p5) + \
                                        (recall_5_val * w_r5) + \
                                        (mrr_val * w_mrr)
            
            # Time penalty: higher penalty for slower times. Max penalty 0.1.
            # Penalize if time > 0.1s. Scale: 0.1s diff = 0.05 penalty.
            time_penalty = max(0, min(0.1, (avg_retrieval_time - 0.1) * 0.5)) 

            total_score = (retrieval_score_component * w_retrieval) + \
                          (coherence_val * w_coherence) + \
                          (efficiency_val * w_efficiency) - \
                          time_penalty
            
            scores[config_key] = total_score
        
        if not scores: # Should not happen if results is not empty
            return None, None

        best_config_key = max(scores, key=scores.get)
        return best_config_key, results[best_config_key]
    
    def visualize_results(self, results: Dict, output_dir: str = "chunking_analysis"):
        """Create visualizations of chunking experiment results."""
        if not results:
            print("No results to visualize.")
            return

        os.makedirs(output_dir, exist_ok=True)
        
        configs_labels = []
        precision_5_vals = []
        recall_5_vals = []
        mrr_scores_vals = []
        coherence_vals = []
        efficiency_vals = []
        
        for config_key, metrics_data in results.items():
            config = metrics_data['config']
            configs_labels.append(f"{config['chunk_size']}/{config['chunk_overlap']}")
            
            ret_metrics = metrics_data.get('retrieval_metrics', {})
            precision_5_vals.append(ret_metrics.get('avg_precision_at_5', 0.0))
            recall_5_vals.append(ret_metrics.get('avg_recall_at_5', 0.0))
            mrr_scores_vals.append(ret_metrics.get('avg_mrr', 0.0))
            coherence_vals.append(metrics_data.get('coherence_score', 0.0))
            efficiency_vals.append(metrics_data.get('efficiency_score', 0.0))

        if not configs_labels:
            print("No configuration data to visualize.")
            return

        fig, axes = plt.subplots(3, 2, figsize=(18, 18))
        fig.suptitle('Chunking Strategy Analysis', fontsize=16)
        
        plot_params = [
            (axes[0, 0], precision_5_vals, 'Average Precision@5', 'skyblue'),
            (axes[0, 1], recall_5_vals, 'Average Recall@5', 'lightgreen'),
            (axes[1, 0], mrr_scores_vals, 'Average MRR', 'lightcoral'),
            (axes[1, 1], coherence_vals, 'Semantic Coherence', 'coral'),
            (axes[2, 0], efficiency_vals, 'Efficiency Score', 'gold'),
        ]

        for ax, data, title, color in plot_params:
            ax.bar(configs_labels, data, color=color)
            ax.set_title(title)
            ax.set_xlabel('Chunk Size/Overlap')
            ax.set_ylabel(title.split()[-1]) # Use last word as Y-label (Precision, Recall, etc.)
            
            # Corrected way to set x-tick label rotation and alignment
            plt.setp(ax.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")
            # If you need to set other tick parameters (like tick size, color, etc., but not label-specific ones like rotation)
            # you can still use ax.tick_params() for those, e.g.:
            # ax.tick_params(axis='x', direction='out', length=6, width=2, colors='r')
            
            ax.grid(axis='y', linestyle='--', alpha=0.7)

        if len(axes.flatten()) > len(plot_params): # Hide unused subplots
            axes[2,1].set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
        plot_path = os.path.join(output_dir, 'chunking_analysis_summary.png')
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"Summary plot saved to {plot_path}")
        
        chunk_sizes_set = sorted(list(set(m['config']['chunk_size'] for m in results.values())))
        chunk_overlaps_set = sorted(list(set(m['config']['chunk_overlap'] for m in results.values())))
        
        if not chunk_sizes_set or not chunk_overlaps_set:
            print("Not enough data for heatmap.")
            return

        # Using MRR for heatmap
        performance_matrix = np.full((len(chunk_sizes_set), len(chunk_overlaps_set)), np.nan)
        
        for i, size in enumerate(chunk_sizes_set):
            for j, overlap in enumerate(chunk_overlaps_set):
                key = f"size_{size}_overlap_{overlap}"
                if key in results and results[key].get('retrieval_metrics'):
                    performance_matrix[i, j] = results[key]['retrieval_metrics'].get('avg_mrr', np.nan)
        
        if np.isnan(performance_matrix).all():
            print("Performance matrix for heatmap is all NaN. Skipping heatmap.")
            return

        plt.figure(figsize=(max(8, len(chunk_overlaps_set)*0.8), max(6, len(chunk_sizes_set)*0.6)))
        sns.heatmap(performance_matrix, 
                    xticklabels=chunk_overlaps_set, yticklabels=chunk_sizes_set,
                    annot=True, fmt='.3f', cmap='YlGnBu',
                    cbar_kws={'label': 'Average MRR Score'}, linewidths=.5, linecolor='gray')
        plt.title('Chunking Performance Heatmap (MRR)')
        plt.xlabel('Chunk Overlap')
        plt.ylabel('Chunk Size')
        plt.tight_layout()
        heatmap_path = os.path.join(output_dir, 'performance_heatmap_mrr.png')
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        print(f"Heatmap saved to {heatmap_path}")
    
    def generate_report(self, results: Dict, output_file: str = "chunking_report.json"):
        """Generate a comprehensive report of chunking experiments."""
        if not results:
            print("No results to generate report from.")
            return {} # Return empty dict if no results
            
        best_config_key, best_metrics_data = self.find_optimal_configuration(results)
        
        report = {
            'experiment_summary': {
                'total_configurations_tested': len(results),
                'embedding_model': self.embedding_model_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'all_results': results 
        }

        if best_config_key and best_metrics_data:
            report['optimal_configuration'] = {
                'config_key': best_config_key,
                'chunk_size': best_metrics_data['config']['chunk_size'],
                'chunk_overlap': best_metrics_data['config']['chunk_overlap'],
                'metrics': best_metrics_data
            }
            report['recommendations'] = self._generate_recommendations(results, best_config_key)
        else:
            report['optimal_configuration'] = None
            report['recommendations'] = ["Could not determine an optimal configuration from the provided results."]
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                # Custom encoder for numpy types
                class NpEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.integer):
                            return int(obj)
                        if isinstance(obj, np.floating):
                            return float(obj)
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()
                        return super(NpEncoder, self).default(obj)
                json.dump(report, f, indent=2, cls=NpEncoder)
            print(f"Report saved to {output_file}")
        except Exception as e:
            print(f"Error saving report to {output_file}: {e}")
        
        return report
    
    def _generate_recommendations(self, results: Dict, best_config_key: str) -> List[str]:
        """Generate recommendations based on experiment results."""
        recommendations = []
        
        if best_config_key not in results or not results[best_config_key]:
            return ["Best configuration data not found or is empty."]

        best_metrics = results[best_config_key]
        config = best_metrics.get('config', {})
        ret_metrics = best_metrics.get('retrieval_metrics', {})
        coherence_s = best_metrics.get('coherence_score', 0.0)

        chunk_s = config.get('chunk_size')
        chunk_o = config.get('chunk_overlap')

        if chunk_s is None or chunk_o is None:
            return ["Optimal configuration details (size/overlap) are missing."]

        recommendations.append(f"The determined optimal configuration is: Chunk Size {chunk_s}, Overlap {chunk_o}.")
        
        if chunk_s < 750:
            recommendations.append(
                "The optimal chunk size is relatively small. This can enhance precision for specific queries but might fragment broader contexts in legal texts. Monitor if recall for complex queries is affected."
            )
        elif chunk_s > 1250:
            recommendations.append(
                "The optimal chunk size is large. This helps maintain context but could lead to less focused chunks, potentially impacting precision or including irrelevant information. Verify if chunks remain topically coherent."
            )
        
        overlap_ratio = chunk_o / chunk_s if chunk_s > 0 else 0.0
        if overlap_ratio < 0.1: # e.g., less than 10%
            recommendations.append(
                f"Overlap ratio ({overlap_ratio:.2f}) is low. This is storage/computationally efficient but increases risk of splitting critical information across chunk boundaries. If context continuity issues arise, consider increasing overlap (e.g., to 0.1-0.2)."
            )
        elif overlap_ratio > 0.3: # e.g., more than 30%
            recommendations.append(
                f"Overlap ratio ({overlap_ratio:.2f}) is high. While ensuring context continuity, this leads to significant data redundancy and higher processing/storage costs. Evaluate if a slightly lower overlap (e.g., 0.15-0.25) could offer a better trade-off."
            )
        
        avg_p5 = ret_metrics.get('avg_precision_at_5', 0.0)
        if avg_p5 < 0.7:
            recommendations.append(
                f"Average Precision@5 ({avg_p5:.3f}) is below the target of 0.7. This suggests retrieved results may not always be highly relevant. Consider: (a) fine-tuning the embedding model on domain-specific data, (b) experimenting with different embedding models, or (c) refining the query formulation."
            )
        
        if coherence_s < 0.7:
            recommendations.append(
                f"Semantic coherence score ({coherence_s:.3f}) is moderate to low. This indicates that consecutive chunks might not always be semantically related as desired. Review: (a) the list of separators used for chunking to better align with logical breaks in legal documents, (b) the embedding model's ability to capture nuanced semantic relationships."
            )
        
        if not recommendations or len(recommendations) == 1: # Only the first general one
            recommendations.append("The chosen configuration appears to provide a reasonable balance across metrics. Continued monitoring with a broader set of documents and queries is advised.")

        return recommendations


# Example usage for standalone execution
if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # Path assumes 'data/processed' is one level up from 'src' where this script is
    processed_data_dir = os.path.join(current_script_dir, '..', 'data', 'processed') 
    
    test_documents_list = []
    
    if not os.path.isdir(processed_data_dir):
        print(f"Error: Processed data directory not found at '{processed_data_dir}'. Please ensure the path is correct.")
    else:
        doc_files_json = [f for f in os.listdir(processed_data_dir) if f.endswith('.json')]
        for filename in doc_files_json[:5]: # Load up to 5 documents
            try:
                with open(os.path.join(processed_data_dir, filename), 'r', encoding='utf-8') as f:
                    doc_content = json.load(f)
                    if 'document_id' in doc_content and 'cleaned_text' in doc_content: # Basic validation
                        test_documents_list.append(doc_content)
                    else:
                        print(f"Warning: Document {filename} missing 'document_id' or 'cleaned_text'. Skipping.")
            except Exception as e:
                print(f"Error loading document {filename}: {e}")

    if not test_documents_list:
        print("No valid test documents loaded. Exiting chunking optimizer example.")
        import sys
        sys.exit(1)

    # Dynamically create relevant_documents lists based on loaded document_ids
    loaded_doc_ids = [doc['document_id'] for doc in test_documents_list]
    
    test_queries_list = []
    if len(loaded_doc_ids) >= 1:
        test_queries_list.append({
            'query': "participation as a judge requirements",
            'relevant_documents': loaded_doc_ids[:min(2, len(loaded_doc_ids))] # First 1 or 2 docs
        })
    if len(loaded_doc_ids) >= 2: # Need at least 2 docs for this query's original logic
         # Use docs starting from the second one, if available
        relevant_for_q2 = []
        if len(loaded_doc_ids) > 1: relevant_for_q2.append(loaded_doc_ids[1])
        if len(loaded_doc_ids) > 2: relevant_for_q2.append(loaded_doc_ids[2])
        if relevant_for_q2:
            test_queries_list.append({
                'query': "extraordinary ability awards criteria",
                'relevant_documents': relevant_for_q2
            })
    
    test_queries_list = [q for q in test_queries_list if q['relevant_documents']] # Ensure queries have ground truth

    if not test_queries_list:
         print("Warning: No test queries could be formulated with relevant documents based on loaded data.")
    
    optimizer_instance = ChunkingOptimizer()
    
    print(f"\nStarting chunking optimization with {len(test_documents_list)} documents and {len(test_queries_list)} queries.")
    print("This may take several minutes depending on the number of configurations and document sizes...")
    
    optimization_results = optimizer_instance.evaluate_chunking_strategies(test_documents_list, test_queries_list)
    
    if optimization_results:
        # Define output paths relative to the script's location or a designated output directory
        output_report_file = "chunking_optimizer_report.json"
        output_visuals_dir = "chunking_optimizer_visuals"

        generated_report = optimizer_instance.generate_report(optimization_results, output_report_file)
        optimizer_instance.visualize_results(optimization_results, output_visuals_dir)
        
        best_conf_key, best_conf_metrics = optimizer_instance.find_optimal_configuration(optimization_results)
        
        if best_conf_key and best_conf_metrics:
            print(f"\n--- Optimal Configuration Summary ---")
            print(f"Key: {best_conf_key}")
            print(f"Chunk Size: {best_conf_metrics['config']['chunk_size']}")
            print(f"Chunk Overlap: {best_conf_metrics['config']['chunk_overlap']}")
            
            ret_m = best_conf_metrics.get('retrieval_metrics', {})
            print(f"  Avg Precision@5: {ret_m.get('avg_precision_at_5', 0.0):.3f}")
            print(f"  Avg Recall@5: {ret_m.get('avg_recall_at_5', 0.0):.3f}")
            print(f"  Avg MRR: {ret_m.get('avg_mrr', 0.0):.3f}")
            print(f"  Avg Retrieval Time: {ret_m.get('avg_retrieval_times', 0.0):.4f}s")

            print(f"Coherence Score: {best_conf_metrics.get('coherence_score', 0.0):.3f}")
            print(f"Efficiency Score: {best_conf_metrics.get('efficiency_score', 0.0):.3f}")
            
            if 'recommendations' in generated_report and generated_report['recommendations']:
                print("\nRecommendations:")
                for rec_item in generated_report['recommendations']:
                    print(f"- {rec_item}")
        else:
            print("\nCould not determine an optimal configuration from the optimization results.")
    else:
        print("\nChunking optimization did not produce any results. Please check input data and configurations.")