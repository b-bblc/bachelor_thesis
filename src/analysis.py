"""
Analysis utilities for dependency parsing results.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import conllu

from .config import RESULTS_GERMAN_DIR, RESULTS_RUSSIAN_DIR, RESULTS_DIR, VISUALIZATION_LIMIT, get_logger

logger = get_logger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class DependencyAnalyzer:
    """Class for analyzing dependency parsing results."""
    
    def __init__(self, results_dir: str = None):
        """
        Initialize the analyzer.
        
        Args:
            results_dir: Directory to save analysis results
        """
        self.results_dir = Path(results_dir) if results_dir else RESULTS_DIR
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        self.data = []
        self.features_df = None
        self.language_stats = {}
    
    def load_conllu_files(self, input_dir: str, language: str = None) -> List[Dict]:
        """
        Load and parse CoNLL-U files.
        
        Args:
            input_dir: Directory containing .conllu files
            language: Language label for the data
            
        Returns:
            List of sentence dictionaries
        """
        input_path = Path(input_dir)
        conllu_files = list(input_path.glob("*.conllu"))
        
        sentences = []
        
        for file_path in conllu_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    parsed_sentences = conllu.parse(content)
                    
                    for sent in parsed_sentences:
                        sentence_data = {
                            'file': file_path.name,
                            'sent_id': sent.metadata.get('sent_id', ''),
                            'text': sent.metadata.get('text', ''),
                            'tokens': sent,
                            'language': language
                        }
                        sentences.append(sentence_data)
                        
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
        
        logger.info(f"Loaded {len(sentences)} sentences from {len(conllu_files)} files")
        return sentences
    
    def extract_features(self, sentences: List[Dict]) -> pd.DataFrame:
        """
        Extract linguistic features from parsed sentences.
        
        Args:
            sentences: List of sentence dictionaries
            
        Returns:
            DataFrame with extracted features
        """
        features = []
        
        for sent_data in sentences:
            tokens = sent_data['tokens']
            
            # Basic features
            feature_dict = {
                'file': sent_data['file'],
                'sent_id': sent_data['sent_id'],
                'text': sent_data['text'],
                'language': sent_data['language'],
                'length': len(tokens),
                'avg_token_length': np.mean([len(token['form']) for token in tokens])
            }
            
            # POS tag features
            pos_counts = Counter(token['upos'] for token in tokens)
            total_tokens = len(tokens)
            
            for pos in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'ADP', 'CONJ', 'DET']:
                feature_dict[f'pos_{pos.lower()}_ratio'] = pos_counts.get(pos, 0) / total_tokens
            
            # Dependency relation features
            dep_counts = Counter(token['deprel'] for token in tokens)
            
            for dep in ['nsubj', 'obj', 'root', 'det', 'nmod', 'advmod', 'amod', 'cc', 'conj']:
                feature_dict[f'dep_{dep}_ratio'] = dep_counts.get(dep, 0) / total_tokens
            
            # Syntactic complexity features
            feature_dict['tree_depth'] = self._calculate_tree_depth(tokens)
            feature_dict['avg_dependency_distance'] = self._calculate_avg_dependency_distance(tokens)
            feature_dict['has_coordination'] = any(token['deprel'] == 'conj' for token in tokens)
            feature_dict['has_subordination'] = any(token['deprel'] in ['ccomp', 'xcomp', 'advcl'] for token in tokens)
            feature_dict['punct_ratio'] = pos_counts.get('PUNCT', 0) / total_tokens
            
            # Morphological features (if available)
            feature_dict['finite_verbs'] = sum(1 for token in tokens 
                                            if token['upos'] == 'VERB' and 
                                            'VerbForm=Fin' in str(token.get('feats', '')))
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def _calculate_tree_depth(self, tokens: List) -> int:
        """Calculate maximum depth of dependency tree."""
        def get_depth(token_id, tokens_dict, current_depth=0):
            children = [t for t in tokens_dict.values() if t.get('head') == token_id]
            if not children:
                return current_depth
            return max(get_depth(child['id'], tokens_dict, current_depth + 1) for child in children)
        
        tokens_dict = {token['id']: token for token in tokens}
        roots = [token for token in tokens if token['head'] == 0]
        
        if not roots:
            return 0
        
        return max(get_depth(root['id'], tokens_dict) for root in roots)
    
    def _calculate_avg_dependency_distance(self, tokens: List) -> float:
        """Calculate average distance between dependent and head."""
        distances = []
        for token in tokens:
            if token['head'] != 0:  # Not root
                distance = abs(token['id'] - token['head'])
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def generate_descriptive_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate descriptive statistics for the dataset."""
        stats = {
            'total_sentences': len(df),
            'avg_sentence_length': df['length'].mean(),
            'sentence_length_std': df['length'].std(),
            'languages': df['language'].value_counts().to_dict() if 'language' in df.columns else {},
            'pos_distribution': {},
            'dep_distribution': {},
            'complexity_stats': {}
        }
        
        # POS distribution
        pos_cols = [col for col in df.columns if col.startswith('pos_') and col.endswith('_ratio')]
        if pos_cols:
            stats['pos_distribution'] = df[pos_cols].mean().to_dict()
        
        # Dependency distribution
        dep_cols = [col for col in df.columns if col.startswith('dep_') and col.endswith('_ratio')]
        if dep_cols:
            stats['dep_distribution'] = df[dep_cols].mean().to_dict()
        
        # Complexity statistics
        complexity_features = ['tree_depth', 'avg_dependency_distance', 'finite_verbs']
        for feature in complexity_features:
            if feature in df.columns:
                stats['complexity_stats'][feature] = {
                    'mean': df[feature].mean(),
                    'std': df[feature].std(),
                    'min': df[feature].min(),
                    'max': df[feature].max()
                }
        
        return stats
    
    def visualize_feature_distributions(self, df: pd.DataFrame, save_plots: bool = True):
        """Create visualizations for feature distributions."""
        
        # 1. Sentence length distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 3, 1)
        plt.hist(df['length'], bins=30, alpha=0.7, edgecolor='black')
        plt.title('EDU Length Distribution')
        plt.xlabel('Number of Tokens')
        plt.ylabel('Frequency')
        
        # 2. POS distribution
        plt.subplot(2, 3, 2)
        pos_cols = [col for col in df.columns if col.startswith('pos_') and col.endswith('_ratio')]
        if pos_cols:
            pos_means = df[pos_cols].mean()
            pos_means.plot(kind='bar')
            plt.title('Average POS Distribution')
            plt.ylabel('Ratio')
            plt.xticks(rotation=45)
        
        # 3. Dependency distribution
        plt.subplot(2, 3, 3)
        dep_cols = [col for col in df.columns if col.startswith('dep_') and col.endswith('_ratio')]
        if dep_cols:
            dep_means = df[dep_cols].mean()
            dep_means.plot(kind='bar')
            plt.title('Average Dependency Distribution')
            plt.ylabel('Ratio')
            plt.xticks(rotation=45)
        
        # 4. Tree depth distribution
        plt.subplot(2, 3, 4)
        if 'tree_depth' in df.columns:
            plt.hist(df['tree_depth'], bins=15, alpha=0.7, edgecolor='black')
            plt.title('Tree Depth Distribution')
            plt.xlabel('Maximum Depth')
            plt.ylabel('Frequency')
        
        # 5. Language comparison (if multiple languages)
        plt.subplot(2, 3, 5)
        if 'language' in df.columns and df['language'].nunique() > 1:
            lang_lengths = df.groupby('language')['length'].mean()
            lang_lengths.plot(kind='bar')
            plt.title('Average EDU Length by Language')
            plt.ylabel('Average Tokens')
            plt.xticks(rotation=45)
        
        # 6. Complexity scatter plot
        plt.subplot(2, 3, 6)
        if 'tree_depth' in df.columns and 'avg_dependency_distance' in df.columns:
            plt.scatter(df['tree_depth'], df['avg_dependency_distance'], alpha=0.6)
            plt.title('Tree Depth vs Dependency Distance')
            plt.xlabel('Tree Depth')
            plt.ylabel('Avg Dependency Distance')
        
        plt.tight_layout()
        
        if save_plots:
            output_path = self.results_dir / 'feature_distributions.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature distributions plot to {output_path}")
        
        plt.show()
    
    def perform_clustering_analysis(self, df: pd.DataFrame, n_clusters: int = 5) -> Dict:
        """
        Perform clustering analysis on EDU features.
        
        Args:
            df: DataFrame with features
            n_clusters: Number of clusters for K-means
            
        Returns:
            Dictionary with clustering results
        """
        # Select numerical features for clustering
        feature_cols = [col for col in df.columns if 
                       col.startswith(('pos_', 'dep_')) and col.endswith('_ratio') or
                       col in ['length', 'tree_depth', 'avg_dependency_distance', 'finite_verbs']]
        
        if len(feature_cols) < 2:
            logger.warning("Not enough numerical features for clustering")
            return {}
        
        X = df[feature_cols].fillna(0)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X, clusters)
        
        # Add cluster labels to dataframe
        df_clustered = df.copy()
        df_clustered['cluster'] = clusters
        
        # Analyze clusters
        cluster_stats = {}
        for i in range(n_clusters):
            cluster_data = df_clustered[df_clustered['cluster'] == i]
            cluster_stats[i] = {
                'size': len(cluster_data),
                'avg_length': cluster_data['length'].mean(),
                'avg_tree_depth': cluster_data['tree_depth'].mean() if 'tree_depth' in cluster_data.columns else 0,
                'dominant_features': self._get_dominant_features(cluster_data, feature_cols)
            }
        
        # Visualize clusters using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        plt.title(f'EDU Clusters (K-means, k={n_clusters})\nSilhouette Score: {silhouette_avg:.3f}')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.colorbar(scatter)
        
        # Save clustering plot
        output_path = self.results_dir / 'clustering_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved clustering plot to {output_path}")
        plt.show()
        
        return {
            'n_clusters': n_clusters,
            'silhouette_score': silhouette_avg,
            'cluster_stats': cluster_stats,
            'feature_importance': dict(zip(feature_cols, pca.explained_variance_ratio_)),
            'clustered_data': df_clustered
        }
    
    def _get_dominant_features(self, cluster_data: pd.DataFrame, feature_cols: List[str], top_k: int = 3) -> List[str]:
        """Get the most dominant features for a cluster."""
        feature_means = cluster_data[feature_cols].mean()
        return feature_means.nlargest(top_k).index.tolist()
    
    def compare_languages(self, df: pd.DataFrame) -> Dict:
        """
        Compare linguistic features across languages.
        
        Args:
            df: DataFrame with language column
            
        Returns:
            Dictionary with comparison results
        """
        if 'language' not in df.columns or df['language'].nunique() < 2:
            logger.warning("Cannot perform language comparison: need multiple languages")
            return {}
        
        comparison = {}
        languages = df['language'].unique()
        
        # Compare basic statistics
        for lang in languages:
            lang_data = df[df['language'] == lang]
            comparison[lang] = {
                'total_sentences': len(lang_data),
                'avg_length': lang_data['length'].mean(),
                'avg_tree_depth': lang_data['tree_depth'].mean() if 'tree_depth' in lang_data.columns else 0,
                'coordination_ratio': lang_data['has_coordination'].mean() if 'has_coordination' in lang_data.columns else 0,
                'subordination_ratio': lang_data['has_subordination'].mean() if 'has_subordination' in lang_data.columns else 0
            }
        
        # Visualize language comparison
        metrics = ['avg_length', 'avg_tree_depth', 'coordination_ratio', 'subordination_ratio']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [comparison[lang][metric] for lang in languages]
            axes[i].bar(languages, values)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Value')
        
        plt.tight_layout()
        
        # Save comparison plot
        output_path = self.results_dir / 'language_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved language comparison plot to {output_path}")
        plt.show()
        
        return comparison
    
    def generate_report(self, df: pd.DataFrame, output_file: str = None) -> str:
        """
        Generate a comprehensive analysis report.
        
        Args:
            df: DataFrame with analysis results
            output_file: Optional output file path
            
        Returns:
            Report text
        """
        report_lines = [
            "# EDU Dependency Analysis Report",
            f"Generated on: {pd.Timestamp.now()}",
            "",
            "## Dataset Overview",
            f"- Total EDUs: {len(df)}",
            f"- Average length: {df['length'].mean():.2f} tokens",
            f"- Length range: {df['length'].min()}-{df['length'].max()} tokens",
            ""
        ]
        
        # Language statistics
        if 'language' in df.columns:
            report_lines.extend([
                "## Language Distribution",
                *[f"- {lang}: {count} EDUs" for lang, count in df['language'].value_counts().items()],
                ""
            ])
        
        # Generate and add other statistics
        stats = self.generate_descriptive_statistics(df)
        
        report_lines.extend([
            "## Complexity Statistics",
            f"- Average tree depth: {stats['complexity_stats'].get('tree_depth', {}).get('mean', 'N/A')}",
            f"- Average dependency distance: {stats['complexity_stats'].get('avg_dependency_distance', {}).get('mean', 'N/A')}",
            ""
        ])
        
        report_text = "\n".join(report_lines)
        
        if output_file:
            output_path = self.results_dir / output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Saved analysis report to {output_path}")
        
        return report_text


def main():
    """Main function for standalone execution."""
    analyzer = DependencyAnalyzer()
    
    # Load German data
    german_sentences = analyzer.load_conllu_files("results/results_german/parsed_dependencies", "German")
    
    # Load Russian data
    russian_sentences = analyzer.load_conllu_files("results/results_russian/parsed_dependencies", "Russian")
    
    # Combine data
    all_sentences = german_sentences + russian_sentences
    
    # Extract features
    features_df = analyzer.extract_features(all_sentences)
    
    # Generate visualizations
    analyzer.visualize_feature_distributions(features_df)
    
    # Perform clustering
    clustering_results = analyzer.perform_clustering_analysis(features_df)
    
    # Compare languages
    language_comparison = analyzer.compare_languages(features_df)
    
    # Generate report
    report = analyzer.generate_report(features_df, "analysis_report.md")
    print(report)


if __name__ == "__main__":
    main()
