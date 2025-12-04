"""
Unit tests for the analysis module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import src package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import DependencyAnalyzer


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def analyzer(tmp_path):
    """Create DependencyAnalyzer instance with temporary results directory."""
    return DependencyAnalyzer(results_dir=str(tmp_path / "results"))


@pytest.fixture
def sample_conllu_content():
    """Sample CoNLL-U content for creating test files."""
    return """# sent_id = 1
# text = The cat sat on the mat.
1	The	the	DET	_	_	2	det	_	_
2	cat	cat	NOUN	_	_	3	nsubj	_	_
3	sat	sit	VERB	_	VerbForm=Fin	0	root	_	_
4	on	on	ADP	_	_	6	case	_	_
5	the	the	DET	_	_	6	det	_	_
6	mat	mat	NOUN	_	_	3	nmod	_	_
7	.	.	PUNCT	_	_	3	punct	_	_

# sent_id = 2
# text = Dogs run and cats sleep.
1	Dogs	dog	NOUN	_	_	2	nsubj	_	_
2	run	run	VERB	_	VerbForm=Fin	0	root	_	_
3	and	and	CCONJ	_	_	5	cc	_	_
4	cats	cat	NOUN	_	_	5	nsubj	_	_
5	sleep	sleep	VERB	_	VerbForm=Fin	2	conj	_	_
6	.	.	PUNCT	_	_	2	punct	_	_

"""


@pytest.fixture
def sample_conllu_file(tmp_path, sample_conllu_content):
    """Create a temporary CoNLL-U file for testing."""
    file_path = tmp_path / "test.conllu"
    file_path.write_text(sample_conllu_content, encoding='utf-8')
    return file_path


@pytest.fixture
def sample_sentences():
    """Sample sentence data in the format returned by load_conllu_files."""
    # Mimicking conllu.TokenList structure
    from collections import OrderedDict
    
    class MockToken(OrderedDict):
        """Mock token that behaves like conllu token."""
        def __getitem__(self, key):
            return super().__getitem__(key)
    
    class MockTokenList(list):
        """Mock token list that behaves like conllu.TokenList."""
        def __init__(self, tokens, metadata=None):
            super().__init__(tokens)
            self.metadata = metadata or {}
    
    # Sentence 1: "The cat sat."
    tokens1 = [
        MockToken([('id', 1), ('form', 'The'), ('lemma', 'the'), ('upos', 'DET'),
                   ('xpos', '_'), ('feats', None), ('head', 2), ('deprel', 'det'),
                   ('deps', '_'), ('misc', '_')]),
        MockToken([('id', 2), ('form', 'cat'), ('lemma', 'cat'), ('upos', 'NOUN'),
                   ('xpos', '_'), ('feats', None), ('head', 3), ('deprel', 'nsubj'),
                   ('deps', '_'), ('misc', '_')]),
        MockToken([('id', 3), ('form', 'sat'), ('lemma', 'sit'), ('upos', 'VERB'),
                   ('xpos', '_'), ('feats', 'VerbForm=Fin'), ('head', 0), ('deprel', 'root'),
                   ('deps', '_'), ('misc', '_')]),
        MockToken([('id', 4), ('form', '.'), ('lemma', '.'), ('upos', 'PUNCT'),
                   ('xpos', '_'), ('feats', None), ('head', 3), ('deprel', 'punct'),
                   ('deps', '_'), ('misc', '_')]),
    ]
    
    # Sentence 2: "Dogs run and cats sleep."
    tokens2 = [
        MockToken([('id', 1), ('form', 'Dogs'), ('lemma', 'dog'), ('upos', 'NOUN'),
                   ('xpos', '_'), ('feats', None), ('head', 2), ('deprel', 'nsubj'),
                   ('deps', '_'), ('misc', '_')]),
        MockToken([('id', 2), ('form', 'run'), ('lemma', 'run'), ('upos', 'VERB'),
                   ('xpos', '_'), ('feats', 'VerbForm=Fin'), ('head', 0), ('deprel', 'root'),
                   ('deps', '_'), ('misc', '_')]),
        MockToken([('id', 3), ('form', 'and'), ('lemma', 'and'), ('upos', 'CCONJ'),
                   ('xpos', '_'), ('feats', None), ('head', 5), ('deprel', 'cc'),
                   ('deps', '_'), ('misc', '_')]),
        MockToken([('id', 4), ('form', 'cats'), ('lemma', 'cat'), ('upos', 'NOUN'),
                   ('xpos', '_'), ('feats', None), ('head', 5), ('deprel', 'nsubj'),
                   ('deps', '_'), ('misc', '_')]),
        MockToken([('id', 5), ('form', 'sleep'), ('lemma', 'sleep'), ('upos', 'VERB'),
                   ('xpos', '_'), ('feats', 'VerbForm=Fin'), ('head', 2), ('deprel', 'conj'),
                   ('deps', '_'), ('misc', '_')]),
        MockToken([('id', 6), ('form', '.'), ('lemma', '.'), ('upos', 'PUNCT'),
                   ('xpos', '_'), ('feats', None), ('head', 2), ('deprel', 'punct'),
                   ('deps', '_'), ('misc', '_')]),
    ]
    
    sent1 = MockTokenList(tokens1, {'sent_id': '1', 'text': 'The cat sat.'})
    sent2 = MockTokenList(tokens2, {'sent_id': '2', 'text': 'Dogs run and cats sleep.'})
    
    return [
        {
            'file': 'test.conllu',
            'sent_id': '1',
            'text': 'The cat sat.',
            'tokens': sent1,
            'language': 'English'
        },
        {
            'file': 'test.conllu',
            'sent_id': '2',
            'text': 'Dogs run and cats sleep.',
            'tokens': sent2,
            'language': 'English'
        }
    ]


@pytest.fixture
def sample_features_df():
    """Sample features DataFrame for testing clustering."""
    np.random.seed(42)
    n_samples = 50
    
    data = {
        'file': [f'file{i}.conllu' for i in range(n_samples)],
        'sent_id': [str(i) for i in range(n_samples)],
        'text': [f'Sentence {i}.' for i in range(n_samples)],
        'language': ['English'] * 25 + ['German'] * 25,
        'length': np.random.randint(3, 20, n_samples),
        'avg_token_length': np.random.uniform(3, 8, n_samples),
        'pos_noun_ratio': np.random.uniform(0.1, 0.4, n_samples),
        'pos_verb_ratio': np.random.uniform(0.1, 0.3, n_samples),
        'pos_adj_ratio': np.random.uniform(0, 0.15, n_samples),
        'pos_adv_ratio': np.random.uniform(0, 0.1, n_samples),
        'pos_pron_ratio': np.random.uniform(0, 0.1, n_samples),
        'pos_adp_ratio': np.random.uniform(0.05, 0.15, n_samples),
        'pos_conj_ratio': np.random.uniform(0, 0.1, n_samples),
        'pos_det_ratio': np.random.uniform(0.05, 0.2, n_samples),
        'dep_nsubj_ratio': np.random.uniform(0.05, 0.2, n_samples),
        'dep_obj_ratio': np.random.uniform(0, 0.15, n_samples),
        'dep_root_ratio': np.random.uniform(0.05, 0.15, n_samples),
        'dep_det_ratio': np.random.uniform(0.05, 0.2, n_samples),
        'dep_nmod_ratio': np.random.uniform(0, 0.15, n_samples),
        'dep_advmod_ratio': np.random.uniform(0, 0.1, n_samples),
        'dep_amod_ratio': np.random.uniform(0, 0.1, n_samples),
        'dep_cc_ratio': np.random.uniform(0, 0.1, n_samples),
        'dep_conj_ratio': np.random.uniform(0, 0.1, n_samples),
        'tree_depth': np.random.randint(1, 6, n_samples),
        'avg_dependency_distance': np.random.uniform(1, 4, n_samples),
        'has_coordination': np.random.choice([True, False], n_samples),
        'has_subordination': np.random.choice([True, False], n_samples),
        'punct_ratio': np.random.uniform(0.05, 0.2, n_samples),
        'finite_verbs': np.random.randint(0, 3, n_samples),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def multilingual_features_df(sample_features_df):
    """DataFrame with data from multiple languages."""
    return sample_features_df.copy()


# =============================================================================
# Tests for extract_features
# =============================================================================

class TestExtractFeatures:
    """Tests for the extract_features method."""
    
    def test_returns_dataframe(self, analyzer, sample_sentences):
        """Should return a pandas DataFrame."""
        result = analyzer.extract_features(sample_sentences)
        
        assert isinstance(result, pd.DataFrame)
    
    def test_correct_number_of_rows(self, analyzer, sample_sentences):
        """Should have one row per sentence."""
        result = analyzer.extract_features(sample_sentences)
        
        assert len(result) == len(sample_sentences)
    
    def test_contains_basic_columns(self, analyzer, sample_sentences):
        """Should contain all basic feature columns."""
        result = analyzer.extract_features(sample_sentences)
        
        expected_columns = ['file', 'sent_id', 'text', 'language', 'length', 'avg_token_length']
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"
    
    def test_contains_pos_ratio_columns(self, analyzer, sample_sentences):
        """Should contain POS tag ratio columns."""
        result = analyzer.extract_features(sample_sentences)
        
        pos_columns = [col for col in result.columns if col.startswith('pos_') and col.endswith('_ratio')]
        assert len(pos_columns) > 0
    
    def test_contains_dep_ratio_columns(self, analyzer, sample_sentences):
        """Should contain dependency relation ratio columns."""
        result = analyzer.extract_features(sample_sentences)
        
        dep_columns = [col for col in result.columns if col.startswith('dep_') and col.endswith('_ratio')]
        assert len(dep_columns) > 0
    
    def test_contains_complexity_columns(self, analyzer, sample_sentences):
        """Should contain syntactic complexity columns."""
        result = analyzer.extract_features(sample_sentences)
        
        complexity_columns = ['tree_depth', 'avg_dependency_distance', 'has_coordination', 'has_subordination']
        for col in complexity_columns:
            assert col in result.columns, f"Missing complexity column: {col}"
    
    def test_length_is_correct(self, analyzer, sample_sentences):
        """Length should match number of tokens."""
        result = analyzer.extract_features(sample_sentences)
        
        # First sentence "The cat sat." has 4 tokens
        assert result.iloc[0]['length'] == 4
        # Second sentence "Dogs run and cats sleep." has 6 tokens
        assert result.iloc[1]['length'] == 6
    
    def test_has_coordination_detected(self, analyzer, sample_sentences):
        """Should detect coordination (conj relation) in second sentence."""
        result = analyzer.extract_features(sample_sentences)
        
        # First sentence has no coordination
        assert result.iloc[0]['has_coordination'] == False
        # Second sentence has coordination (and cats sleep)
        assert result.iloc[1]['has_coordination'] == True
    
    def test_ratios_are_valid(self, analyzer, sample_sentences):
        """All ratio columns should be between 0 and 1."""
        result = analyzer.extract_features(sample_sentences)
        
        ratio_columns = [col for col in result.columns if col.endswith('_ratio')]
        for col in ratio_columns:
            assert all(result[col] >= 0), f"{col} has values < 0"
            assert all(result[col] <= 1), f"{col} has values > 1"
    
    def test_handles_empty_list(self, analyzer):
        """Should return empty DataFrame for empty input."""
        result = analyzer.extract_features([])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# =============================================================================
# Tests for perform_clustering_analysis
# =============================================================================

class TestPerformClusteringAnalysis:
    """Tests for the perform_clustering_analysis method."""
    
    def test_returns_dict(self, analyzer, sample_features_df):
        """Should return a dictionary with results."""
        result = analyzer.perform_clustering_analysis(sample_features_df, n_clusters=3)
        
        assert isinstance(result, dict)
    
    def test_returns_correct_number_of_clusters(self, analyzer, sample_features_df):
        """Should return the requested number of clusters."""
        n_clusters = 4
        result = analyzer.perform_clustering_analysis(sample_features_df, n_clusters=n_clusters)
        
        assert result['n_clusters'] == n_clusters
        assert len(result['cluster_stats']) == n_clusters
    
    def test_returns_silhouette_score(self, analyzer, sample_features_df):
        """Should return silhouette score between -1 and 1."""
        result = analyzer.perform_clustering_analysis(sample_features_df, n_clusters=3)
        
        assert 'silhouette_score' in result
        assert -1 <= result['silhouette_score'] <= 1
    
    def test_returns_clustered_data(self, analyzer, sample_features_df):
        """Should return DataFrame with cluster labels."""
        result = analyzer.perform_clustering_analysis(sample_features_df, n_clusters=3)
        
        assert 'clustered_data' in result
        clustered_df = result['clustered_data']
        
        assert isinstance(clustered_df, pd.DataFrame)
        assert 'cluster' in clustered_df.columns
    
    def test_cluster_labels_are_valid(self, analyzer, sample_features_df):
        """Cluster labels should be in range [0, n_clusters-1]."""
        n_clusters = 4
        result = analyzer.perform_clustering_analysis(sample_features_df, n_clusters=n_clusters)
        
        clustered_df = result['clustered_data']
        unique_clusters = clustered_df['cluster'].unique()
        
        assert all(0 <= c < n_clusters for c in unique_clusters)
    
    def test_cluster_stats_structure(self, analyzer, sample_features_df):
        """Cluster stats should contain expected keys."""
        result = analyzer.perform_clustering_analysis(sample_features_df, n_clusters=3)
        
        for cluster_id, stats in result['cluster_stats'].items():
            assert 'size' in stats
            assert 'avg_length' in stats
            assert 'dominant_features' in stats
            assert isinstance(stats['dominant_features'], list)
    
    def test_all_samples_assigned(self, analyzer, sample_features_df):
        """All samples should be assigned to a cluster."""
        result = analyzer.perform_clustering_analysis(sample_features_df, n_clusters=3)
        
        clustered_df = result['clustered_data']
        
        # Sum of cluster sizes should equal total samples
        total_assigned = sum(stats['size'] for stats in result['cluster_stats'].values())
        assert total_assigned == len(sample_features_df)
    
    def test_handles_different_cluster_counts(self, analyzer, sample_features_df):
        """Should work with different numbers of clusters."""
        for n_clusters in [2, 5, 7]:
            result = analyzer.perform_clustering_analysis(sample_features_df, n_clusters=n_clusters)
            
            assert result['n_clusters'] == n_clusters
            assert len(result['cluster_stats']) == n_clusters
    
    def test_returns_empty_dict_for_insufficient_features(self, analyzer):
        """Should return empty dict if not enough features for clustering."""
        df = pd.DataFrame({
            'text': ['a', 'b', 'c'],
            'single_feature': [1, 2, 3]
        })
        
        result = analyzer.perform_clustering_analysis(df, n_clusters=2)
        
        assert result == {}


# =============================================================================
# Tests for generate_descriptive_statistics
# =============================================================================

class TestGenerateDescriptiveStatistics:
    """Tests for the generate_descriptive_statistics method."""
    
    def test_returns_dict(self, analyzer, sample_features_df):
        """Should return a dictionary."""
        result = analyzer.generate_descriptive_statistics(sample_features_df)
        
        assert isinstance(result, dict)
    
    def test_contains_basic_stats(self, analyzer, sample_features_df):
        """Should contain basic statistical measures."""
        result = analyzer.generate_descriptive_statistics(sample_features_df)
        
        assert 'total_sentences' in result
        assert 'avg_sentence_length' in result
        assert 'sentence_length_std' in result
    
    def test_total_sentences_correct(self, analyzer, sample_features_df):
        """Total sentences should match DataFrame length."""
        result = analyzer.generate_descriptive_statistics(sample_features_df)
        
        assert result['total_sentences'] == len(sample_features_df)
    
    def test_contains_language_distribution(self, analyzer, multilingual_features_df):
        """Should contain language distribution."""
        result = analyzer.generate_descriptive_statistics(multilingual_features_df)
        
        assert 'languages' in result
        assert isinstance(result['languages'], dict)
        assert 'English' in result['languages']
        assert 'German' in result['languages']
    
    def test_contains_pos_distribution(self, analyzer, sample_features_df):
        """Should contain POS distribution."""
        result = analyzer.generate_descriptive_statistics(sample_features_df)
        
        assert 'pos_distribution' in result
        assert isinstance(result['pos_distribution'], dict)
    
    def test_contains_dep_distribution(self, analyzer, sample_features_df):
        """Should contain dependency distribution."""
        result = analyzer.generate_descriptive_statistics(sample_features_df)
        
        assert 'dep_distribution' in result
        assert isinstance(result['dep_distribution'], dict)
    
    def test_contains_complexity_stats(self, analyzer, sample_features_df):
        """Should contain complexity statistics."""
        result = analyzer.generate_descriptive_statistics(sample_features_df)
        
        assert 'complexity_stats' in result
        complexity = result['complexity_stats']
        
        # Should have stats for tree_depth
        if 'tree_depth' in complexity:
            assert 'mean' in complexity['tree_depth']
            assert 'std' in complexity['tree_depth']
            assert 'min' in complexity['tree_depth']
            assert 'max' in complexity['tree_depth']


# =============================================================================
# Tests for compare_languages
# =============================================================================

class TestCompareLanguages:
    """Tests for the compare_languages method."""
    
    def test_returns_dict(self, analyzer, multilingual_features_df):
        """Should return a dictionary."""
        result = analyzer.compare_languages(multilingual_features_df)
        
        assert isinstance(result, dict)
    
    def test_returns_empty_for_single_language(self, analyzer, sample_features_df):
        """Should return empty dict if only one language."""
        single_lang_df = sample_features_df[sample_features_df['language'] == 'English'].copy()
        
        result = analyzer.compare_languages(single_lang_df)
        
        assert result == {}
    
    def test_contains_all_languages(self, analyzer, multilingual_features_df):
        """Should have entry for each language."""
        result = analyzer.compare_languages(multilingual_features_df)
        
        assert 'English' in result
        assert 'German' in result
    
    def test_contains_expected_metrics(self, analyzer, multilingual_features_df):
        """Each language should have expected metrics."""
        result = analyzer.compare_languages(multilingual_features_df)
        
        expected_metrics = ['total_sentences', 'avg_length', 'avg_tree_depth', 
                           'coordination_ratio', 'subordination_ratio']
        
        for lang in ['English', 'German']:
            for metric in expected_metrics:
                assert metric in result[lang], f"Missing metric {metric} for {lang}"
    
    def test_total_sentences_correct(self, analyzer, multilingual_features_df):
        """Sentence counts should be correct per language."""
        result = analyzer.compare_languages(multilingual_features_df)
        
        assert result['English']['total_sentences'] == 25
        assert result['German']['total_sentences'] == 25


# =============================================================================
# Tests for _calculate_tree_depth and _calculate_avg_dependency_distance
# =============================================================================

class TestTreeCalculations:
    """Tests for tree depth and dependency distance calculations."""
    
    def test_tree_depth_simple(self, analyzer):
        """Test tree depth for simple sentence."""
        # Simple structure: root <- child
        from collections import OrderedDict
        
        tokens = [
            OrderedDict([('id', 1), ('form', 'A'), ('head', 2), ('deprel', 'det')]),
            OrderedDict([('id', 2), ('form', 'B'), ('head', 0), ('deprel', 'root')]),
        ]
        
        depth = analyzer._calculate_tree_depth(tokens)
        
        assert depth >= 0
    
    def test_tree_depth_empty(self, analyzer):
        """Tree depth for empty token list should be 0."""
        depth = analyzer._calculate_tree_depth([])
        
        assert depth == 0
    
    def test_avg_dependency_distance_simple(self, analyzer):
        """Test average dependency distance calculation."""
        from collections import OrderedDict
        
        # Token 1 -> head 2 (distance 1)
        # Token 2 -> root (not counted)
        # Token 3 -> head 2 (distance 1)
        tokens = [
            OrderedDict([('id', 1), ('form', 'A'), ('head', 2), ('deprel', 'det')]),
            OrderedDict([('id', 2), ('form', 'B'), ('head', 0), ('deprel', 'root')]),
            OrderedDict([('id', 3), ('form', 'C'), ('head', 2), ('deprel', 'nmod')]),
        ]
        
        distance = analyzer._calculate_avg_dependency_distance(tokens)
        
        # Average of [1, 1] = 1.0
        assert distance == 1.0
    
    def test_avg_dependency_distance_empty(self, analyzer):
        """Distance for empty token list should be 0."""
        distance = analyzer._calculate_avg_dependency_distance([])
        
        assert distance == 0.0


# =============================================================================
# Tests for generate_report
# =============================================================================

class TestGenerateReport:
    """Tests for the generate_report method."""
    
    def test_returns_string(self, analyzer, sample_features_df):
        """Should return a string report."""
        result = analyzer.generate_report(sample_features_df)
        
        assert isinstance(result, str)
    
    def test_contains_header(self, analyzer, sample_features_df):
        """Report should contain header."""
        result = analyzer.generate_report(sample_features_df)
        
        assert "# EDU Dependency Analysis Report" in result
    
    def test_contains_dataset_overview(self, analyzer, sample_features_df):
        """Report should contain dataset overview."""
        result = analyzer.generate_report(sample_features_df)
        
        assert "Dataset Overview" in result
        assert "Total EDUs" in result
    
    def test_saves_to_file_if_specified(self, analyzer, sample_features_df, tmp_path):
        """Should save report to file if output_file specified."""
        output_file = "test_report.md"
        
        analyzer.generate_report(sample_features_df, output_file=output_file)
        
        report_path = analyzer.results_dir / output_file
        assert report_path.exists()


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for DependencyAnalyzer."""
    
    def test_full_analysis_pipeline(self, analyzer, tmp_path, sample_conllu_content):
        """Test complete analysis pipeline."""
        # Create test directory with conllu file
        input_dir = tmp_path / "conllu_files"
        input_dir.mkdir()
        (input_dir / "test.conllu").write_text(sample_conllu_content, encoding='utf-8')
        
        # Load sentences
        sentences = analyzer.load_conllu_files(str(input_dir), "English")
        assert len(sentences) > 0
        
        # Extract features
        features = analyzer.extract_features(sentences)
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        
        # Generate statistics
        stats = analyzer.generate_descriptive_statistics(features)
        assert 'total_sentences' in stats
    
    def test_clustering_produces_valid_output(self, analyzer, sample_features_df):
        """Test that clustering produces consistent, valid output."""
        result = analyzer.perform_clustering_analysis(sample_features_df, n_clusters=3)
        
        # Check consistency
        assert result['n_clusters'] == 3
        
        # All samples should be in exactly one cluster
        clustered_df = result['clustered_data']
        assert len(clustered_df) == len(sample_features_df)
        
        # Silhouette score should be computed
        assert 'silhouette_score' in result
        assert isinstance(result['silhouette_score'], float)

