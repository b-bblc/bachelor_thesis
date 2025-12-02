"""
Unit tests for the boundary_detection module.
"""

import pytest
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.boundary_detection import (
    read_conllu_with_boundaries,
    extract_edu_boundary_features,
    extract_position_features
)


class TestReadConlluWithBoundaries:
    """Tests for the read_conllu_with_boundaries function."""
    
    def test_returns_list_of_sentences(self, sample_conllu_file):
        """Should return a list of sentence lists."""
        sentences = read_conllu_with_boundaries(str(sample_conllu_file))
        
        assert isinstance(sentences, list)
        assert len(sentences) == 2
    
    def test_parses_token_fields_correctly(self, sample_conllu_file):
        """Each token should have all required CoNLL-U fields."""
        sentences = read_conllu_with_boundaries(str(sample_conllu_file))
        
        first_sentence = sentences[0]
        first_token = first_sentence[0]
        
        # Check all CoNLL-U fields are present
        required_fields = ['id', 'form', 'lemma', 'upos', 'xpos', 
                          'feats', 'head', 'deprel', 'deps', 'misc']
        for field in required_fields:
            assert field in first_token, f"Missing field: {field}"
        
        # Check specific values
        assert first_token['form'] == 'This'
        assert first_token['upos'] == 'DET'
        assert first_token['deprel'] == 'det'
    
    def test_sentence_contains_correct_number_of_tokens(self, sample_conllu_file):
        """Each sentence should contain the expected number of tokens."""
        sentences = read_conllu_with_boundaries(str(sample_conllu_file))
        
        assert len(sentences[0]) == 5  # "This is a test ."
        assert len(sentences[1]) == 4  # "Another sentence here ."
    
    def test_handles_empty_file(self, empty_conllu_file):
        """Should return empty list for empty files."""
        sentences = read_conllu_with_boundaries(str(empty_conllu_file))
        
        assert sentences == []
    
    def test_file_not_found_raises_error(self):
        """Should raise FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            read_conllu_with_boundaries("/nonexistent/path.conllu")
    
    def test_handles_comments_correctly(self, sample_conllu_file):
        """Comments should be ignored, not treated as tokens."""
        sentences = read_conllu_with_boundaries(str(sample_conllu_file))
        
        # Comments should not appear in token data
        for sentence in sentences:
            for token in sentence:
                assert not token['form'].startswith('#')


class TestExtractEduBoundaryFeatures:
    """Tests for the extract_edu_boundary_features function."""
    
    def test_returns_dataframe(self, sample_sentences):
        """Should return a pandas DataFrame."""
        result = extract_edu_boundary_features(sample_sentences, 'English')
        
        assert isinstance(result, pd.DataFrame)
    
    def test_includes_expected_columns(self, sample_sentences):
        """DataFrame should include all expected feature columns."""
        result = extract_edu_boundary_features(sample_sentences, 'English')
        
        expected_columns = [
            'language', 'sentence_id', 'token_id', 'form', 'upos', 'deprel',
            'is_sentence_start', 'is_sentence_end', 'relative_position',
            'is_punctuation', 'is_comma', 'is_period',
            'is_coordinating_conj', 'is_subordinating_conj', 'is_conjunction',
            'is_root', 'dep_is_conj', 'dep_is_cc',
            'is_verb', 'is_aux',
            'distance_to_next_punct', 'distance_from_prev_punct',
            'prev_upos', 'next_upos'
        ]
        
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"
    
    def test_language_label_is_correct(self, sample_sentences):
        """Language column should contain the specified language."""
        result = extract_edu_boundary_features(sample_sentences, 'German')
        
        assert all(result['language'] == 'German')
    
    def test_sentence_id_increments(self, sample_sentences):
        """Sentence IDs should correspond to sentence indices."""
        result = extract_edu_boundary_features(sample_sentences, 'English')
        
        # Should have tokens from both sentences (0 and 1)
        assert set(result['sentence_id'].unique()) == {0, 1}
    
    def test_is_sentence_start_feature(self, sample_sentences):
        """is_sentence_start should be True only for first token."""
        result = extract_edu_boundary_features(sample_sentences, 'English')
        
        # First tokens in each sentence should have is_sentence_start=True
        first_tokens = result[result['token_id'] == 1]
        assert all(first_tokens['is_sentence_start'])
        
        # Other tokens should have is_sentence_start=False
        other_tokens = result[result['token_id'] != 1]
        assert not any(other_tokens['is_sentence_start'])
    
    def test_is_punctuation_feature(self, sample_sentences):
        """is_punctuation should correctly identify PUNCT tokens."""
        result = extract_edu_boundary_features(sample_sentences, 'English')
        
        punct_rows = result[result['form'] == '.']
        assert all(punct_rows['is_punctuation'])
        
        non_punct_rows = result[result['upos'] != 'PUNCT']
        assert not any(non_punct_rows['is_punctuation'])
    
    def test_is_root_feature(self, sample_sentences):
        """is_root should be True for tokens with head='0'."""
        result = extract_edu_boundary_features(sample_sentences, 'English')
        
        # 'test' and 'sentence' are roots in our sample
        root_rows = result[result['deprel'] == 'root']
        assert all(root_rows['is_root'])
    
    def test_skips_very_short_sentences(self):
        """Should skip sentences with fewer than 2 tokens."""
        short_sentences = [
            [{'id': '1', 'form': 'Hi', 'lemma': 'hi', 'upos': 'INTJ',
              'xpos': '_', 'feats': '_', 'head': '0', 'deprel': 'root',
              'deps': '_', 'misc': '_'}]
        ]
        result = extract_edu_boundary_features(short_sentences, 'English')
        
        assert len(result) == 0
    
    def test_german_conjunction_detection(self, german_sample_sentences):
        """Should detect German-specific conjunctions."""
        result = extract_edu_boundary_features(german_sample_sentences, 'German')
        
        und_row = result[result['form'] == 'und']
        assert len(und_row) == 1
        assert und_row.iloc[0]['is_und_aber'] == True
        assert und_row.iloc[0]['is_coordinating_conj'] == True
    
    def test_relative_position_ranges(self, sample_sentences):
        """relative_position should be between 0 and 1."""
        result = extract_edu_boundary_features(sample_sentences, 'English')
        
        assert all(result['relative_position'] >= 0)
        assert all(result['relative_position'] <= 1)
    
    def test_context_features_for_first_token(self, sample_sentences):
        """First token should have 'START' for prev_* features."""
        result = extract_edu_boundary_features(sample_sentences, 'English')
        
        first_tokens = result[result['is_sentence_start']]
        assert all(first_tokens['prev_upos'] == 'START')
        assert all(first_tokens['prev_form'] == 'START')
    
    def test_context_features_for_last_token(self, sample_sentences):
        """Last token should have 'END' for next_* features."""
        result = extract_edu_boundary_features(sample_sentences, 'English')
        
        last_tokens = result[result['is_sentence_end']]
        assert all(last_tokens['next_upos'] == 'END')
        assert all(last_tokens['next_form'] == 'END')


class TestExtractPositionFeatures:
    """Tests for the extract_position_features function."""
    
    def test_returns_dataframe(self, sample_sentences):
        """Should return a pandas DataFrame."""
        features_df = extract_edu_boundary_features(sample_sentences, 'English')
        result = extract_position_features(features_df)
        
        assert isinstance(result, pd.DataFrame)
    
    def test_adds_sentence_position_column(self, sample_sentences):
        """Should add sentence_position column."""
        features_df = extract_edu_boundary_features(sample_sentences, 'English')
        result = extract_position_features(features_df)
        
        assert 'sentence_position' in result.columns
    
    def test_preserves_original_columns(self, sample_sentences):
        """Should preserve all original columns."""
        features_df = extract_edu_boundary_features(sample_sentences, 'English')
        original_columns = set(features_df.columns)
        
        result = extract_position_features(features_df)
        
        # All original columns should be present
        for col in original_columns:
            assert col in result.columns
    
    def test_does_not_modify_original_dataframe(self, sample_sentences):
        """Should not modify the original DataFrame."""
        features_df = extract_edu_boundary_features(sample_sentences, 'English')
        original_shape = features_df.shape
        original_columns = list(features_df.columns)
        
        extract_position_features(features_df)
        
        # Original should be unchanged
        assert features_df.shape == original_shape
        assert list(features_df.columns) == original_columns
    
    def test_sentence_position_equals_relative_position(self, sample_sentences):
        """sentence_position should equal relative_position."""
        features_df = extract_edu_boundary_features(sample_sentences, 'English')
        result = extract_position_features(features_df)
        
        assert all(result['sentence_position'] == result['relative_position'])


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_full_pipeline(self, sample_conllu_file):
        """Test complete pipeline from file to features."""
        # Read file
        sentences = read_conllu_with_boundaries(str(sample_conllu_file))
        assert len(sentences) == 2
        
        # Extract features
        features = extract_edu_boundary_features(sentences, 'English')
        assert len(features) > 0
        
        # Add position features
        final_features = extract_position_features(features)
        assert 'sentence_position' in final_features.columns
        
        # Verify data integrity
        assert len(final_features) == sum(len(s) for s in sentences)

