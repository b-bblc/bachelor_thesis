"""
Utilities for EDU boundary detection from dependency parses.

This module provides functions to analyze CoNLL-U parsed data and extract
features that indicate potential Elementary Discourse Unit (EDU) boundaries.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any


def read_conllu_with_boundaries(file_path: str) -> List[List[Dict[str, str]]]:
    """
    Read CoNLL-U file and identify potential EDU boundaries.
    
    Args:
        file_path: Path to the .conllu file
        
    Returns:
        List of sentences, where each sentence is a list of token dictionaries.
        Each token dictionary contains CoNLL-U fields: id, form, lemma, upos,
        xpos, feats, head, deprel, deps, misc.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CoNLL-U file not found: {file_path}")
    
    sentences: List[List[Dict[str, str]]] = []
    current_sentence: List[Dict[str, str]] = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Handle sentence boundaries
            if not line or line.startswith('#'):
                if current_sentence and not line:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue
            
            # Parse token line
            fields = line.split('\t')
            if len(fields) >= 10:
                token = {
                    'id': fields[0],
                    'form': fields[1],
                    'lemma': fields[2],
                    'upos': fields[3],
                    'xpos': fields[4],
                    'feats': fields[5],
                    'head': fields[6],
                    'deprel': fields[7],
                    'deps': fields[8],
                    'misc': fields[9]
                }
                current_sentence.append(token)
    
    # Add last sentence if exists
    if current_sentence:
        sentences.append(current_sentence)
    
    return sentences


def extract_edu_boundary_features(
    sentences: List[List[Dict[str, str]]], 
    language: str
) -> pd.DataFrame:
    """
    Extract features that could indicate EDU boundaries.
    
    Args:
        sentences: List of sentences from read_conllu_with_boundaries
        language: Language label ('German', 'Russian', 'English')
        
    Returns:
        DataFrame with extracted boundary features for each token.
    """
    boundary_data = []
    
    for sent_idx, sentence in enumerate(sentences):
        if len(sentence) < 2:  # Skip very short sentences
            continue
            
        for token_idx, token in enumerate(sentence):
            # Skip compound tokens (those with ranges like "1-2")
            if '-' in token['id']:
                continue
            
            # Basic token information
            try:
                token_id = int(token['id'])
            except ValueError:
                continue  # Skip non-integer token IDs
                
            form = token['form']
            upos = token['upos']
            deprel = token['deprel']
            head = token['head']
            
            # Boundary features
            features: Dict[str, Any] = {
                'language': language,
                'sentence_id': sent_idx,
                'token_id': token_id,
                'form': form,
                'upos': upos,
                'deprel': deprel,
                
                # Position features
                'is_sentence_start': token_id == 1,
                'is_sentence_end': token_idx == len(sentence) - 1,
                'relative_position': token_idx / len(sentence),
                
                # Punctuation features
                'is_punctuation': upos == 'PUNCT',
                'is_comma': form == ',',
                'is_period': form == '.',
                'is_colon': form == ':',
                'is_semicolon': form == ';',
                'ends_with_punct': any(p in form for p in '.!?'),
                
                # Conjunction features
                'is_coordinating_conj': upos == 'CCONJ',
                'is_subordinating_conj': upos == 'SCONJ',
                'is_conjunction': upos in ['CCONJ', 'SCONJ'],
                
                # Common conjunctions (language-specific)
                'is_und_aber': form.lower() in ['und', 'aber', 'oder', 'doch'] if language == 'German' else False,
                'is_i_no': form.lower() in ['и', 'а', 'но', 'или', 'да'] if language == 'Russian' else False,
                
                # Dependency features
                'is_root': head == '0',
                'dep_is_conj': deprel == 'conj',
                'dep_is_cc': deprel == 'cc',
                'dep_is_advcl': deprel == 'advcl',
                'dep_is_acl': deprel == 'acl',
                'dep_is_ccomp': deprel == 'ccomp',
                'dep_is_xcomp': deprel == 'xcomp',
                
                # Verb features
                'is_verb': upos == 'VERB',
                'is_aux': upos == 'AUX',
                'is_finite_verb': upos == 'VERB' and 'VerbForm=Fin' in token.get('feats', ''),
            }
            
            # Calculate distance to next punctuation
            next_punct_distance = 0
            for i in range(token_idx + 1, len(sentence)):
                next_punct_distance += 1
                if sentence[i]['upos'] == 'PUNCT':
                    break
            else:
                next_punct_distance = -1  # No punctuation found
            
            features['distance_to_next_punct'] = next_punct_distance
            
            # Calculate distance to previous punctuation
            prev_punct_distance = 0
            for i in range(token_idx - 1, -1, -1):
                prev_punct_distance += 1
                if sentence[i]['upos'] == 'PUNCT':
                    break
            else:
                prev_punct_distance = -1  # No punctuation found
            
            features['distance_from_prev_punct'] = prev_punct_distance
            
            # Context features (look at neighboring tokens)
            if token_idx > 0:
                prev_token = sentence[token_idx - 1]
                features['prev_upos'] = prev_token['upos']
                features['prev_form'] = prev_token['form']
                features['prev_deprel'] = prev_token['deprel']
            else:
                features['prev_upos'] = 'START'
                features['prev_form'] = 'START'
                features['prev_deprel'] = 'START'
                
            if token_idx < len(sentence) - 1:
                next_token = sentence[token_idx + 1]
                features['next_upos'] = next_token['upos']
                features['next_form'] = next_token['form']
                features['next_deprel'] = next_token['deprel']
            else:
                features['next_upos'] = 'END'
                features['next_form'] = 'END'
                features['next_deprel'] = 'END'
            
            boundary_data.append(features)
    
    return pd.DataFrame(boundary_data)


def extract_position_features(boundary_features: pd.DataFrame) -> pd.DataFrame:
    """
    Extract position-based features for boundary detection.
    
    Args:
        boundary_features: DataFrame from extract_edu_boundary_features
        
    Returns:
        DataFrame with additional position-based features.
    """
    result = boundary_features.copy()
    
    # Rename existing relative_position to sentence_position for consistency
    if 'relative_position' in result.columns:
        result['sentence_position'] = result['relative_position']
    
    return result

