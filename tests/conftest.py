"""
Shared pytest fixtures for the EDU dependency analysis test suite.
"""

import pytest
from pathlib import Path


@pytest.fixture
def sample_conllu_content():
    """Sample CoNLL-U content for testing."""
    return """# sent_id = 1
# text = This is a test.
1	This	this	DET	_	_	4	det	_	_
2	is	be	AUX	_	_	4	cop	_	_
3	a	a	DET	_	_	4	det	_	_
4	test	test	NOUN	_	_	0	root	_	_
5	.	.	PUNCT	_	_	4	punct	_	_

# sent_id = 2
# text = Another sentence here.
1	Another	another	DET	_	_	2	det	_	_
2	sentence	sentence	NOUN	_	_	0	root	_	_
3	here	here	ADV	_	_	2	advmod	_	_
4	.	.	PUNCT	_	_	2	punct	_	_

"""


@pytest.fixture
def sample_conllu_file(tmp_path, sample_conllu_content):
    """Create a temporary CoNLL-U file for testing."""
    file_path = tmp_path / "test.conllu"
    file_path.write_text(sample_conllu_content, encoding='utf-8')
    return file_path


@pytest.fixture
def sample_sentences():
    """Sample parsed sentence structures for testing."""
    return [
        [
            {'id': '1', 'form': 'This', 'lemma': 'this', 'upos': 'DET', 
             'xpos': '_', 'feats': '_', 'head': '4', 'deprel': 'det', 
             'deps': '_', 'misc': '_'},
            {'id': '2', 'form': 'is', 'lemma': 'be', 'upos': 'AUX', 
             'xpos': '_', 'feats': '_', 'head': '4', 'deprel': 'cop', 
             'deps': '_', 'misc': '_'},
            {'id': '3', 'form': 'a', 'lemma': 'a', 'upos': 'DET', 
             'xpos': '_', 'feats': '_', 'head': '4', 'deprel': 'det', 
             'deps': '_', 'misc': '_'},
            {'id': '4', 'form': 'test', 'lemma': 'test', 'upos': 'NOUN', 
             'xpos': '_', 'feats': '_', 'head': '0', 'deprel': 'root', 
             'deps': '_', 'misc': '_'},
            {'id': '5', 'form': '.', 'lemma': '.', 'upos': 'PUNCT', 
             'xpos': '_', 'feats': '_', 'head': '4', 'deprel': 'punct', 
             'deps': '_', 'misc': '_'},
        ],
        [
            {'id': '1', 'form': 'Another', 'lemma': 'another', 'upos': 'DET', 
             'xpos': '_', 'feats': '_', 'head': '2', 'deprel': 'det', 
             'deps': '_', 'misc': '_'},
            {'id': '2', 'form': 'sentence', 'lemma': 'sentence', 'upos': 'NOUN', 
             'xpos': '_', 'feats': '_', 'head': '0', 'deprel': 'root', 
             'deps': '_', 'misc': '_'},
            {'id': '3', 'form': 'here', 'lemma': 'here', 'upos': 'ADV', 
             'xpos': '_', 'feats': '_', 'head': '2', 'deprel': 'advmod', 
             'deps': '_', 'misc': '_'},
            {'id': '4', 'form': '.', 'lemma': '.', 'upos': 'PUNCT', 
             'xpos': '_', 'feats': '_', 'head': '2', 'deprel': 'punct', 
             'deps': '_', 'misc': '_'},
        ]
    ]


@pytest.fixture
def german_sample_sentences():
    """Sample German sentences for language-specific testing."""
    return [
        [
            {'id': '1', 'form': 'Das', 'lemma': 'das', 'upos': 'DET', 
             'xpos': '_', 'feats': '_', 'head': '2', 'deprel': 'det', 
             'deps': '_', 'misc': '_'},
            {'id': '2', 'form': 'ist', 'lemma': 'sein', 'upos': 'AUX', 
             'xpos': '_', 'feats': 'VerbForm=Fin', 'head': '0', 'deprel': 'root', 
             'deps': '_', 'misc': '_'},
            {'id': '3', 'form': 'gut', 'lemma': 'gut', 'upos': 'ADJ', 
             'xpos': '_', 'feats': '_', 'head': '2', 'deprel': 'advmod', 
             'deps': '_', 'misc': '_'},
            {'id': '4', 'form': ',', 'lemma': ',', 'upos': 'PUNCT', 
             'xpos': '_', 'feats': '_', 'head': '2', 'deprel': 'punct', 
             'deps': '_', 'misc': '_'},
            {'id': '5', 'form': 'und', 'lemma': 'und', 'upos': 'CCONJ', 
             'xpos': '_', 'feats': '_', 'head': '7', 'deprel': 'cc', 
             'deps': '_', 'misc': '_'},
            {'id': '6', 'form': 'das', 'lemma': 'das', 'upos': 'DET', 
             'xpos': '_', 'feats': '_', 'head': '7', 'deprel': 'det', 
             'deps': '_', 'misc': '_'},
            {'id': '7', 'form': 'auch', 'lemma': 'auch', 'upos': 'ADV', 
             'xpos': '_', 'feats': '_', 'head': '2', 'deprel': 'conj', 
             'deps': '_', 'misc': '_'},
            {'id': '8', 'form': '.', 'lemma': '.', 'upos': 'PUNCT', 
             'xpos': '_', 'feats': '_', 'head': '2', 'deprel': 'punct', 
             'deps': '_', 'misc': '_'},
        ]
    ]


@pytest.fixture
def empty_conllu_file(tmp_path):
    """Create an empty CoNLL-U file for testing."""
    file_path = tmp_path / "empty.conllu"
    file_path.write_text("", encoding='utf-8')
    return file_path

