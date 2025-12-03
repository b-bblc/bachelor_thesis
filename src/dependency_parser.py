"""
Dependency parsing utilities for EDUs using spaCy.
"""
import spacy
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from spacy.tokens import Doc

from .config import SPACY_MODELS, CONLLU_EXTENSION, TXT_EXTENSION, get_logger

logger = get_logger(__name__)


class DependencyParser:
    """Class for parsing dependencies in EDUs using spaCy."""
    
    def __init__(self, language: str = 'german'):
        """
        Initialize the dependency parser.
        
        Args:
            language: Language for parsing ('german', 'english', 'russian')
        """
        self.language = language
        self.model_name = SPACY_MODELS.get(language)
        
        if not self.model_name:
            raise ValueError(f"Unsupported language: {language}")
        
        try:
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Loaded {self.model_name} model for {language}")
        except OSError:
            logger.error(f"Model {self.model_name} not found. Please install it with:")
            logger.error(f"python -m spacy download {self.model_name}")
            raise
    
    def parse_edu(self, edu_text: str) -> Doc:
        """
        Parse a single EDU for dependencies.
        
        Args:
            edu_text: Text of the EDU
            
        Returns:
            spaCy Doc object with dependency information
        """
        return self.nlp(edu_text)
    
    def parse_text_file(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """
        Parse a text file containing EDUs (one per line) and save as CoNLL-U.
        
        Args:
            input_path: Path to input .txt file
            output_path: Path to output .conllu file
            
        Returns:
            Dictionary with parsing statistics
        """
        stats = {
            'total_edus': 0,
            'total_tokens': 0,
            'avg_edu_length': 0.0,
            'parsing_errors': 0
        }
        
        try:
            with open(input_path, 'r', encoding='utf-8') as infile, \
                 open(output_path, 'w', encoding='utf-8') as outfile:
                
                for edu_num, line in enumerate(infile, 1):
                    edu_text = line.strip()
                    
                    if not edu_text:
                        continue
                    
                    try:
                        # Parse the EDU
                        doc = self.parse_edu(edu_text)
                        
                        # Write CoNLL-U format
                        self._write_conllu_sentence(outfile, doc, edu_num, edu_text)
                        
                        stats['total_edus'] += 1
                        stats['total_tokens'] += len(doc)
                        
                    except ValueError as e:
                        logger.warning(f"Value error parsing EDU {edu_num}: {e}")
                        stats['parsing_errors'] += 1
                    except RuntimeError as e:
                        logger.warning(f"Runtime error parsing EDU {edu_num}: {e}")
                        stats['parsing_errors'] += 1
                
                # Calculate average EDU length
                if stats['total_edus'] > 0:
                    stats['avg_edu_length'] = stats['total_tokens'] / stats['total_edus']
                
                logger.info(f"Parsed {input_path}: {stats['total_edus']} EDUs, "
                          f"{stats['total_tokens']} tokens")
                
        except FileNotFoundError:
            logger.error(f"File not found: {input_path}")
            stats['parsing_errors'] += 1
        except PermissionError:
            logger.error(f"Permission denied: {input_path}")
            stats['parsing_errors'] += 1
        except OSError as e:
            logger.error(f"OS error processing {input_path}: {e}")
            stats['parsing_errors'] += 1
        
        return stats
    
    def _write_conllu_sentence(self, outfile, doc: Doc, sent_id: int, text: str):
        """Write a sentence in CoNLL-U format."""
        outfile.write(f"# sent_id = {sent_id}\n")
        outfile.write(f"# text = {text}\n")
        
        for i, token in enumerate(doc, 1):
            # Handle head index (spaCy uses 0-based, CoNLL-U uses 1-based)
            head_idx = token.head.i + 1 if token.head.i != token.i else 0
            
            # Write token information in CoNLL-U format
            outfile.write(
                f"{i}\t{token.text}\t{token.lemma_}\t{token.pos_}\t_\t"
                f"{token.morph}\t{head_idx}\t{token.dep_}\t_\t_\n"
            )
        
        outfile.write("\n")
    
    def parse_directory(self, input_dir: str, output_dir: str) -> Dict[str, Dict]:
        """
        Parse all .txt files in a directory.
        
        Args:
            input_dir: Directory containing .txt files with EDUs
            output_dir: Directory to save .conllu files
            
        Returns:
            Dictionary with parsing statistics for each file
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        all_stats = {}
        
        # Find all .txt files
        txt_files = list(input_path.glob(f"*{TXT_EXTENSION}"))
        
        for txt_file in txt_files:
            # Generate output filename
            output_filename = txt_file.stem + "_parsed" + CONLLU_EXTENSION
            output_filepath = output_path / output_filename
            
            # Parse the file
            stats = self.parse_text_file(str(txt_file), str(output_filepath))
            all_stats[str(txt_file)] = stats
        
        return all_stats
    
    def extract_dependency_features(self, doc: Doc) -> Dict[str, Any]:
        """
        Extract dependency features from a parsed document.
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            Dictionary with dependency features
        """
        features = {
            'length': len(doc),
            'pos_counts': {},
            'dep_counts': {},
            'tree_depth': 0,
            'root_pos': None,
            'has_subordinate_clauses': False,
            'conjunctions_count': 0
        }
        
        # Count POS tags and dependency relations
        for token in doc:
            pos = token.pos_
            dep = token.dep_
            
            features['pos_counts'][pos] = features['pos_counts'].get(pos, 0) + 1
            features['dep_counts'][dep] = features['dep_counts'].get(dep, 0) + 1
            
            # Check for root
            if dep == 'ROOT':
                features['root_pos'] = pos
            
            # Check for subordinate clauses
            if dep in ['mark', 'nsubj:xsubj', 'ccomp', 'xcomp', 'advcl']:
                features['has_subordinate_clauses'] = True
            
            # Count conjunctions
            if pos == 'CCONJ' or dep == 'cc':
                features['conjunctions_count'] += 1
        
        # Calculate tree depth
        features['tree_depth'] = self._calculate_tree_depth(doc)
        
        return features
    
    def _calculate_tree_depth(self, doc: Doc) -> int:
        """Calculate the maximum depth of the dependency tree."""
        def get_depth(token, current_depth=0):
            if not list(token.children):
                return current_depth
            return max(get_depth(child, current_depth + 1) for child in token.children)
        
        # Find root tokens
        roots = [token for token in doc if token.head == token]
        if not roots:
            return 0
        
        return max(get_depth(root) for root in roots)


def main():
    """Main function for standalone execution."""
    # Parse German EDUs
    parser = DependencyParser('german')
    stats = parser.parse_directory("extracted_txts", "parsed_results")
    
    print(f"Parsing complete. Processed {len(stats)} files.")
    
    # Print summary statistics
    total_edus = sum(file_stats['total_edus'] for file_stats in stats.values())
    total_tokens = sum(file_stats['total_tokens'] for file_stats in stats.values())
    
    print(f"Total EDUs parsed: {total_edus}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average tokens per EDU: {total_tokens / total_edus if total_edus > 0 else 0:.2f}")


if __name__ == "__main__":
    main()
