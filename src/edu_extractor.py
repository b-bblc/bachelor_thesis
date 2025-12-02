"""
Utilities for extracting EDUs from RST (.rs3) files.
"""
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict

from .config import RST_EXTENSION, TXT_EXTENSION, get_logger

logger = get_logger(__name__)


class EDUExtractor:
    """Class for extracting Elementary Discourse Units from RST files."""
    
    def __init__(self):
        self.extracted_edus = []
        self.extraction_stats = {}
    
    def extract_edus_from_rs3(self, rs3_path: str) -> List[str]:
        """
        Extract EDUs from a single .rs3 file.
        
        Args:
            rs3_path: Path to the .rs3 file
            
        Returns:
            List of EDU texts
        """
        try:
            tree = ET.parse(rs3_path)
            root = tree.getroot()
            
            # Find all segment elements
            segments = root.findall('.//segment')
            
            # Extract text from segments, clean and filter
            edus = []
            for segment in segments:
                if segment.text:
                    edu_text = segment.text.strip()
                    # Remove placeholder text
                    edu_text = edu_text.replace("#####", "")
                    # Filter out very short or URL-containing EDUs
                    if len(edu_text) > 1 and "http" not in edu_text.lower():
                        edus.append(edu_text)
            
            logger.info(f"Extracted {len(edus)} EDUs from {rs3_path}")
            return edus
            
        except ET.ParseError as e:
            logger.error(f"Error parsing {rs3_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error processing {rs3_path}: {e}")
            return []
    
    def extract_edus_from_directory(self, input_dir: str, output_dir: str) -> Dict[str, int]:
        """
        Extract EDUs from all .rs3 files in a directory.
        
        Args:
            input_dir: Directory containing .rs3 files
            output_dir: Directory to save extracted EDUs as .txt files
            
        Returns:
            Dictionary with filenames and EDU counts
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Find all .rs3 files recursively
        rs3_files = list(input_path.rglob(f"*{RST_EXTENSION}"))
        
        extraction_stats = {}
        
        for rs3_file in rs3_files:
            # Extract EDUs
            edus = self.extract_edus_from_rs3(str(rs3_file))
            
            # Generate output filename
            output_filename = rs3_file.stem + TXT_EXTENSION
            output_filepath = output_path / output_filename
            
            # Save EDUs to text file
            with open(output_filepath, 'w', encoding='utf-8') as f:
                for edu in edus:
                    f.write(edu + '\n')
            
            extraction_stats[output_filename] = len(edus)
            logger.info(f"Saved {len(edus)} EDUs to {output_filepath}")
        
        return extraction_stats
    
    def get_sentence_boundaries(self, edus: List[str]) -> List[Tuple[int, int]]:
        """
        Identify sentence boundaries in a list of EDUs.
        Groups EDUs that likely belong to the same sentence.
        
        Args:
            edus: List of EDU texts
            
        Returns:
            List of (start_idx, end_idx) tuples for sentence boundaries
        """
        boundaries = []
        current_start = 0
        
        for i, edu in enumerate(edus):
            # Check if EDU ends with sentence-final punctuation
            if edu.strip().endswith(('.', '!', '?', ':', ';')):
                boundaries.append((current_start, i + 1))
                current_start = i + 1
        
        # Add remaining EDUs as final sentence if any
        if current_start < len(edus):
            boundaries.append((current_start, len(edus)))
        
        return boundaries
    
    def group_edus_to_sentences(self, edus: List[str]) -> List[str]:
        """
        Group EDUs into complete sentences.
        
        Args:
            edus: List of EDU texts
            
        Returns:
            List of complete sentences
        """
        boundaries = self.get_sentence_boundaries(edus)
        sentences = []
        
        for start, end in boundaries:
            sentence = " ".join(edus[start:end])
            sentences.append(sentence)
        
        return sentences


def main():
    """Main function for standalone execution."""
    extractor = EDUExtractor()
    
    # Extract German EDUs
    german_stats = extractor.extract_edus_from_directory(
        "PotsdamCommentaryCorpus/rst", 
        "extracted_txts"
    )
    
    print(f"German extraction complete. Total files: {len(german_stats)}")
    print(f"Total EDUs extracted: {sum(german_stats.values())}")


if __name__ == "__main__":
    main()
