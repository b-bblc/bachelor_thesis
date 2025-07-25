"""
Legacy EDU extraction script.
Use src/edu_extractor.py for new projects.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.edu_extractor import EDUExtractor
import warnings

warnings.warn(
    "export.py is deprecated. Use 'python main.py' or the src/edu_extractor.py module instead.",
    DeprecationWarning,
    stacklevel=2
)

def main():
    """Legacy main function for backward compatibility."""
    extractor = EDUExtractor()
    
    # Extract German EDUs
    german_stats = extractor.extract_edus_from_directory(
        "PotsdamCommentaryCorpus/rst", 
        "extracted_txts"
    )
    
    print(f"German extraction complete. Total files: {len(german_stats)}")
    print(f"Total EDUs extracted: {sum(german_stats.values())}")
    
    print("\n⚠️  Consider using the new main.py script for enhanced functionality.")

if __name__ == "__main__":
    main()