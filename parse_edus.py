"""
Legacy EDU parsing script. 
Use src/dependency_parser.py for new projects.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.dependency_parser import DependencyParser
import warnings

warnings.warn(
    "parse_edus.py is deprecated. Use 'python main.py' or the src/dependency_parser.py module instead.",
    DeprecationWarning,
    stacklevel=2
)

def main():
    """Legacy main function for backward compatibility."""
    parser = DependencyParser('german')
    stats = parser.parse_directory("extracted_txts", "parsed_results")
    
    print(f"Parsing complete. Processed {len(stats)} files.")
    
    # Print summary statistics
    total_edus = sum(file_stats['total_edus'] for file_stats in stats.values())
    total_tokens = sum(file_stats['total_tokens'] for file_stats in stats.values())
    
    print(f"Total EDUs parsed: {total_edus}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average tokens per EDU: {total_tokens / total_edus if total_edus > 0 else 0:.2f}")
    
    print("\n⚠️  Consider using the new main.py script for enhanced functionality.")

if __name__ == "__main__":
    main()
