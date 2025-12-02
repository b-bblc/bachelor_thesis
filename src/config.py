"""
Configuration settings for the EDU dependency analysis project.
"""
import logging
from pathlib import Path


# =============================================================================
# Logging Configuration
# =============================================================================
def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure logging for the entire project.
    Call this once at application startup (e.g., in main.py).
    
    Args:
        level: Logging level (default: logging.INFO)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Usually __name__ of the calling module
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
GERMAN_CORPUS_DIR = PROJECT_ROOT / "PotsdamCommentaryCorpus"
RUSSIAN_CORPUS_DIR = PROJECT_ROOT / "RuRsTreebank_full"
ENGLISH_EXAMPLE = PROJECT_ROOT / "en_example.rs3"

# Output directories
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_GERMAN_DIR = RESULTS_DIR / "results_german"
RESULTS_RUSSIAN_DIR = RESULTS_DIR / "results_russian"
VISUALIZATIONS_DIR = RESULTS_DIR / "visualizations"

# Legacy directories for backwards compatibility (will be deprecated)
EXTRACTED_TXTS_DIR = PROJECT_ROOT / "extracted_txts"
PARSED_RESULTS_DIR = PROJECT_ROOT / "parsed_results"

# Create output directories if they don't exist
for dir_path in [RESULTS_DIR, RESULTS_GERMAN_DIR, RESULTS_RUSSIAN_DIR, VISUALIZATIONS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Language models
SPACY_MODELS = {
    'german': 'de_core_news_md',
    'english': 'en_core_web_sm', 
    'russian': 'ru_core_news_sm'
}

# Analysis parameters
MAX_SENTENCE_LENGTH = 100  # Maximum sentence length for filtering
MIN_SENTENCE_LENGTH = 2    # Minimum sentence length for filtering
VISUALIZATION_LIMIT = 10   # Max number of items to show in plots

# File extensions
RST_EXTENSION = '.rs3'
CONLLU_EXTENSION = '.conllu'
TXT_EXTENSION = '.txt'
