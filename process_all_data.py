#!/usr/bin/env python3
"""
Full data processing pipeline for German and Russian EDU analysis.
This script runs the complete pipeline from extraction to analysis.
"""

import logging
import sys
import time
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_corpus_data():
    """Check that all corpus data is available."""
    logger.info("Checking corpus data availability...")
    
    checks = {
        'German corpus': Path('PotsdamCommentaryCorpus/PotsdamCommentaryCorpus/rst'),
        'Russian corpus': Path('RuRsTreebank_full')
    }
    
    for name, path in checks.items():
        if not path.exists():
            logger.error(f"{name} not found at {path}")
            return False
        
        rst_files = list(path.rglob("*.rs3"))
        logger.info(f"✓ {name}: {len(rst_files)} RST files found")
    
    return True

def process_language(language):
    """Process a single language through the full pipeline."""
    logger.info(f"=" * 60)
    logger.info(f"PROCESSING {language.upper()}")
    logger.info(f"=" * 60)
    
    start_time = time.time()
    
    # Import here to avoid import issues
    sys.path.append(str(Path(__file__).parent / 'src'))
    from src.edu_extractor import EDUExtractor
    from src.dependency_parser import DependencyParser
    
    # Configuration for each language
    configs = {
        'german': {
            'input_dir': 'PotsdamCommentaryCorpus/PotsdamCommentaryCorpus/rst',
            'extract_dir': 'results/results_german/extracted_edus',
            'parse_dir': 'results/results_german/parsed_dependencies'
        },
        'russian': {
            'input_dir': 'RuRsTreebank_full',
            'extract_dir': 'results/results_russian/extracted_edus', 
            'parse_dir': 'results/results_russian/parsed_dependencies'
        }
    }
    
    if language not in configs:
        logger.error(f"Unknown language: {language}")
        return False
    
    config = configs[language]
    
    try:
        # Step 1: Extract EDUs
        logger.info(f"Step 1: Extracting EDUs for {language}...")
        extractor = EDUExtractor()
        
        # Create directories
        Path(config['extract_dir']).mkdir(exist_ok=True, parents=True)
        Path(config['parse_dir']).mkdir(exist_ok=True, parents=True)
        
        extraction_stats = extractor.extract_edus_from_directory(
            config['input_dir'], 
            config['extract_dir']
        )
        
        total_edus = sum(extraction_stats.values())
        logger.info(f"✓ Extracted {total_edus} EDUs from {len(extraction_stats)} files")
        
        # Step 2: Parse dependencies
        logger.info(f"Step 2: Parsing dependencies for {language}...")
        parser = DependencyParser(language)
        
        parsing_stats = parser.parse_directory(
            config['extract_dir'], 
            config['parse_dir']
        )
        
        total_tokens = sum(stats.get('total_tokens', 0) for stats in parsing_stats.values())
        logger.info(f"✓ Parsed {total_tokens} tokens from {len(parsing_stats)} files")
        
        # Step 3: Generate summary
        elapsed_time = time.time() - start_time
        logger.info(f"✓ {language.title()} processing complete in {elapsed_time:.1f}s")
        logger.info(f"  - Files processed: {len(extraction_stats)}")
        logger.info(f"  - EDUs extracted: {total_edus}")
        logger.info(f"  - Tokens parsed: {total_tokens}")
        
        return {
            'language': language,
            'files_processed': len(extraction_stats),
            'total_edus': total_edus,
            'total_tokens': total_tokens,
            'processing_time': elapsed_time,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Error processing {language}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'language': language,
            'success': False,
            'error': str(e)
        }

def generate_summary(results):
    """Generate processing summary."""
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 60)
    
    total_files = 0
    total_edus = 0
    total_tokens = 0
    total_time = 0
    
    for result in results:
        if result['success']:
            lang = result['language'].title()
            logger.info(f"{lang}:")
            logger.info(f"  ✓ Files: {result['files_processed']}")
            logger.info(f"  ✓ EDUs: {result['total_edus']:,}")
            logger.info(f"  ✓ Tokens: {result['total_tokens']:,}")
            logger.info(f"  ✓ Time: {result['processing_time']:.1f}s")
            
            total_files += result['files_processed']
            total_edus += result['total_edus']
            total_tokens += result['total_tokens']
            total_time += result['processing_time']
        else:
            logger.error(f"{result['language'].title()}: FAILED - {result['error']}")
    
    logger.info(f"\nCOMBINED TOTALS:")
    logger.info(f"  📁 Files processed: {total_files}")
    logger.info(f"  📝 EDUs extracted: {total_edus:,}")
    logger.info(f"  🔤 Tokens parsed: {total_tokens:,}")
    logger.info(f"  ⏱️  Total time: {total_time:.1f}s")
    
    # Next steps
    logger.info(f"\n🚀 NEXT STEPS:")
    logger.info(f"1. Run analysis: python src/analysis.py")
    logger.info(f"2. Check results in results/results_german/ and results/results_russian/")
    logger.info(f"3. Run Jupyter notebooks for visualization")

def main():
    """Main processing function."""
    logger.info("🚀 STARTING FULL DATA PROCESSING")
    logger.info("📊 Languages: German 🇩🇪 and Russian 🇷🇺")
    
    # Check data availability
    if not check_corpus_data():
        logger.error("❌ Required corpus data not found. Please run download_german_corpus.py first.")
        return False
    
    # Process each language
    languages = ['german', 'russian']
    results = []
    
    start_total = time.time()
    
    for language in languages:
        result = process_language(language)
        results.append(result)
        
        # Small break between languages
        if language != languages[-1]:
            logger.info("⏸️  Brief pause before next language...\n")
            time.sleep(2)
    
    # Generate final summary
    total_elapsed = time.time() - start_total
    generate_summary(results)
    
    logger.info(f"\n✅ FULL PROCESSING COMPLETE!")
    logger.info(f"⏱️  Total elapsed time: {total_elapsed:.1f}s")
    
    # Check for any failures
    failed = [r for r in results if not r['success']]
    if failed:
        logger.warning(f"⚠️  {len(failed)} language(s) failed processing")
        return False
    else:
        logger.info("🎉 All languages processed successfully!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
