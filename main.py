#!/usr/bin/env python3
"""
Main processing script for EDU dependency analysis across multiple languages.
This script orchestrates the complete pipeline from extraction to analysis.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.edu_extractor import EDUExtractor
from src.dependency_parser import DependencyParser
from src.analysis import DependencyAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_language(language: str, input_dir: str, 
                    extract_dir: str, parse_dir: str) -> dict:
    """
    Process a single language through the complete pipeline.
    
    Args:
        language: Language name ('german', 'russian', 'english')
        input_dir: Directory containing RST files
        extract_dir: Directory for extracted EDUs
        parse_dir: Directory for parsed results
        
    Returns:
        Dictionary with processing statistics
    """
    logger.info(f"Processing {language.upper()} data...")
    
    stats = {
        'language': language,
        'extraction_stats': {},
        'parsing_stats': {},
        'total_edus': 0,
        'total_tokens': 0
    }
    
    try:
        # Step 1: Extract EDUs
        logger.info(f"Extracting EDUs for {language}...")
        extractor = EDUExtractor()
        
        if language == 'english':
            # Special handling for single English file
            edus = extractor.extract_edus_from_rs3(input_dir)
            extract_path = Path(extract_dir)
            extract_path.mkdir(exist_ok=True, parents=True)
            
            output_file = extract_path / "en_example.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                for edu in edus:
                    f.write(edu + '\n')
            
            stats['extraction_stats'] = {'en_example.txt': len(edus)}
        else:
            # Standard directory processing
            stats['extraction_stats'] = extractor.extract_edus_from_directory(
                input_dir, extract_dir
            )
        
        stats['total_edus'] = sum(stats['extraction_stats'].values())
        logger.info(f"Extracted {stats['total_edus']} EDUs for {language}")
        
        # Step 2: Parse dependencies
        logger.info(f"Parsing dependencies for {language}...")
        parser = DependencyParser(language)
        stats['parsing_stats'] = parser.parse_directory(extract_dir, parse_dir)
        
        # Calculate total tokens
        for file_stats in stats['parsing_stats'].values():
            stats['total_tokens'] += file_stats.get('total_tokens', 0)
        
        logger.info(f"Parsed {stats['total_tokens']} tokens for {language}")
        
    except Exception as e:
        logger.error(f"Error processing {language}: {e}")
        stats['error'] = str(e)
    
    return stats


def run_analysis(languages: list) -> dict:
    """
    Run comprehensive analysis across all languages.
    
    Args:
        languages: List of language dictionaries with processing stats
        
    Returns:
        Dictionary with analysis results
    """
    logger.info("Running comprehensive analysis...")
    
    analyzer = DependencyAnalyzer()
    combined_features = []
    
    # Load data for each language
    for lang_stats in languages:
        if 'error' in lang_stats:
            logger.warning(f"Skipping {lang_stats['language']} due to error")
            continue
        
        language = lang_stats['language']
        parse_dir = f"parsed_results_{language}" if language != 'german' else "parsed_results"
        
        try:
            # Load sentences and extract features
            sentences = analyzer.load_conllu_files(parse_dir, language.title())
            features = analyzer.extract_features(sentences)
            combined_features.append(features)
            
            logger.info(f"Loaded {len(features)} EDUs for {language}")
            
        except Exception as e:
            logger.error(f"Error analyzing {language}: {e}")
    
    if not combined_features:
        logger.error("No data available for analysis")
        return {}
    
    # Combine all features
    import pandas as pd
    all_features = pd.concat(combined_features, ignore_index=True)
    
    # Generate comprehensive analysis
    analysis_results = {
        'total_edus': len(all_features),
        'languages': all_features['language'].value_counts().to_dict(),
        'descriptive_stats': analyzer.generate_descriptive_statistics(all_features)
    }
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    analyzer.visualize_feature_distributions(all_features, save_plots=True)
    
    # Perform clustering analysis
    logger.info("Performing clustering analysis...")
    clustering_results = analyzer.perform_clustering_analysis(all_features, n_clusters=5)
    analysis_results['clustering'] = clustering_results
    
    # Cross-linguistic comparison
    if all_features['language'].nunique() > 1:
        logger.info("Performing cross-linguistic comparison...")
        lang_comparison = analyzer.compare_languages(all_features)
        analysis_results['language_comparison'] = lang_comparison
    
    # Generate report
    logger.info("Generating analysis report...")
    report = analyzer.generate_report(all_features, 'comprehensive_analysis_report.md')
    
    return analysis_results


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Process EDU dependency analysis across multiple languages"
    )
    parser.add_argument(
        '--languages', 
        nargs='+', 
        default=['german', 'russian', 'english'],
        help='Languages to process (default: german russian english)'
    )
    parser.add_argument(
        '--skip-extraction', 
        action='store_true',
        help='Skip EDU extraction step'
    )
    parser.add_argument(
        '--skip-parsing', 
        action='store_true',
        help='Skip dependency parsing step'
    )
    parser.add_argument(
        '--analysis-only', 
        action='store_true',
        help='Run analysis only (skip extraction and parsing)'
    )
    
    args = parser.parse_args()
    
    # Define language configurations
    language_configs = {
        'german': {
            'input_dir': 'PotsdamCommentaryCorpus/rst',
            'extract_dir': 'extracted_txts',
            'parse_dir': 'parsed_results'
        },
        'russian': {
            'input_dir': 'RuRsTreebank_full',
            'extract_dir': 'extracted_txts_russian',
            'parse_dir': 'parsed_results_russian'
        },
        'english': {
            'input_dir': 'en_example.rs3',
            'extract_dir': 'extracted_txts_english',
            'parse_dir': 'parsed_results_english'
        }
    }
    
    # Process each language
    all_language_stats = []
    
    if not args.analysis_only:
        for language in args.languages:
            if language not in language_configs:
                logger.warning(f"Unknown language: {language}. Skipping.")
                continue
            
            config = language_configs[language]
            
            # Check if input data exists
            input_path = Path(config['input_dir'])
            if not input_path.exists():
                logger.warning(f"Input directory not found for {language}: {input_path}")
                continue
            
            # Process the language
            stats = process_language(
                language=language,
                input_dir=str(input_path),
                extract_dir=config['extract_dir'],
                parse_dir=config['parse_dir']
            )
            all_language_stats.append(stats)
    
    # Run comprehensive analysis
    analysis_results = run_analysis(all_language_stats)
    
    # Print summary
    logger.info("\\n" + "="*50)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*50)
    
    if analysis_results:
        logger.info(f"Total EDUs analyzed: {analysis_results['total_edus']}")
        logger.info("Language distribution:")
        for lang, count in analysis_results['languages'].items():
            logger.info(f"  {lang}: {count} EDUs")
        
        if 'clustering' in analysis_results:
            clustering = analysis_results['clustering']
            logger.info(f"Clustering: {clustering['n_clusters']} clusters, "
                       f"silhouette score: {clustering['silhouette_score']:.3f}")
    
    logger.info("\\nCheck the results/ directory for detailed analysis outputs.")
    logger.info("Run the comprehensive_multilingual_analysis.ipynb notebook for interactive exploration.")


if __name__ == "__main__":
    main()
