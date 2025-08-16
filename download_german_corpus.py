#!/usr/bin/env python3
"""
Script to download and extract the Potsdam Commentary Corpus (German RST corpus).
This corpus contains German newspaper commentaries annotated with RST structure.
"""

import requests
import zipfile
import os
import shutil
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# URLs and paths
CORPUS_URL = "https://angcl.ling.uni-potsdam.de/resources/pcc2.2.zip"
DOWNLOAD_DIR = Path("downloads")
EXTRACT_DIR = Path("PotsdamCommentaryCorpus")
ZIP_FILENAME = "pcc2.2.zip"

def download_corpus():
    """Download the PCC corpus zip file."""
    logger.info(f"Starting download of Potsdam Commentary Corpus from {CORPUS_URL}")
    
    # Create downloads directory
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    zip_path = DOWNLOAD_DIR / ZIP_FILENAME
    
    try:
        # Download with progress tracking
        response = requests.get(CORPUS_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\rDownload progress: {progress:.1f}%", end='', flush=True)
        
        print()  # New line after progress
        logger.info(f"Downloaded {downloaded_size:,} bytes to {zip_path}")
        return zip_path
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download corpus: {e}")
        raise

def extract_corpus(zip_path):
    """Extract the corpus and organize RST files."""
    logger.info(f"Extracting corpus from {zip_path}")
    
    # Remove existing extraction directory
    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)
    
    # Extract zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("temp_extract")
    
    # Find the actual corpus directory
    temp_dir = Path("temp_extract")
    corpus_dirs = list(temp_dir.glob("**/"))
    
    # Look for directory containing RST files
    rst_dir = None
    for d in corpus_dirs:
        rst_files = list(d.glob("**/*.rs3"))
        if rst_files:
            rst_dir = d
            break
    
    if rst_dir is None:
        # Try to find any directory with corpus files
        for d in corpus_dirs:
            if any(d.glob("**/*")):
                rst_dir = d
                break
    
    if rst_dir:
        # Move to final location
        shutil.move(str(rst_dir), str(EXTRACT_DIR))
        logger.info(f"Corpus extracted to {EXTRACT_DIR}")
    else:
        # Just move the entire temp directory
        shutil.move("temp_extract", str(EXTRACT_DIR))
        logger.info(f"Corpus extracted to {EXTRACT_DIR}")
    
    # Clean up temp directory if it still exists
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

def find_rst_files():
    """Find and count RST files in the extracted corpus."""
    logger.info("Searching for RST files...")
    
    rst_files = list(EXTRACT_DIR.rglob("*.rs3"))
    
    if rst_files:
        logger.info(f"Found {len(rst_files)} RST files:")
        
        # Group by subdirectory
        dirs = {}
        for f in rst_files:
            parent = f.parent.name
            if parent not in dirs:
                dirs[parent] = []
            dirs[parent].append(f.name)
        
        for dir_name, files in dirs.items():
            logger.info(f"  {dir_name}: {len(files)} files")
            
        # Show first few file examples
        logger.info("\nFirst 5 RST files found:")
        for i, f in enumerate(rst_files[:5]):
            logger.info(f"  {i+1}. {f.relative_to(EXTRACT_DIR)}")
            
        return rst_files
    else:
        logger.warning("No RST files found! Let's check what was extracted:")
        
        # List all files to debug
        all_files = list(EXTRACT_DIR.rglob("*"))[:20]  # First 20 files
        for f in all_files:
            if f.is_file():
                logger.info(f"  Found file: {f.relative_to(EXTRACT_DIR)}")
        
        return []

def setup_corpus_structure():
    """Set up proper directory structure for the project."""
    logger.info("Setting up corpus structure for analysis...")
    
    # Create necessary directories
    directories = [
        "extracted_txts_german",
        "parsed_results_german", 
        "results/german"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def main():
    """Main function to download and set up the German corpus."""
    logger.info("="*50)
    logger.info("POTSDAM COMMENTARY CORPUS DOWNLOAD SCRIPT")
    logger.info("="*50)
    
    try:
        # Step 1: Download corpus
        zip_path = download_corpus()
        
        # Step 2: Extract corpus  
        extract_corpus(zip_path)
        
        # Step 3: Find RST files
        rst_files = find_rst_files()
        
        # Step 4: Set up project structure
        setup_corpus_structure()
        
        # Step 5: Summary
        logger.info("\n" + "="*50)
        logger.info("DOWNLOAD COMPLETE!")
        logger.info("="*50)
        logger.info(f"German corpus location: {EXTRACT_DIR}")
        logger.info(f"RST files found: {len(rst_files)}")
        logger.info("\nNext steps:")
        logger.info("1. Run German EDU extraction: python main.py --languages german")
        logger.info("2. Check extracted_txts_german/ for EDU files")
        logger.info("3. Check parsed_results_german/ for dependency parsing results")
        
        # Clean up download
        if (DOWNLOAD_DIR / ZIP_FILENAME).exists():
            os.remove(DOWNLOAD_DIR / ZIP_FILENAME)
            logger.info(f"Cleaned up downloaded zip file")
            
    except Exception as e:
        logger.error(f"Error during corpus setup: {e}")
        raise

if __name__ == "__main__":
    main()
