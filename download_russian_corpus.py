#!/usr/bin/env python3
"""
Script to download and extract the Russian RST Treebank corpus.
This corpus contains Russian texts annotated with RST (Rhetorical Structure Theory) structure.

Source: https://rstreebank.ru/
License: CC BY-NC-SA 4.0

Citation:
    Pisarevskaya D. et al. (2017), Towards building a discourse-annotated corpus of Russian.
    In Computational Linguistics and Intellectual Technologies: Proc. of the Int. Conf. "Dialogue",
    Vol. 1, pp. 194-204.
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
CORPUS_URL = "https://api.rstreebank.ru/archive/RuRsTreebank_full.zip"
DOWNLOAD_DIR = Path("downloads")
EXTRACT_DIR = Path("RuRsTreebank_full")
ZIP_FILENAME = "RuRsTreebank_full.zip"


def download_corpus():
    """Download the Russian RST Treebank corpus zip file."""
    logger.info(f"Starting download of Russian RST Treebank from {CORPUS_URL}")
    
    # Create downloads directory
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    zip_path = DOWNLOAD_DIR / ZIP_FILENAME
    
    try:
        # Download with progress tracking
        response = requests.get(CORPUS_URL, stream=True, timeout=60)
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
    """Extract the corpus to the target directory."""
    logger.info(f"Extracting corpus from {zip_path}")
    
    # Remove existing extraction directory
    if EXTRACT_DIR.exists():
        logger.info(f"Removing existing directory: {EXTRACT_DIR}")
        shutil.rmtree(EXTRACT_DIR)
    
    # Extract zip file to temp directory first
    temp_dir = Path("temp_extract_russian")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Check if archive contains a root folder (RuRsTreebank_full/)
    extracted_contents = list(temp_dir.iterdir())
    
    if len(extracted_contents) == 1 and extracted_contents[0].is_dir():
        # Archive has a single root directory, move its contents
        root_dir = extracted_contents[0]
        shutil.move(str(root_dir), str(EXTRACT_DIR))
        logger.info(f"Corpus extracted to {EXTRACT_DIR}")
    else:
        # Archive has multiple items at root, move the entire temp dir
        shutil.move(str(temp_dir), str(EXTRACT_DIR))
        logger.info(f"Corpus extracted to {EXTRACT_DIR}")
    
    # Clean up temp directory if it still exists
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def find_rst_files():
    """Find and count RST files in the extracted corpus."""
    logger.info("Searching for RST files...")
    
    rst_files = list(EXTRACT_DIR.rglob("*.rs3"))
    txt_files = list(EXTRACT_DIR.rglob("*.txt"))
    
    if rst_files:
        logger.info(f"Found {len(rst_files)} RST (.rs3) files")
        logger.info(f"Found {len(txt_files)} text (.txt) files")
        
        # Group by subdirectory
        dirs = {}
        for f in rst_files:
            # Get relative path parts
            rel_path = f.relative_to(EXTRACT_DIR)
            if len(rel_path.parts) >= 2:
                subdir = rel_path.parts[0]
            else:
                subdir = "root"
            
            if subdir not in dirs:
                dirs[subdir] = []
            dirs[subdir].append(f.name)
        
        logger.info("\nCorpus structure:")
        for dir_name in sorted(dirs.keys()):
            files = dirs[dir_name]
            logger.info(f"  {dir_name}: {len(files)} RST files")
        
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
        "extracted_txts_russian",
        "parsed_results_russian",
        "results/russian",
        "results/results_russian/extracted_edus",
        "results/results_russian/parsed_dependencies",
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")


def show_corpus_info():
    """Display information about the downloaded corpus."""
    readme_path = EXTRACT_DIR / "README.txt"
    
    if readme_path.exists():
        logger.info("\n" + "="*50)
        logger.info("CORPUS README:")
        logger.info("="*50)
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
            for line in content.strip().split('\n'):
                logger.info(f"  {line}")


def main():
    """Main function to download and set up the Russian RST Treebank corpus."""
    logger.info("="*60)
    logger.info("RUSSIAN RST TREEBANK DOWNLOAD SCRIPT")
    logger.info("="*60)
    
    # Check if corpus already exists
    if EXTRACT_DIR.exists():
        rst_files = list(EXTRACT_DIR.rglob("*.rs3"))
        if rst_files:
            logger.info(f"Corpus already exists at {EXTRACT_DIR} with {len(rst_files)} RST files")
            response = input("Do you want to re-download? (y/N): ").strip().lower()
            if response != 'y':
                logger.info("Skipping download. Using existing corpus.")
                show_corpus_info()
                return
    
    try:
        # Step 1: Download corpus
        zip_path = download_corpus()
        
        # Step 2: Extract corpus
        extract_corpus(zip_path)
        
        # Step 3: Find RST files
        rst_files = find_rst_files()
        
        # Step 4: Show corpus info
        show_corpus_info()
        
        # Step 5: Set up project structure
        setup_corpus_structure()
        
        # Step 6: Summary
        logger.info("\n" + "="*60)
        logger.info("DOWNLOAD COMPLETE!")
        logger.info("="*60)
        logger.info(f"Russian corpus location: {EXTRACT_DIR}")
        logger.info(f"RST files found: {len(rst_files)}")
        logger.info("\nCorpus contains:")
        logger.info("  - news1/: News articles (set 1)")
        logger.info("  - news2/: News articles (set 2)")
        logger.info("  - blogs/: Blog posts")
        logger.info("  - sci.comp/: Scientific articles (Computer Science)")
        logger.info("  - sci.ling/: Scientific articles (Linguistics)")
        logger.info("\nNext steps:")
        logger.info("1. Run Russian EDU extraction: python main.py --languages russian")
        logger.info("2. Check extracted_txts_russian/ for EDU files")
        logger.info("3. Check parsed_results_russian/ for dependency parsing results")
        
        # Clean up download
        if zip_path.exists():
            os.remove(zip_path)
            logger.info(f"Cleaned up downloaded zip file")
        
        # Remove downloads directory if empty
        if DOWNLOAD_DIR.exists() and not any(DOWNLOAD_DIR.iterdir()):
            DOWNLOAD_DIR.rmdir()
            
    except Exception as e:
        logger.error(f"Error during corpus setup: {e}")
        raise


if __name__ == "__main__":
    main()

