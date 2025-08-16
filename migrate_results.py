#!/usr/bin/env python3
"""
Migration script to move existing results to the new structure.
This script moves data from the old structure to the new results/ directory.
"""

import shutil
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def migrate_results():
    """Migrate existing results to new directory structure."""
    logger.info("🔄 Starting results migration...")
    
    # Define old and new paths
    migrations = [
        {
            'old': 'extracted_txts_german',
            'new': 'results/results_german/extracted_edus',
            'description': 'German extracted EDUs'
        },
        {
            'old': 'extracted_txts_russian', 
            'new': 'results/results_russian/extracted_edus',
            'description': 'Russian extracted EDUs'
        },
        {
            'old': 'parsed_results_german',
            'new': 'results/results_german/parsed_dependencies',
            'description': 'German parsed dependencies'
        },
        {
            'old': 'parsed_results_russian',
            'new': 'results/results_russian/parsed_dependencies', 
            'description': 'Russian parsed dependencies'
        }
    ]
    
    # Create new directory structure
    Path('results/results_german').mkdir(exist_ok=True, parents=True)
    Path('results/results_russian').mkdir(exist_ok=True, parents=True)
    
    # Perform migrations
    for migration in migrations:
        old_path = Path(migration['old'])
        new_path = Path(migration['new'])
        
        if old_path.exists():
            if new_path.exists():
                logger.warning(f"⚠️  Target directory {new_path} already exists, skipping {migration['description']}")
                continue
                
            logger.info(f"📦 Moving {migration['description']} from {old_path} to {new_path}")
            
            try:
                # Create parent directory if needed
                new_path.parent.mkdir(exist_ok=True, parents=True)
                
                # Move the directory
                shutil.move(str(old_path), str(new_path))
                logger.info(f"✅ Successfully moved {migration['description']}")
                
            except Exception as e:
                logger.error(f"❌ Failed to move {migration['description']}: {e}")
        else:
            logger.info(f"ℹ️  {migration['description']} not found at {old_path}, skipping")
    
    logger.info("🎉 Migration completed!")
    
    # Show new structure
    logger.info("\n📁 New directory structure:")
    for path in ['results/results_german', 'results/results_russian']:
        path_obj = Path(path)
        if path_obj.exists():
            logger.info(f"  {path}/")
            for subdir in path_obj.iterdir():
                if subdir.is_dir():
                    file_count = len(list(subdir.glob('*')))
                    logger.info(f"    {subdir.name}/ ({file_count} files)")

def main():
    """Main function."""
    migrate_results()

if __name__ == "__main__":
    main()
