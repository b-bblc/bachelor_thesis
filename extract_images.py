#!/usr/bin/env python3
"""
Script to extract all PNG images from Jupyter notebooks and save them to the images folder.
"""

import json
import base64
import os
from pathlib import Path

def extract_images_from_notebook(notebook_path, output_dir):
    """
    Extract all PNG images from a Jupyter notebook and save them to output directory.
    
    Args:
        notebook_path (str): Path to the .ipynb file
        output_dir (str): Directory to save extracted images
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    image_count = 0
    notebook_name = Path(notebook_path).stem
    
    # Iterate through cells
    for cell_idx, cell in enumerate(notebook['cells']):
        if 'outputs' in cell:
            for output_idx, output in enumerate(cell['outputs']):
                if 'data' in output and 'image/png' in output['data']:
                    # Extract base64 encoded PNG data
                    png_data = output['data']['image/png']
                    
                    # Handle both string and list formats
                    if isinstance(png_data, list):
                        png_data = ''.join(png_data)
                    
                    # Decode base64 data
                    try:
                        image_bytes = base64.b64decode(png_data)
                        
                        # Generate filename
                        filename = f"{notebook_name}_cell_{cell_idx+1}_output_{output_idx+1}.png"
                        filepath = os.path.join(output_dir, filename)
                        
                        # Save image
                        with open(filepath, 'wb') as img_file:
                            img_file.write(image_bytes)
                        
                        image_count += 1
                        print(f"Extracted image: {filename}")
                        
                    except Exception as e:
                        print(f"Error extracting image from cell {cell_idx+1}, output {output_idx+1}: {e}")
    
    print(f"\nTotal images extracted: {image_count}")
    return image_count

def main():
    # Define paths
    notebooks_dir = "/Users/arturbegichev/University/bachelor_thesis/notebooks"
    output_dir = "/Users/arturbegichev/University/bachelor_thesis/images"
    
    print(f"Searching for notebooks in: {notebooks_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    # Find all notebook files
    notebook_files = []
    for file in os.listdir(notebooks_dir):
        if file.endswith('.ipynb'):
            notebook_files.append(os.path.join(notebooks_dir, file))
    
    if not notebook_files:
        print("No notebook files found!")
        return
    
    total_images = 0
    
    # Process each notebook
    for notebook_path in notebook_files:
        notebook_name = os.path.basename(notebook_path)
        print(f"\n📓 Processing: {notebook_name}")
        print("-" * 30)
        
        count = extract_images_from_notebook(notebook_path, output_dir)
        total_images += count
        
        if count > 0:
            print(f"✅ Extracted {count} images from {notebook_name}")
        else:
            print(f"⚠️  No images found in {notebook_name}")
    
    print(f"\n🎉 Total images extracted from all notebooks: {total_images}")
    print(f"📁 All images saved to: {output_dir}")

if __name__ == "__main__":
    main()