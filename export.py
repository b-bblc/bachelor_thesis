import os
import xml.etree.ElementTree as ET

# Path to the folder containing .rs3 files
rs3_folder = 'PotsdamCommentaryCorpus/rst'
# Path to the folder where .txt files will be saved
output_folder = 'extracted_txts'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate over all files in the .rs3 folder
for filename in os.listdir(rs3_folder):
    if filename.endswith('.rs3'):
        rs3_path = os.path.join(rs3_folder, filename)  # Full path to the .rs3 file
        tree = ET.parse(rs3_path)  # Parse the XML file
        root = tree.getroot()      # Get the root element of the XML

        # Extract all segments (EDUs) from the XML
        segments = root.findall('.//segment')

        # Collect the text content of each EDU, stripping whitespace
        edu_texts = [seg.text.strip() for seg in segments if seg.text]

        # Generate the output filename by replacing .rs3 with .txt
        txt_filename = filename.replace('.rs3', '.txt')
        txt_path = os.path.join(output_folder, txt_filename)

        # Write all EDU texts line by line into the .txt file
        with open(txt_path, 'w', encoding='utf-8') as f_out:
            for edu in edu_texts:
                f_out.write(edu + '\n')

        # Print a message indicating completion of the current file
        print(f'Processed file: {filename}, extracted EDU count: {len(edu_texts)}')



