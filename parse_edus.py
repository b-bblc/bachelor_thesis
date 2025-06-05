import spacy
import os

# Load the German language model
nlp = spacy.load('de_core_news_md')

# Directories
input_folder = 'extracted_txts'  
output_folder = 'parsed_results'  

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Loop through each .txt file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.txt'):  
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace('.txt', '_parsed.conllu'))

        with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
            edus = infile.readlines()  
            
           
            for edu_num, edu in enumerate(edus, 1):
                edu = edu.strip()  

                if not edu:  
                    continue

                # Process the EDU using spaCy
                doc = nlp(edu)

                # Write the result in CoNLL-U format
                outfile.write(f"# sent_id = {edu_num}\n")  # Sentence ID
                outfile.write(f"# text = {edu}\n")  # Text (EDU)

                # For each token in the EDU, write its details (ID, text, lemma, POS, head, dependency relation, etc.)
                for i, token in enumerate(doc, 1):
                    outfile.write(f"{i}\t{token.text}\t{token.lemma_}\t{token.pos_}\t_\t{token.morph}\t{token.head.i + 1}\t{token.dep_}\t_\t_\n")
                outfile.write("\n")  

        print(f"Parsed file: {filename}")  
