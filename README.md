# Dependency Structures in Elementary Discourse Units

[![Tests & Type Checking](https://github.com/b-bblc/bachelor_thesis/actions/workflows/tests.yml/badge.svg)](https://github.com/b-bblc/bachelor_thesis/actions/workflows/tests.yml)

## Bachelor Thesis - University of Potsdam, Summer 2025

### Overview
This thesis investigates dependency structures in Elementary Discourse Units (EDUs) across multiple languages (German, Russian, English) using automatic dependency parsing and statistical analysis.

### Research Questions
1. **Primary**: Can automatic dependency parsing reveal a reasonably-small set of patterns or commonalities among EDUs in a corpus?
2. **Secondary**: To what extent are the resulting clusters language-specific vs. universal?

### Data Sources
- **German**: Potsdam Commentary Corpus (RST-annotated)
- **Russian**: Ru-RSTreebank (Russian RST corpus)
- **English**: RST Discourse Treebank / GUM corpus examples

### Project Structure
```
├── data/                          # Raw data
│   ├── PotsdamCommentaryCorpus/  # German RST corpus
│   ├── RuRsTreebank_full/        # Russian RST corpus
│   └── en_example.rs3            # English RST example
├── extracted_txts/               # Extracted EDUs (German)
├── parsed_results/               # CoNLL-U dependency parses
├── notebooks/                    # Jupyter notebooks for analysis
├── src/                          # Source code modules
├── results/                      # Analysis outputs and visualizations
├── docs/                         # Documentation and literature
└── reports/                      # Thesis chapters and reports
```

### Installation & Setup
```bash
pip install -r requirements.txt
python -m spacy download de_core_news_md
python -m spacy download en_core_web_sm
python -m spacy download ru_core_news_sm
```

### Downloading Corpora

The corpus data is not included in the repository due to size and licensing constraints.
Use the following scripts to download the required corpora:

#### Russian RST Treebank
```bash
python download_russian_corpus.py
```
- **Source**: [rstreebank.ru](https://rstreebank.ru/)
- **Direct link**: https://api.rstreebank.ru/archive/RuRsTreebank_full.zip
- **License**: CC BY-NC-SA 4.0
- **Citation**: Pisarevskaya D. et al. (2017). Towards building a discourse-annotated corpus of Russian.

#### Potsdam Commentary Corpus (German)
```bash
python download_german_corpus.py
```
- **Source**: [Potsdam University ANGCL](https://angcl.ling.uni-potsdam.de/)
- **License**: See corpus documentation

### Usage
1. Extract EDUs: `python src/extract_edus.py`
2. Parse dependencies: `python src/parse_dependencies.py`
3. Run analysis: See notebooks in `notebooks/`

### Key Findings
[To be completed]

### References
- Mann, W. & Thompson, S. (1988). Rhetorical Structure Theory
- Carlson, L. et al. (2003). RST Discourse Treebank
- Shahmohammadi & Stede (2024). German RST Corpora
