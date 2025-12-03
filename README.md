# Dependency Structures in Elementary Discourse Units

[![Tests & Type Checking](https://github.com/b-bblc/bachelor_thesis/actions/workflows/tests.yml/badge.svg)](https://github.com/b-bblc/bachelor_thesis/actions/workflows/tests.yml)

## Bachelor Thesis — University of Potsdam, Summer 2025

### Overview

This thesis investigates dependency structures in Elementary Discourse Units (EDUs) across multiple languages (German, Russian, English) using automatic dependency parsing and statistical analysis.

### Research Questions

1. **Primary**: Can automatic dependency parsing reveal a reasonably-small set of patterns or commonalities among EDUs in a corpus?
2. **Secondary**: To what extent are the resulting clusters language-specific vs. universal?

### Data Sources

| Language | Corpus | License |
|----------|--------|---------|
| German | [Potsdam Commentary Corpus](https://angcl.ling.uni-potsdam.de/) | See corpus documentation |
| Russian | [Ru-RSTreebank](https://rstreebank.ru/) | CC BY-NC-SA 4.0 |
| English (Test) | RST Discourse Treebank / GUM corpus | See corpus documentation |

---

## Requirements

- **Python**: 3.9, 3.10, or 3.11
- **Disk space**: ~2GB for spaCy language models
- **OS**: Linux, macOS, or Windows

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/b-bblc/bachelor_thesis.git
cd bachelor_thesis
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download spaCy language models

```bash
python -m spacy download de_core_news_md  # German
python -m spacy download en_core_web_sm   # English
python -m spacy download ru_core_news_sm  # Russian
```

---

## Downloading Corpora

The corpus data is not included in the repository due to size and licensing constraints.

### Russian RST Treebank

```bash
python download_russian_corpus.py
```

- **Source**: [rstreebank.ru](https://rstreebank.ru/)
- **Direct link**: https://api.rstreebank.ru/archive/RuRsTreebank_full.zip
- **License**: CC BY-NC-SA 4.0
- **Citation**: Pisarevskaya D. et al. (2017). Towards building a discourse-annotated corpus of Russian.

### Potsdam Commentary Corpus (German)

```bash
python download_german_corpus.py
```

- **Source**: [Potsdam University ANGCL](https://angcl.ling.uni-potsdam.de/)
- **License**: See corpus documentation

---

## Project Structure

```
bachelor_thesis/
├── .github/workflows/           # CI/CD configuration
│   └── tests.yml               # Tests & type checking workflow
├── src/                         # Python source modules
│   ├── __init__.py
│   ├── config.py               # Centralized configuration & logging
│   ├── analysis.py             # Dependency analysis & clustering
│   ├── boundary_detection.py   # EDU boundary feature extraction
│   ├── dependency_parser.py    # spaCy-based dependency parsing
│   ├── edu_extractor.py        # RST (.rs3) file processing
│   └── visualization.py        # Dependency tree visualizations
├── tests/                       # Unit tests (pytest)
│   ├── __init__.py
│   ├── conftest.py             # Shared pytest fixtures
│   └── test_boundary_detection.py
├── notebooks/                   # Jupyter notebooks for analysis
│   ├── comprehensive_multilingual_analysis.ipynb
│   ├── edu_boundary_detection.ipynb
│   ├── russian_edu_dependency_parse_en_with_markdown.ipynb
│   └── examples/               # Example notebooks
│       └── english_rst_parse_and_visualize_run_ready.ipynb
├── drafts/                      # Work-in-progress notebooks
├── results/                     # Analysis outputs
│   ├── results_german/         # German corpus results
│   │   ├── extracted_edus/     # Extracted EDU text files
│   │   └── parsed_dependencies/ # CoNLL-U parsed files
│   ├── results_russian/        # Russian corpus results
│   │   ├── extracted_edus/
│   │   └── parsed_dependencies/
│   ├── visualizations/         # Generated plots (PNG)
│   └── *.csv, *.json           # Analysis summaries
├── extracted_txts/              # Legacy: extracted EDUs (176 files)
├── parsed_results/              # Legacy: CoNLL-U parses (176 files)
├── PotsdamCommentaryCorpus/     # German RST corpus (not in repo)
├── RuRsTreebank_full/           # Russian RST corpus (not in repo)
├── docs/                        # Documentation
│   └── methodology.md
├── images/                      # Static images for docs
├── main.py                      # Main pipeline entry point
├── process_all_data.py          # Full data processing script
├── download_german_corpus.py    # Script to download German corpus
├── download_russian_corpus.py   # Script to download Russian corpus
├── requirements.txt             # Python dependencies
├── mypy.ini                     # Type checking configuration
├── setup.py                     # Package setup
├── LICENSE                      # Project license
└── README.md
```

---

## Modules

| Module | Description |
|--------|-------------|
| `src/config.py` | Centralized logging setup and project constants |
| `src/analysis.py` | `DependencyAnalyzer` class for feature extraction, clustering (KMeans), and visualization |
| `src/boundary_detection.py` | Functions for reading CoNLL-U files and extracting EDU boundary features |
| `src/dependency_parser.py` | `DependencyParser` class for parsing EDUs with spaCy and outputting CoNLL-U format |
| `src/edu_extractor.py` | `EDUExtractor` class for extracting EDUs from RST (.rs3) files |
| `src/visualization.py` | `DependencyVisualizer` class for creating dependency tree plots |

---

## Usage

### Quick Start

```bash
# Run the main pipeline
python main.py

# Or run individual steps:
python -m src.edu_extractor      # Extract EDUs from RST files
python -m src.dependency_parser  # Parse dependencies with spaCy
python -m src.analysis           # Analyze and cluster EDUs
```

### Python API Example

```python
from src.boundary_detection import read_conllu_with_boundaries, extract_edu_boundary_features

# Read CoNLL-U file
sentences = read_conllu_with_boundaries("parsed_results/sample.conllu")
print(f"Loaded {len(sentences)} sentences")

# Extract features for boundary detection
features_df = extract_edu_boundary_features(sentences, language="German")
print(features_df.head())
```

### Jupyter Notebooks

Interactive analysis is available in `notebooks/`:

- **`comprehensive_multilingual_analysis.ipynb`** — Full cross-linguistic analysis with clustering
- **`edu_boundary_detection.ipynb`** — EDU boundary detection experiments

---

## Testing

### Run all tests

```bash
pytest tests/ -v
```

### Run with coverage

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

### Type checking

```bash
mypy src/ --config-file mypy.ini
```

---

## CI/CD

This project uses GitHub Actions for continuous integration:

- **Tests**: Run on Python 3.9, 3.10, 3.11
- **Type checking**: mypy on Python 3.11
- **Trigger**: Every push and pull request

See [`.github/workflows/tests.yml`](.github/workflows/tests.yml) for details.

---

## Key Findings

*[To be completed after analysis]*

---

## References

- Mann, W. & Thompson, S. (1988). Rhetorical Structure Theory: Toward a functional theory of text organization. *Text*, 8(3), 243-281.
- Carlson, L., Marcu, D., & Okurowski, M. E. (2003). Building a discourse-tagged corpus in the framework of Rhetorical Structure Theory. *Current and new directions in discourse and dialogue*, 85-112.
- Pisarevskaya, D. et al. (2017). Towards building a discourse-annotated corpus of Russian. In *Computational Linguistics and Intellectual Technologies: Proc. of the Int. Conf. "Dialogue"*, Vol. 1, pp. 194-204.
- Shahmohammadi, S. & Stede, M. (2024). German RST Corpora.

---

## License

This project is developed as part of a Bachelor's thesis at the University of Potsdam.
The code is available for academic and research purposes.

## Author

Artur Begichev — University of Potsdam, 2025
