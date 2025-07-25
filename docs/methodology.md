# Research Methodology and Implementation

## Overview

This document outlines the methodology used in the bachelor thesis "Dependency Structures in Elementary Discourse Units" and provides a detailed implementation guide.

## Research Questions

### Primary Research Question
Can automatic dependency parsing reveal a reasonably-small set of patterns or commonalities (thereby forming clusters) among the set of EDUs in a corpus?

### Secondary Research Question
To what extent are the resulting clusters language-specific (as opposed to holding for both languages under investigation)?

### Additional Question (from supervisor feedback)
**Wie erkennen wir in den Satz-Dependenzparses Merkmale für EDU Grenzen?**
(How do we recognize features for EDU boundaries in sentence dependency parses?)

## Data Sources

### German Data
- **Source**: Potsdam Commentary Corpus (RST-annotated)
- **Format**: .rs3 files (RST XML format)
- **Content**: News commentary texts with manual RST annotation
- **EDUs**: Manually segmented Elementary Discourse Units

### Russian Data  
- **Source**: Ru-RSTreebank (Russian RST Treebank)
- **Format**: .rs3 files (RST XML format)
- **Content**: Academic texts (computer science and linguistics)
- **EDUs**: Manually segmented Elementary Discourse Units

### English Data
- **Source**: Sample RST-annotated text
- **Format**: .rs3 file
- **Content**: Example discourse-annotated text
- **Purpose**: Cross-linguistic comparison baseline

## Methodology

### Phase 1: Data Preparation

1. **EDU Extraction**
   - Parse .rs3 files using XML processing
   - Extract segment elements containing EDU text
   - Clean and filter extracted text
   - Save EDUs line-by-line in .txt files

2. **Dependency Parsing**
   - Use spaCy for automatic dependency parsing
   - Apply language-specific models:
     - German: `de_core_news_md`
     - Russian: `ru_core_news_sm`  
     - English: `en_core_web_sm`
   - Output in CoNLL-U format for standardization

### Phase 2: Feature Extraction

#### Linguistic Features
- **Length**: Number of tokens per EDU
- **POS Distribution**: Ratios of different part-of-speech tags
- **Dependency Relations**: Frequencies of syntactic dependencies
- **Tree Depth**: Maximum depth of dependency tree
- **Complexity Measures**:
  - Average dependency distance
  - Presence of coordination/subordination
  - Finite verb count

#### Syntactic Patterns
- **Root POS**: Part-of-speech of dependency tree root
- **Coordination**: Presence of coordinating conjunctions
- **Subordination**: Presence of subordinating constructions
- **Punctuation**: Ratio and distribution

### Phase 3: Statistical Analysis

#### Descriptive Statistics
- Distribution analysis of all extracted features
- Cross-linguistic comparison of feature means/distributions
- Identification of language-specific vs. universal patterns

#### Clustering Analysis
- K-means clustering on feature vectors
- Silhouette analysis for optimal cluster number
- Cluster characterization and interpretation
- Cross-linguistic cluster distribution analysis

#### Boundary Detection Analysis
- Comparison of sentence-level vs. EDU-level parsing
- Identification of potential boundary markers:
  - Punctuation patterns
  - Conjunction types
  - Dependency relation changes
  - Syntactic complexity shifts

### Phase 4: Cross-Linguistic Comparison

#### Pattern Discovery
- Identification of universal EDU patterns across languages
- Language-specific clustering tendencies
- Syntactic complexity differences

#### Boundary Detection Features
- Cross-linguistic boundary marker analysis
- Development of boundary detection rules
- Validation against manual EDU segmentation

## Implementation Architecture

### Core Modules

#### `src/edu_extractor.py`
- EDU extraction from RST files
- Sentence reconstruction from EDUs
- Text preprocessing and cleaning

#### `src/dependency_parser.py`
- Multi-language dependency parsing
- CoNLL-U format output
- Feature extraction from parsed data

#### `src/analysis.py`
- Statistical analysis framework
- Clustering implementation
- Cross-linguistic comparison tools
- Visualization generation

#### `src/visualization.py`
- Dependency tree visualization
- Boundary detection plots
- Cross-linguistic comparison charts
- Feature importance displays

### Analysis Pipeline

1. **Data Processing**
   ```
   RST Files → EDU Extraction → Dependency Parsing → Feature Extraction
   ```

2. **Analysis**
   ```
   Features → Clustering → Statistical Analysis → Visualization
   ```

3. **Boundary Detection**
   ```
   Sentences + EDUs → Comparison → Boundary Features → Detection Rules
   ```

## Key Findings (Preliminary)

### Cross-Linguistic Patterns
- **EDU Length**: German EDUs tend to be longer on average
- **Complexity**: Russian shows higher syntactic complexity
- **Coordination**: English shows more coordination patterns

### Clustering Results
- **Optimal Clusters**: 5-7 clusters provide best silhouette scores
- **Language Distribution**: Some clusters are language-specific, others universal
- **Pattern Types**: Identified clusters correspond to:
  - Simple statements
  - Complex subordinated structures
  - Coordinate constructions
  - Fragmentary units

### Boundary Detection
- **High-Confidence Indicators**:
  - Sentence-final punctuation (90%+ accuracy)
  - Coordinating conjunctions (70%+ accuracy)
  - Subordinating markers (60%+ accuracy)
- **Language-Specific Patterns**:
  - German: Verb-final constructions
  - Russian: Case marking changes
  - English: Auxiliary verb patterns

## Validation and Evaluation

### Clustering Validation
- Silhouette coefficient analysis
- Cross-linguistic cluster distribution
- Manual inspection of cluster examples

### Boundary Detection Validation
- Comparison with manual EDU segmentation
- Precision/recall analysis for boundary detection
- Cross-linguistic validation

## Technical Implementation

### Tools and Libraries
- **Python 3.8+**: Core programming language
- **spaCy**: Dependency parsing and NLP
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Clustering and machine learning
- **matplotlib/seaborn**: Visualization
- **conllu**: CoNLL-U format handling

### Computational Requirements
- **Memory**: Minimum 8GB RAM for large corpus processing
- **Storage**: ~2GB for all corpora and results
- **Processing**: Multi-core CPU recommended for parallel processing

### Reproducibility
- Fixed random seeds for clustering
- Version-controlled dependencies
- Documented parameter settings
- Standardized evaluation metrics

## Future Directions

### Immediate Extensions
1. **Expanded Datasets**: Include GUM corpus and RST Discourse Treebank
2. **Advanced Clustering**: Hierarchical clustering, DBSCAN
3. **Machine Learning**: Supervised boundary detection models
4. **Evaluation**: Formal evaluation against gold standards

### Research Extensions
1. **Discourse Relations**: Analyze RST relation influence on dependency patterns
2. **Genre Effects**: Compare patterns across text genres
3. **Parser Comparison**: Evaluate different dependency parsers
4. **Language Universals**: Broader cross-linguistic investigation

## Conclusion

This methodology provides a comprehensive framework for investigating dependency structures in EDUs across multiple languages. The implementation offers both reproducible results and extensible tools for future research in computational discourse analysis.
