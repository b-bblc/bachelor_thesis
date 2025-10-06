# Extracted Images from Jupyter Notebooks

This folder contains all PNG images extracted from the Jupyter notebooks in the `notebooks/` directory.

## Overview

**Total images extracted: 14**

### From `edu_boundary_detection.ipynb` (4 images):
- `edu_boundary_detection_cell_6_output_3.png`
- `edu_boundary_detection_cell_8_output_3.png`
- `edu_boundary_detection_cell_10_output_2.png`
- `edu_boundary_detection_cell_13_output_2.png`

### From `comprehensive_multilingual_analysis.ipynb` (10 images):
- `comprehensive_multilingual_analysis_cell_8_output_1.png`
- `comprehensive_multilingual_analysis_cell_14_output_2.png`
- `comprehensive_multilingual_analysis_cell_19_output_2.png`
- `comprehensive_multilingual_analysis_cell_21_output_1.png`
- `comprehensive_multilingual_analysis_cell_23_output_1.png`
- `comprehensive_multilingual_analysis_cell_25_output_2.png`
- `comprehensive_multilingual_analysis_cell_27_output_2.png`
- `comprehensive_multilingual_analysis_cell_27_output_4.png`
- `comprehensive_multilingual_analysis_cell_29_output_5.png`
- `comprehensive_multilingual_analysis_cell_31_output_2.png`

## File Naming Convention

The files are named using the following pattern:
`{notebook_name}_cell_{cell_number}_output_{output_number}.png`

Where:
- `notebook_name`: Name of the source notebook (without .ipynb extension)
- `cell_number`: Sequential number of the cell in the notebook (starting from 1)
- `output_number`: Sequential number of the output within that cell (starting from 1)

## Extraction Method

Images were extracted using the `extract_images.py` script which:
1. Parses all `.ipynb` files in the `notebooks/` directory
2. Extracts base64-encoded PNG data from cell outputs
3. Decodes and saves them as PNG files in this directory

---

*Generated automatically on 1 октября 2025 г.*