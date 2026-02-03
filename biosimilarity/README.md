# Biosimilarity Testing Applications

Interactive Streamlit applications for analytical biosimilarity assessment.

## Applications

### 3SD_test_demo_streamlit.py
Simple range test for analytical biosimilarity based on the 3SD criterion.

**Features:**
- Interactive simulation of reference and biosimilar products
- Demonstrates two test criteria:
  - Criterion 1: X% of biosimilar within Y SD of reference
  - Criterion 2: Biosimilar mean ±3SD fully contained within reference range
- Real-time parameter adjustment
- Power analysis simulation

**Run:**
```bash
streamlit run 3SD_test_demo_streamlit.py
```

### biosimilarity_contour_comparison.py
Comprehensive comparison of 4 different biosimilarity tests with contour plot analysis.

**Features:**
- Compares 3SD, TOST, Bootstrap, and combined tests
- Contour plots showing acceptance rates across parameter space
- Interactive parameter controls
- Visual comparison of test performance

**Run:**
```bash
streamlit run biosimilarity_contour_comparison.py
```

## Requirements

Install from root directory:
```bash
cd ..
pip install -r requirements.txt
```

## References

- Zahel, T. (2022). Bootstrap test for analytical biosimilarity
- FDA Guidance on analytical similarity
