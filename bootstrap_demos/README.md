# Bootstrap Confidence Interval Demonstrations

Interactive Streamlit applications demonstrating bootstrap methods for confidence, prediction, and tolerance intervals with non-normal distributions.

## Applications

### bootstrap_intervals_demo_streamlit.py
Comprehensive demo of bootstrap CI/PI/TI for various distributions.

**Features:**
- Multiple distribution types (Normal, Log-Normal, Exponential, Gamma, Chi-Square, Beta)
- Visualizes three interval types:
  - **Confidence Interval (CI)**: Where the population mean likely is
  - **Prediction Interval (PI)**: Where a new observation will likely fall
  - **Tolerance Interval (TI)**: Contains X% of the population with Y% confidence
- Bootstrap vs parametric comparison
- Interactive parameter controls

**Run:**
```bash
streamlit run bootstrap_intervals_demo_streamlit.py
```

### bootstrap_skewed_demo_streamlit.py
Focused demo on bootstrap confidence intervals for skewed distributions.

**Features:**
- Six distribution types with adjustable parameters
- Bootstrap percentile, BCa, and basic methods
- Comparison with parametric intervals
- Coverage probability simulation
- Visual histogram with overlay of true distribution

**Run:**
```bash
streamlit run bootstrap_skewed_demo_streamlit.py
```

## Requirements

Install from root directory:
```bash
cd ..
pip install -r requirements.txt
```

## Key Concepts

- **Bootstrap**: Resampling method that doesn't assume normality
- **BCa**: Bias-corrected and accelerated bootstrap (most accurate for skewed data)
- **Percentile Method**: Simple quantiles of bootstrap distribution
- **Coverage**: How often the interval contains the true parameter

## Use Cases

- Small sample sizes
- Skewed or non-normal data
- When parametric assumptions are violated
- Validating analytical methods in CMC
