# Dissolution Profile Comparison

Streamlit application for comparing dissolution profiles using f2 similarity factor and T2EQ (Hotelling's T²-based equivalence test).

## Application

### dissolution_f2_t2eq_bootstrap_streamlit.py
Bootstrap-based dissolution profile comparison with both f2 and T2EQ methods.

**Features:**
- **f2 similarity factor**: Classic FDA metric (acceptance: f2 ≥ 50)
- **E(f2)**: Bias-corrected expected f2 accounting for variability
- **Bootstrap confidence intervals**: Non-parametric CI for f2
- **T2EQ test**: Multivariate equivalence test using Mahalanobis distance
- Interactive parameter controls:
  - Sample sizes, time points, profile shapes
  - Variance levels, mean shifts
  - Bootstrap iterations
- Comprehensive visualizations:
  - Dissolution profiles with confidence bands
  - Bootstrap distributions
  - Statistical test results

**Run:**
```bash
streamlit run dissolution_f2_t2eq_bootstrap_streamlit.py
```

## Methods

### f2 Similarity Factor
```
f2 = 100 - 25 * log10(1 + MSD)
```
where MSD is mean squared difference between profiles.
- **Acceptance**: f2 ≥ 50 (FDA guideline)
- **Limitation**: Point estimate, doesn't account for variability

### E(f2) - Expected f2
Incorporates sample variance into similarity assessment.

### T2EQ Test
- Multivariate approach treating time points jointly
- Uses Hotelling's T² statistic and Mahalanobis distance
- More powerful for detecting differences across entire profile
- Accounts for correlation between time points

## Requirements

Install from root directory:
```bash
cd ..
pip install -r requirements.txt
```

## References

- FDA Guidance on dissolution testing (1997)
- Hoffelder, T. (2019). Equivalence analyses of dissolution profiles with the Mahalanobis distance
- Shah et al. (1998). In vitro dissolution profile comparison
