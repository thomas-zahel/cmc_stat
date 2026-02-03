# Common Modules

Shared Python modules used by multiple applications.

## Modules

### bootstrap_biosimilarity_test.py
Core implementation of the bootstrap test for analytical biosimilarity (Zahel, 2022).

**Function:**
```python
biosimilarity_bootstrap_test(
    tp,                  # Test product batch data
    rp,                  # Reference product batch data  
    p_tp=0.99,          # Fraction of TP within RP range
    p_rp=0.99,          # Fraction of RP defining range
    n_boot=1000,        # Bootstrap resamples
    alpha=0.05,         # Significance level
    random_state=None   # RNG seed
)
```

**Returns:**
- p_value
- reject_null (biosimilar or not)
- rejection_rate
- test_statistic_observed

**Used by:**
- biosimilarity/biosimilarity_contour_comparison.py

### t2eq.py
Implementation of the T2EQ test (Hotelling's T²-based equivalence test) for dissolution profile similarity.

**Main Function:**
```python
t2eq_test(
    ref_profiles,        # Reference dissolution data (n x p)
    test_profiles,       # Test dissolution data (m x p)
    Delta=10.0,          # Equivalence margin (typically 10)
    alpha=0.05,          # Significance level
    method='hoffelder'   # Calculation method
)
```

**Returns:**
- T2EQResult with p_value, decision, statistics

**Features:**
- Handles unequal sample sizes
- Non-central F distribution
- Multiple calculation methods
- Fallback to mpmath for edge cases

**Used by:**
- dissolution/dissolution_f2_t2eq_bootstrap_streamlit.py

## Usage

These modules are automatically imported by the apps that need them. The import paths are configured to find this common directory from subdirectories.

## References

- Zahel, T. (2022). Bootstrap test for analytical biosimilarity
- Hoffelder, T. (2019). Equivalence analyses of dissolution profiles with the Mahalanobis distance
