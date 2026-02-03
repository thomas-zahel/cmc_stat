# Bayesian Weibull Dissolution Modeling

Streamlit application for Bayesian modeling and comparison of dissolution profiles using Weibull functions.

## Application

### dissolution_weibull_comparison_streamlit.py
Fits Weibull dissolution models to reference and test products using PyMC (Bayesian inference).

**Features:**
- **Weibull model**: f(t) = f_max * (1 - exp(-(t/td)^beta))
  - f_max: maximum dissolution (asymptote)
  - td: time scale parameter
  - beta: shape parameter
- **Bayesian inference** using MCMC (Markov Chain Monte Carlo)
- **Reference product fitting**: Establish expected dissolution curve
- **Test product comparison**: Compare against reference
- **Similarity metrics**:
  - RMSD (Root Mean Squared Difference) between fitted curves
  - Visual overlay of profiles with uncertainty bands
  - Parameter posterior distributions
- Interactive diagnostics:
  - Trace plots for MCMC convergence
  - Posterior distributions for all parameters
  - Predictive checks

**Run:**
```bash
streamlit run dissolution_weibull_comparison_streamlit.py
```

## Installation

This app requires additional Bayesian modeling dependencies:

```bash
pip install -r requirements.txt
```

This will install:
- pymc==5.18.2 (Bayesian modeling framework)
- arviz==0.20.0 (Bayesian visualization)
- pytensor==2.25.5 (computational backend)

⚠️ **Note**: These are large packages (~200MB) and may require C++ compilers. Installation can take several minutes.

## When to Use This Method

- **Mechanistic understanding**: Weibull parameters have physical meaning
- **Incomplete profiles**: Can extrapolate beyond measured time points
- **Small sample sizes**: Bayesian approach quantifies uncertainty
- **Non-linear profiles**: Handles complex dissolution kinetics
- **Parameter comparison**: Compare td, beta between products

## Advantages Over f2

- More interpretable parameters
- Handles any number of time points
- Quantifies uncertainty in similarity
- Can incorporate prior knowledge
- Better for mechanistic understanding

## References

- Weibull, W. (1951). A statistical distribution function
- Costa, P. & Lobo, J.M.S. (2001). Modeling and comparison of dissolution profiles
- Shah et al. (1998). In vitro dissolution profile comparison
