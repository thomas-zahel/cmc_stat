# CMC Statistical Analysis - Interactive Applications

Educational Streamlit applications for CMC (Chemistry, Manufacturing, and Controls) statistical analyses, with focus on biosimilarity testing and dissolution profile comparisons.

## 📁 Repository Structure

```
cmc_stat/
├── biosimilarity/          # Biosimilarity testing apps
├── bootstrap_demos/        # Bootstrap CI/PI/TI demonstrations
├── dissolution/            # Standard dissolution testing (f2, T2EQ)
├── dissolution_weibull/    # Bayesian Weibull modeling (requires PyMC)
├── common/                 # Shared modules (bootstrap test, t2eq)
├── requirements.txt        # Core dependencies
└── requirements-weibull.txt # Additional Bayesian dependencies
```

## 🚀 Quick Start

### Installation

### Streamlit Community Cloud

This repo includes [runtime.txt](runtime.txt) to pin the Python version used by Streamlit Community Cloud (set to Python 3.12). This avoids slow/fragile source builds of scientific packages that can happen on newer Python versions.

Recommended Python: **3.12** (scientific packages ship wheels; Python 3.13 may trigger slow source builds).

#### Option A: pip + venv (most universal)

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### Option B: uv (fast, recommended)

Create a repo-local virtual environment:

```bash
uv venv --python 3.12
```

Install into that environment explicitly (avoids accidentally targeting some other venv):

```bash
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

### Running Applications

Navigate to any folder and run the app:
```bash
cd biosimilarity
streamlit run 3SD_test_demo_streamlit.py
```

Or run via the repo-local venv without activating:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\biosimilarity\3SD_test_demo_streamlit.py
```

## 📦 Applications by Category

### Biosimilarity Testing ([biosimilarity/](biosimilarity/))
- **3SD_test_demo_streamlit.py** - Interactive 3SD range test demo
- **biosimilarity_contour_comparison.py** - Compare 4 biosimilarity tests

### Bootstrap Methods ([bootstrap_demos/](bootstrap_demos/))
- **bootstrap_intervals_demo_streamlit.py** - CI/PI/TI for non-normal distributions
- **bootstrap_skewed_demo_streamlit.py** - Bootstrap CI for skewed data

### Dissolution Testing ([dissolution/](dissolution/))
- **dissolution_f2_t2eq_bootstrap_streamlit.py** - f2 and T2EQ comparison

### Bayesian Weibull Analysis ([dissolution_weibull/](dissolution_weibull/))
- **dissolution_weibull_comparison_streamlit.py** - Weibull modeling with PyMC
  - ⚠️ **Requires additional installation**: `cd dissolution_weibull && pip install -r requirements.txt`

## 📚 Core Dependencies

See [requirements.txt](requirements.txt) for the authoritative version ranges.

## ⚙️ Optional Dependencies (Weibull App Only)

The Bayesian Weibull app requires large dependencies (~200MB):
- pymc (<6)
- arviz (<1)

Install only if you need the Weibull analysis:
```bash
cd dissolution_weibull
pip install -r requirements.txt
```

## 🧪 Testing

Verify all dependencies are installed:
```bash
python test_dependencies.py
```

## 📖 Documentation

Each folder contains its own README with:
- Detailed app descriptions
- Feature lists
- Usage examples
- Method explanations
- References

## 🎯 Use Cases

- **Analytical Biosimilarity**: Demonstrate similarity between test and reference products
- **Method Validation**: Bootstrap intervals for small samples or non-normal data
- **Dissolution Testing**: Compare dissolution profiles using f2, T2EQ, or Weibull models
- **Statistical Education**: Interactive tools for understanding CMC statistics

## 📄 License

See [LICENSE](LICENSE) file for details.

## 🔗 References

- Zahel, T. (2022). Bootstrap test for analytical biosimilarity
- Hoffelder, T. (2019). Equivalence analyses of dissolution profiles with the Mahalanobis distance
- FDA Guidance on dissolution testing and analytical similarity

## 👤 Author

Educational examples for CMC statistics
