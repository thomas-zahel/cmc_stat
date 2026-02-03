# Bootstrap Confidence Interval Demo - Non-Normal Distributions

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from scipy import stats

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.branding import BrandingConfig, apply_branding

apply_branding(
    BrandingConfig(
        app_title="Bootstrap CI Demo (Skewed)",
        header_title="Bootstrap Confidence Intervals",
        header_subtitle="Exploring skewed and non-normal distributions",
    )
)

# Set random seed for reproducibility in session
if 'random_seed' not in st.session_state:
    st.session_state.random_seed = 42

# Function to generate samples from different distributions
def generate_sample(dist_type, sample_size, **params):
    """Generate sample from specified distribution"""
    if dist_type == "Normal":
        return np.random.normal(params['mean'], params['std'], sample_size)
    elif dist_type == "Log-Normal":
        return np.random.lognormal(params['mean'], params['std'], sample_size)
    elif dist_type == "Exponential":
        return np.random.exponential(params['scale'], sample_size)
    elif dist_type == "Gamma":
        return np.random.gamma(params['shape'], params['scale'], sample_size)
    elif dist_type == "Chi-Square":
        return np.random.chisquare(params['df'], sample_size)
    elif dist_type == "Beta (Skewed)":
        return np.random.beta(params['alpha'], params['beta'], sample_size)

# Function to get distribution properties
def get_distribution_properties(dist_type, **params):
    """Get mean, mode, and pdf function for the distribution"""
    if dist_type == "Normal":
        mean = params['mean']
        mode = params['mean']
        std = params['std']
        pdf = lambda x: stats.norm.pdf(x, mean, std)
        x_range = (mean - 4*std, mean + 4*std)
        
    elif dist_type == "Log-Normal":
        # For lognormal: if X ~ LogNormal(μ, σ), then ln(X) ~ Normal(μ, σ)
        mu = params['mean']
        sigma = params['std']
        mean = np.exp(mu + sigma**2 / 2)
        mode = np.exp(mu - sigma**2)
        pdf = lambda x: stats.lognorm.pdf(x, sigma, scale=np.exp(mu))
        x_range = (0, mean + 4*np.sqrt((np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2)))
        
    elif dist_type == "Exponential":
        scale = params['scale']
        mean = scale
        mode = 0
        pdf = lambda x: stats.expon.pdf(x, scale=scale)
        x_range = (0, mean * 4)
        
    elif dist_type == "Gamma":
        shape = params['shape']
        scale = params['scale']
        mean = shape * scale
        mode = (shape - 1) * scale if shape >= 1 else 0
        pdf = lambda x: stats.gamma.pdf(x, shape, scale=scale)
        x_range = (0, mean + 4*np.sqrt(shape * scale**2))
        
    elif dist_type == "Chi-Square":
        df = params['df']
        mean = df
        mode = max(df - 2, 0)
        pdf = lambda x: stats.chi2.pdf(x, df)
        x_range = (0, mean + 4*np.sqrt(2*df))
        
    elif dist_type == "Beta (Skewed)":
        alpha = params['alpha']
        beta = params['beta']
        mean = alpha / (alpha + beta)
        if alpha > 1 and beta > 1:
            mode = (alpha - 1) / (alpha + beta - 2)
        else:
            mode = 0 if alpha <= 1 else 1
        pdf = lambda x: stats.beta.pdf(x, alpha, beta)
        x_range = (0, 1)
    
    return mean, mode, pdf, x_range

# Function to calculate analytical confidence interval
def analytical_ci(sample, confidence=0.95):
    """Calculate confidence interval using analytical formula (t-distribution)"""
    n = len(sample)
    mean = np.mean(sample)
    se = stats.sem(sample)  # Standard error
    ci = stats.t.interval(confidence, n-1, loc=mean, scale=se)
    return mean, ci

# Function to calculate bootstrap confidence interval
def bootstrap_ci(sample, n_bootstrap=1000, confidence=0.95):
    """Calculate confidence interval using bootstrap resampling"""
    bootstrap_means = []
    bootstrap_samples = []
    
    for i in range(n_bootstrap):
        # Resample with replacement
        resampled = np.random.choice(sample, size=len(sample), replace=True)
        bootstrap_means.append(np.mean(resampled))
        # Store first 3 bootstrap samples for display
        if i < 3:
            bootstrap_samples.append(resampled)
    
    # Calculate confidence interval from bootstrap distribution
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha/2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha/2))
    
    return np.mean(bootstrap_means), (lower, upper), bootstrap_samples, bootstrap_means


def bca_interval(sample, bootstrap_stats, stat_hat, confidence=0.95):
    """Compute bias-corrected and accelerated (BCa) CI for a statistic."""
    alpha = 1 - confidence

    # Bias correction z0
    prop_less = np.mean(np.array(bootstrap_stats) < stat_hat)
    # Avoid p=0 or p=1 which would give inf
    prop_less = min(max(prop_less, 1e-6), 1 - 1e-6)
    z0 = stats.norm.ppf(prop_less)

    # Acceleration a using jackknife
    n = len(sample)
    if n <= 2:
        a = 0.0
    else:
        # Leave-one-out means can be computed vectorized for the mean statistic
        total = np.sum(sample)
        jackknife_stats = (total - sample) / (n - 1)
        mean_jack = np.mean(jackknife_stats)
        num = np.sum((mean_jack - jackknife_stats) ** 3)
        den = 6 * (np.sum((mean_jack - jackknife_stats) ** 2) ** 1.5)
        a = num / den if den != 0 else 0.0

    def adjusted_percentile(prob):
        z = stats.norm.ppf(prob)
        adj = stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        return np.clip(adj, 0, 1)

    lower_pct = adjusted_percentile(alpha / 2) * 100
    upper_pct = adjusted_percentile(1 - alpha / 2) * 100

    lower = np.percentile(bootstrap_stats, lower_pct)
    upper = np.percentile(bootstrap_stats, upper_pct)

    return lower, upper


# Streamlit App
st.title("Bootstrap CI Demo: Normal vs Non-Normal Distributions")
st.markdown("---")

# Custom CSS to increase sidebar font size
st.markdown("""
    <style>
    section[data-testid="stSidebar"] * {
        font-size: 18px !important;
    }
    section[data-testid="stSidebar"] h2 {
        font-size: 24px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.header("Parameters")
    
    st.subheader("Distribution Type")
    dist_type = st.selectbox(
        "Select Distribution:",
        ["Normal", "Log-Normal", "Exponential", "Gamma", "Chi-Square", "Beta (Skewed)"]
    )
    
    st.markdown("---")
    st.subheader("Distribution Parameters")
    
    # Different parameters for different distributions
    if dist_type == "Normal":
        param1 = st.slider("Mean (μ):", -5.0, 5.0, 0.0, 0.1)
        param2 = st.slider("Std Dev (σ):", 0.1, 5.0, 1.0, 0.1)
        dist_params = {'mean': param1, 'std': param2}
        
    elif dist_type == "Log-Normal":
        param1 = st.slider("Log Mean (μ):", -2.0, 2.0, 0.0, 0.1)
        param2 = st.slider("Log Std Dev (σ):", 0.1, 2.0, 0.5, 0.1)
        dist_params = {'mean': param1, 'std': param2}
        
    elif dist_type == "Exponential":
        param1 = st.slider("Scale (λ⁻¹):", 0.1, 5.0, 1.0, 0.1)
        dist_params = {'scale': param1}
        
    elif dist_type == "Gamma":
        param1 = st.slider("Shape (k):", 0.5, 10.0, 2.0, 0.5)
        param2 = st.slider("Scale (θ):", 0.1, 5.0, 1.0, 0.1)
        dist_params = {'shape': param1, 'scale': param2}
        
    elif dist_type == "Chi-Square":
        param1 = st.slider("Degrees of Freedom:", 1, 20, 5, 1)
        dist_params = {'df': param1}
        
    elif dist_type == "Beta (Skewed)":
        param1 = st.slider("Alpha (α):", 0.5, 10.0, 2.0, 0.5)
        param2 = st.slider("Beta (β):", 0.5, 10.0, 5.0, 0.5)
        dist_params = {'alpha': param1, 'beta': param2}
    
    st.markdown("---")
    sample_size = st.slider(
        "Sample Size:",
        min_value=4,
        max_value=100,
        value=7,
        step=1
    )
    
    st.markdown("---")
    st.subheader("Bootstrap Settings")
    
    n_bootstrap = st.slider(
        "Number of Bootstrap Samples:",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100
    )

    compare_bca = st.checkbox(
        "Check BCa (bias-corrected) interval",
        value=False,
        help="Compute bias-corrected and accelerated CI and flag when it differs from the basic percentile bootstrap."
    )
    
    confidence_level = st.slider(
        "Confidence Level:",
        min_value=0.50,
        max_value=0.99,
        value=0.95,
        step=0.01,
        format="%.2f"
    )
    
    st.markdown("---")
    resample_button = st.button("Draw New Sample", type="secondary", use_container_width=True)
    bootstrap_button = st.button("Run Bootstrap", type="primary", use_container_width=True)

# Initialize or resample
if 'sample' not in st.session_state or resample_button:
    np.random.seed(st.session_state.random_seed)
    st.session_state.sample = generate_sample(dist_type, sample_size, **dist_params)
    st.session_state.sample_mean = np.mean(st.session_state.sample)
    st.session_state.random_seed += 1
    st.session_state.bootstrap_done = False

# Run bootstrap analysis
if bootstrap_button:
    st.session_state.bootstrap_done = True
    
    # Calculate analytical CI
    sample_mean, analytical_interval = analytical_ci(st.session_state.sample, confidence_level)
    
    # Calculate bootstrap CI
    bootstrap_mean, bootstrap_interval, example_samples, all_bootstrap_means = bootstrap_ci(
        st.session_state.sample, n_bootstrap, confidence_level
    )

    bootstrap_bca_interval = None
    bca_differs = None
    if compare_bca:
        bootstrap_bca_interval = bca_interval(
            st.session_state.sample,
            all_bootstrap_means,
            st.session_state.sample_mean,
            confidence_level
        )
        tol = 1e-8
        bca_differs = (
            abs(bootstrap_bca_interval[0] - bootstrap_interval[0]) > tol
            or abs(bootstrap_bca_interval[1] - bootstrap_interval[1]) > tol
        )
    
    # Store results in session state
    st.session_state.sample_mean = sample_mean
    st.session_state.analytical_ci = analytical_interval
    st.session_state.bootstrap_mean = bootstrap_mean
    st.session_state.bootstrap_ci = bootstrap_interval
    st.session_state.bootstrap_bca_ci = bootstrap_bca_interval
    st.session_state.bca_differs = bca_differs
    st.session_state.example_samples = example_samples
    st.session_state.all_bootstrap_means = all_bootstrap_means

# Get distribution properties
true_mean, true_mode, pdf, x_range = get_distribution_properties(dist_type, **dist_params)

# Display Ground Truth Distribution
st.subheader("Ground Truth Population Distribution")

# Info box explaining the concept
if dist_type != "Normal":
    st.info(f"""
    **Important Concept**: For skewed distributions, the **mean ≠ mode**.
    - True Population Mean: {true_mean:.3f} (what we're estimating)
    - True Population Mode: {true_mode:.3f} (most frequent value)
    
    **Confidence Intervals estimate the MEAN**, not the mode! Both analytical and bootstrap CIs should cover the true mean.
    """)

fig1, ax1 = plt.subplots(figsize=(12, 6))

# Plot population distribution
x = np.linspace(x_range[0], x_range[1], 1000)
if x_range[0] >= 0:  # For distributions defined on [0, ∞) or [0, 1]
    x = x[x >= 0]
y = pdf(x)

ax1.fill_between(x, y, alpha=0.3, color='gray', label='True Population Distribution')
ax1.plot(x, y, 'k-', linewidth=2)

# Add true mean and mode
ax1.axvline(true_mean, color='black', linestyle='-', linewidth=3, 
            label=f'True Mean (μ={true_mean:.3f}) ← TARGET')
if abs(true_mean - true_mode) > 0.01:  # Only show mode if different from mean
    ax1.axvline(true_mode, color='gray', linestyle=':', linewidth=2, 
                label=f'True Mode ({true_mode:.3f})')

# Add sample mean
if 'sample' in st.session_state:
    ax1.axvline(st.session_state.sample_mean, color='red', linestyle='--', 
                linewidth=2, label=f'Sample Mean ({st.session_state.sample_mean:.2f})')

# Add confidence intervals if bootstrap was run
if st.session_state.get('bootstrap_done', False):
    y_level = 0.05 * max(y)
    
    # Analytical CI (green)
    ax1.plot(st.session_state.analytical_ci, [y_level, y_level], 
             color='green', linewidth=5, marker='|', markersize=20, 
             label=f'Analytical CI [{st.session_state.analytical_ci[0]:.2f}, {st.session_state.analytical_ci[1]:.2f}]')
    
    # Bootstrap CI (blue)
    ax1.plot(st.session_state.bootstrap_ci, [2*y_level, 2*y_level], 
             color='blue', linewidth=5, marker='|', markersize=20,
             label=f'Bootstrap CI [{st.session_state.bootstrap_ci[0]:.2f}, {st.session_state.bootstrap_ci[1]:.2f}]')

    # BCa CI (purple) when requested
    if st.session_state.get('bootstrap_bca_ci'):
        ax1.plot(st.session_state.bootstrap_bca_ci, [3*y_level, 3*y_level],
                 color='purple', linewidth=5, marker='|', markersize=20,
                 label=f'BCa Bootstrap CI [{st.session_state.bootstrap_bca_ci[0]:.2f}, {st.session_state.bootstrap_bca_ci[1]:.2f}]')
    
    # Check coverage
    analytical_covers = st.session_state.analytical_ci[0] <= true_mean <= st.session_state.analytical_ci[1]
    bootstrap_covers = st.session_state.bootstrap_ci[0] <= true_mean <= st.session_state.bootstrap_ci[1]
    bca_covers = None
    if st.session_state.get('bootstrap_bca_ci'):
        bca_covers = st.session_state.bootstrap_bca_ci[0] <= true_mean <= st.session_state.bootstrap_bca_ci[1]
    
    # Add coverage indicators
    coverage_text = f"\n\n✓ Analytical CI {'COVERS' if analytical_covers else 'MISSES'} true mean"
    coverage_text += f"\n✓ Bootstrap CI {'COVERS' if bootstrap_covers else 'MISSES'} true mean"
    if bca_covers is not None:
        coverage_text += f"\n✓ BCa Bootstrap CI {'COVERS' if bca_covers else 'MISSES'} true mean"
    
    ax1.text(0.02, 0.98, coverage_text, transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=12, family='monospace')

ax1.set_xlabel('Value', fontsize=12)
ax1.set_ylabel('Probability Density', fontsize=12)
ax1.set_title(f'Population Distribution: {dist_type}', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)

st.pyplot(fig1)
plt.close()

# Display Sample Data
st.markdown("---")
st.subheader("Drawn Sample")

if 'sample' in st.session_state:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sample Size", len(st.session_state.sample))
    with col2:
        st.metric("Sample Mean", f"{np.mean(st.session_state.sample):.3f}")
    with col3:
        st.metric("Sample Median", f"{np.median(st.session_state.sample):.3f}")
    with col4:
        st.metric("Sample Std Dev", f"{np.std(st.session_state.sample, ddof=1):.3f}")
    
    # Display sample in a scrollable dataframe
    sample_df = pd.DataFrame({
        'Index': range(1, len(st.session_state.sample) + 1),
        'Value': st.session_state.sample
    })
    st.dataframe(sample_df, use_container_width=True, height=200)

# Display Bootstrap Results
if st.session_state.get('bootstrap_done', False):
    # Display example bootstrap samples in tabs
    st.markdown("---")
    st.subheader("Example Bootstrap Samples")
    st.markdown(f"*Showing 3 out of {n_bootstrap} bootstrap samples*")
    
    tab1, tab2, tab3 = st.tabs(["Bootstrap Sample 1", "Bootstrap Sample 2", "Bootstrap Sample 3"])
    
    for idx, (tab, example_sample) in enumerate(zip([tab1, tab2, tab3], st.session_state.example_samples)):
        with tab:
            boot_mean = np.mean(example_sample)
            boot_std = np.std(example_sample, ddof=1)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Bootstrap Mean", f"{boot_mean:.4f}")
            with col_b:
                st.metric("Bootstrap Std Dev", f"{boot_std:.4f}")
            
            # Create dataframe for bootstrap sample
            boot_df = pd.DataFrame({
                'Index': range(1, len(example_sample) + 1),
                'Value': example_sample
            })
            st.dataframe(boot_df, use_container_width=True, height=200)
    
    st.markdown("---")
    st.subheader("Bootstrap Analysis Results")
    
    # Check coverage
    analytical_covers = st.session_state.analytical_ci[0] <= true_mean <= st.session_state.analytical_ci[1]
    bootstrap_covers = st.session_state.bootstrap_ci[0] <= true_mean <= st.session_state.bootstrap_ci[1]
    
    # Display confidence intervals comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Analytical Method (t-distribution)**")
        st.markdown("*Assumes normality*")
        st.metric("Sample Mean", f"{st.session_state.sample_mean:.4f}")
        st.metric(f"{int(confidence_level*100)}% Confidence Interval", 
                 f"[{st.session_state.analytical_ci[0]:.4f}, {st.session_state.analytical_ci[1]:.4f}]")
        st.metric("CI Width", f"{st.session_state.analytical_ci[1] - st.session_state.analytical_ci[0]:.4f}")
        
        if analytical_covers:
            st.success(f"✓ COVERS true mean ({true_mean:.4f})")
        else:
            st.error(f"✗ MISSES true mean ({true_mean:.4f})")
    
    with col2:
        st.markdown("**Bootstrap Method**")
        st.markdown("*No distributional assumptions*")
        st.metric("Bootstrap Mean", f"{st.session_state.bootstrap_mean:.4f}")
        st.metric(f"{int(confidence_level*100)}% Confidence Interval", 
                 f"[{st.session_state.bootstrap_ci[0]:.4f}, {st.session_state.bootstrap_ci[1]:.4f}]")
        st.metric("CI Width", f"{st.session_state.bootstrap_ci[1] - st.session_state.bootstrap_ci[0]:.4f}")
        
        if bootstrap_covers:
            st.success(f"✓ COVERS true mean ({true_mean:.4f})")
        else:
            st.error(f"✗ MISSES true mean ({true_mean:.4f})")

        if st.session_state.get('bootstrap_bca_ci'):
            bca_low, bca_high = st.session_state.bootstrap_bca_ci
            bca_covers = bca_low <= true_mean <= bca_high
            st.markdown("---")
            st.markdown("**BCa (bias-corrected & accelerated)**")
            st.metric(f"{int(confidence_level*100)}% BCa CI", f"[{bca_low:.4f}, {bca_high:.4f}]")
            st.metric("BCa CI Width", f"{bca_high - bca_low:.4f}")
            if bca_covers:
                st.success(f"✓ BCa CI covers true mean ({true_mean:.4f})")
            else:
                st.error(f"✗ BCa CI misses true mean ({true_mean:.4f})")
            if st.session_state.bca_differs:
                st.warning("BCa interval differs from the percentile bootstrap (bias/acceleration detected).")
            else:
                st.info("BCa interval matches the percentile bootstrap (little bias/acceleration detected).")
    
    
    # Display bootstrap distribution
    st.markdown("---")
    st.subheader("Bootstrap Sampling Distribution of the Mean")
    
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    # Histogram of bootstrap means
    ax2.hist(st.session_state.all_bootstrap_means, bins=50, density=True, 
             alpha=0.7, color='skyblue', edgecolor='black', label='Bootstrap Distribution')
    
    # Add vertical lines
    ax2.axvline(st.session_state.bootstrap_mean, color='blue', linestyle='-', 
                linewidth=2, label=f'Bootstrap Mean ({st.session_state.bootstrap_mean:.3f})')
    ax2.axvline(st.session_state.sample_mean, color='red', linestyle='--', 
                linewidth=2, label=f'Original Sample Mean ({st.session_state.sample_mean:.3f})')
    ax2.axvline(true_mean, color='black', linestyle='-', linewidth=3, 
                label=f'True Population Mean ({true_mean:.3f})')
    
    # Add confidence interval shading
    ax2.axvspan(st.session_state.bootstrap_ci[0], st.session_state.bootstrap_ci[1], 
                alpha=0.2, color='blue', label=f'{int(confidence_level*100)}% Bootstrap CI')

    if st.session_state.get('bootstrap_bca_ci'):
        ax2.axvspan(st.session_state.bootstrap_bca_ci[0], st.session_state.bootstrap_bca_ci[1],
                    alpha=0.15, color='purple', label=f'{int(confidence_level*100)}% BCa Bootstrap CI')
    
    ax2.set_xlabel('Mean Value', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title(f'Distribution of Bootstrap Means (n={n_bootstrap})', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig2)
    plt.close()
    

else:
    pass #st.info("👈 Click **Run Bootstrap** in the sidebar to perform bootstrap analysis and estimate confidence intervals.")
    


# Analytical CI Formula Section (always shown at bottom)
st.markdown("---")
st.subheader("Analytical Confidence Interval Formula")
st.markdown(
    r"""
    **Confidence Interval for the Mean (t-distribution):**
    
    $$\text{CI} = \bar{x} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}$$
    
    where:
    - $\bar{x}$ = sample mean
    - $t_{\alpha/2, n-1}$ = critical value from t-distribution with $n-1$ degrees of freedom
    - $s$ = sample standard deviation
    - $n$ = sample size
    - $\alpha$ = significance level (e.g., $\alpha = 0.05$ for 95% CI)
    
    **Standard Error (SE):**
    
    $$SE = \frac{s}{\sqrt{n}}$$
    
    **Margin of Error:**
    
    $$\text{Margin of Error} = t_{\alpha/2, n-1} \cdot SE$$
    
    **Note:** The analytical formula assumes the data come from a normal distribution. 
    For skewed distributions, this assumption may not hold, and bootstrap methods can provide more reliable intervals.
    """
)
