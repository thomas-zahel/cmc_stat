# Bootstrap f2 Dissolution Profile Comparison with T2EQ Test

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from math import log10
from scipy.stats import norm

# Import T2EQ test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.branding import BrandingConfig, apply_branding
from common.t2eq import t2eq_test

apply_branding(
    BrandingConfig(
        app_title="Dissolution Profile Comparison",
        header_title="Dissolution Profiles",
        header_subtitle="Bootstrap f2 and T2EQ equivalence testing",
    )
)

# Seed per session
if "random_seed" not in st.session_state:
    st.session_state.random_seed = 2026


# ---------- f2 and E(f2) ----------
def f2_stat(ref, test):
    """Calculate f2 similarity factor"""
    R = ref.mean(axis=0)
    T = test.mean(axis=0)
    msd = np.mean((T - R)**2)
    return 100.0 - 25.0 * log10(1.0 + msd)


def Ef2_stat(ref, test):
    """Calculate E(f2) - expected f2 accounting for variability"""
    R = ref.mean(axis=0)
    T = test.mean(axis=0)
    n = ref.shape[0]
    sR2 = ref.var(axis=0, ddof=1)
    sT2 = test.var(axis=0, ddof=1)
    term = np.mean((T - R)**2 + (sR2 + sT2) / n)
    return 100.0 - 25.0 * log10(1.0 + term)


# ---------- Bootstrap helpers ----------
def bootstrap_f2(ref, test, B=1000, rng=None):
    """Bootstrap f2 distribution"""
    if rng is None:
        rng = np.random.default_rng()
    n = ref.shape[0]
    out = np.empty(B)
    for b in range(B):
        idx_r = rng.integers(0, n, size=n)
        idx_t = rng.integers(0, n, size=n)
        out[b] = f2_stat(ref[idx_r, :], test[idx_t, :])
    return out


def bootstrap_Ef2(ref, test, B=1000, rng=None):
    """Bootstrap E(f2) distribution"""
    if rng is None:
        rng = np.random.default_rng()
    n = ref.shape[0]
    out = np.empty(B)
    for b in range(B):
        idx_r = rng.integers(0, n, size=n)
        idx_t = rng.integers(0, n, size=n)
        out[b] = Ef2_stat(ref[idx_r, :], test[idx_t, :])
    return out


def jackknife_f2(ref, test):
    """Jackknife samples for BCa calculation"""
    n = ref.shape[0]
    vals = []
    for i in range(n):
        vals.append(f2_stat(np.delete(ref, i, axis=0), test))
    for i in range(n):
        vals.append(f2_stat(ref, np.delete(test, i, axis=0)))
    return np.array(vals)


def percentile_ci(vals, alpha=0.10):
    """Percentile confidence interval"""
    vals = np.asarray(vals)
    return float(np.quantile(vals, alpha/2)), float(np.quantile(vals, 1 - alpha/2))


def bca_ci(boot_vals, t0, jack_vals, alpha=0.10):
    """Bias-corrected and accelerated confidence interval"""
    boot_vals = np.asarray(boot_vals)
    jack_vals = np.asarray(jack_vals)
    prop = np.mean(boot_vals < t0)
    prop = np.clip(prop, 1e-10, 1-1e-10)
    z0 = norm.ppf(prop)
    jmean = jack_vals.mean()
    num = np.sum((jmean - jack_vals)**3)
    den = 6.0 * (np.sum((jmean - jack_vals)**2) ** 1.5 + 1e-16)
    a = num / den

    def adj(alpha_tail):
        z = norm.ppf(alpha_tail)
        denom = 1 - a * (z0 + z)
        denom = np.where(np.abs(denom) < 1e-12, np.sign(denom) * 1e-12, denom)
        adjq = norm.cdf(z0 + (z0 + z) / denom)
        adjq = float(np.clip(adjq, 0.0, 1.0))
        return np.quantile(boot_vals, adjq)

    return adj(alpha/2), adj(1 - alpha/2)


# ---------- Shah (1998) Example #2 data: 12 tablets × 4 times (30/60/90/180) ----------
ref = np.array([
    [36.1, 58.6, 80.0, 93.3],
    [33.0, 59.5, 80.8, 95.7],
    [35.7, 62.3, 83.0, 97.1],
    [32.1, 62.3, 81.3, 92.8],
    [36.1, 53.6, 72.6, 88.8],
    [34.1, 63.2, 83.0, 97.4],
    [32.4, 61.3, 80.0, 96.8],
    [39.6, 61.8, 80.4, 98.6],
    [34.5, 58.0, 76.9, 93.3],
    [38.0, 59.2, 79.3, 94.0],
    [32.2, 56.2, 77.2, 96.3],
    [35.2, 58.0, 76.7, 96.8],
])

T1 = np.array([
    [38.75, 61.79, 85.14, 100.2],
    [36.16, 61.21, 84.25, 97.3],
    [38.49, 63.89, 84.94, 96.39],
    [37.27, 62.52, 85.65, 95.47],
    [48.12, 77.18, 95.32, 99.3],
    [48.45, 80.62, 95.05, 98.94],
    [41.08, 67.62, 84.94, 99.03],
    [39.64, 63.68, 80.73, 95.63],
    [36.06, 61.59, 82.22, 96.12],
    [36.69, 63.60, 84.50, 98.42],
    [39.95, 67.98, 87.40, 98.10],
    [43.41, 74.07, 93.95, 97.80],
])

T2 = np.array([
    [48, 60, 84, 103],
    [52, 75, 89, 99],
    [48, 60, 83, 101],
    [53, 70, 93, 103],
    [45, 60, 84, 105],
    [48, 66, 90, 103],
    [51, 71, 91, 100],
    [49, 63, 89, 104],
    [44, 60, 84, 103],
    [53, 68, 81, 104],
    [49, 63, 86, 105],
    [52, 68, 87, 104],
])

T3 = np.array([
    [28.7, 48.2, 63.8, 85.6],
    [26.4, 53.1, 68.3, 90.6],
    [25.4, 52.4, 70.0, 89.5],
    [23.2, 49.5, 65.5, 92.2],
    [25.1, 50.7, 68.0, 87.6],
    [28.7, 54.1, 70.8, 93.6],
    [23.5, 50.3, 66.1, 85.1],
    [26.2, 50.6, 67.7, 88.0],
    [25.0, 49.1, 63.6, 85.8],
    [24.9, 49.5, 66.7, 86.6],
    [30.4, 53.9, 70.4, 89.9],
    [22.0, 46.3, 63.0, 88.7],
])

T4 = np.array([
    [17.1, 58.6, 80.0, 93.3],
    [16.0, 59.5, 80.8, 95.7],
    [12.7, 62.3, 83.0, 97.1],
    [15.1, 62.3, 81.3, 92.8],
    [14.1, 53.6, 72.6, 88.8],
    [12.1, 63.2, 83.0, 97.4],
    [14.4, 61.3, 80.0, 96.8],
    [19.6, 61.8, 80.4, 98.6],
    [14.5, 58.0, 76.9, 93.3],
    [14.0, 59.2, 79.3, 94.0],
    [18.2, 56.2, 77.2, 96.3],
    [13.2, 58.0, 76.7, 96.8],
])

T5 = np.array([
    [41.5, 78.0, 86.4, 98.3],
    [43.7, 78.3, 85.9, 102.9],
    [46.3, 78.3, 86.9, 96.4],
    [44.0, 79.9, 88.6, 96.0],
    [42.6, 73.2, 81.4, 95.5],
    [44.4, 78.4, 86.2, 98.4],
    [43.0, 79.0, 87.5, 99.5],
    [44.4, 79.6, 87.3, 99.9],
    [44.8, 78.7, 86.9, 97.8],
    [41.7, 76.9, 84.5, 100.0],
    [42.3, 77.0, 81.9, 97.9],
    [42.0, 78.2, 92.4, 100.3],
])

time_points = [30, 60, 90, 180]
batches = {'Test1': T1, 'Test2': T2, 'Test3': T3, 'Test4': T4, 'Test5': T5}

# UI
st.title("Bootstrap f2 Dissolution Profile Comparison with T²EQ Test")
st.markdown("---")

with st.sidebar:
    st.header("Settings")
    
    test_batch = st.selectbox(
        "Select Test Batch:",
        list(batches.keys())
    )
    
    st.markdown("---")
    st.subheader("Bootstrap Settings")
    
    n_bootstrap = st.slider("Bootstrap Samples", 100, 10000, 1000, 100)
    confidence_level = st.slider("Confidence Level", 0.5, 0.99, 0.90, 0.01, format="%.2f")
    
    st.markdown("---")
    show_f2 = st.checkbox("Show f2 Bootstrap", value=True)
    show_ef2 = st.checkbox("Show E(f2) Bootstrap", value=True)
    use_bca = st.checkbox("Use BCa for f2", value=True, help="Use bias-corrected & accelerated CI instead of percentile")
    
    st.markdown("---")
    st.subheader("T²EQ Test Settings")
    run_t2eq = st.checkbox("Run T²EQ Test", value=False, 
                           help="Hotelling's T²-based multivariate equivalence test")
    t2eq_alpha = st.slider("T²EQ Significance Level (α)", 0.01, 0.10, 0.05, 0.01, format="%.2f")
    
    st.markdown("---")
    bootstrap_button = st.button("Run Analysis", type="primary", use_container_width=True)

# Get selected test batch
test_data = batches[test_batch]

# Display dissolution profiles
st.subheader("Dissolution Profiles")

ref_mean = ref.mean(axis=0)
ref_std = ref.std(axis=0, ddof=1)
test_mean = test_data.mean(axis=0)
test_std = test_data.std(axis=0, ddof=1)

fig_prof, ax_prof = plt.subplots(figsize=(11, 5))
ax_prof.errorbar(time_points, ref_mean, yerr=ref_std, marker='o', capsize=5, 
                 linewidth=2, markersize=8, label='Reference', color='blue')
ax_prof.errorbar(time_points, test_mean, yerr=test_std, marker='s', capsize=5,
                 linewidth=2, markersize=8, label=test_batch, color='red')
ax_prof.set_xlabel("Time (min)", fontsize=12)
ax_prof.set_ylabel("% Dissolved", fontsize=12)
ax_prof.set_title("Mean Dissolution Profiles ± SD", fontsize=14, fontweight='bold')
ax_prof.legend(fontsize=11)
ax_prof.grid(alpha=0.3)
st.pyplot(fig_prof)
plt.close()

# Display data tables
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Reference Data**")
    ref_df = pd.DataFrame(ref, columns=[f"{t} min" for t in time_points])
    ref_df.insert(0, "Tablet", range(1, len(ref)+1))
    st.dataframe(ref_df, height=300, use_container_width=True)
with col2:
    st.markdown(f"**{test_batch} Data**")
    test_df = pd.DataFrame(test_data, columns=[f"{t} min" for t in time_points])
    test_df.insert(0, "Tablet", range(1, len(test_data)+1))
    st.dataframe(test_df, height=300, use_container_width=True)

# Analysis
if bootstrap_button:
    st.session_state.bootstrap_done = True
    
    # Calculate point estimates
    point_f2 = f2_stat(ref, test_data)
    point_ef2 = Ef2_stat(ref, test_data)
    
    # Bootstrap
    rng_f2 = np.random.default_rng(st.session_state.random_seed)
    rng_ef2 = np.random.default_rng(st.session_state.random_seed + 1000)
    
    boot_f2_vals = bootstrap_f2(ref, test_data, B=n_bootstrap, rng=rng_f2) if show_f2 else None
    boot_ef2_vals = bootstrap_Ef2(ref, test_data, B=n_bootstrap, rng=rng_ef2) if show_ef2 else None
    
    # Confidence intervals
    alpha = 1 - confidence_level
    
    if show_f2:
        if use_bca:
            jack_vals = jackknife_f2(ref, test_data)
            ci_f2 = bca_ci(boot_f2_vals, point_f2, jack_vals, alpha=alpha)
            ci_f2_label = "BCa"
        else:
            ci_f2 = percentile_ci(boot_f2_vals, alpha=alpha)
            ci_f2_label = "Percentile"
    
    if show_ef2:
        ci_ef2 = percentile_ci(boot_ef2_vals, alpha=alpha)
    
    # T2EQ test
    t2eq_result = None
    if run_t2eq:
        try:
            t2eq_result = t2eq_test(ref, test_data, alpha=t2eq_alpha)
        except Exception as e:
            st.error(f"T²EQ test failed: {str(e)}")
    
    st.session_state.results = {
        'point_f2': point_f2,
        'point_ef2': point_ef2,
        'boot_f2': boot_f2_vals,
        'boot_ef2': boot_ef2_vals,
        'ci_f2': ci_f2 if show_f2 else None,
        'ci_ef2': ci_ef2 if show_ef2 else None,
        'ci_f2_label': ci_f2_label if show_f2 else None,
        't2eq': t2eq_result,
    }
    st.session_state.random_seed += 1

if st.session_state.get('bootstrap_done', False):
    st.markdown("---")
    st.subheader("Results")
    
    r = st.session_state.results
    
    # T2EQ Results (if run)
    if r['t2eq'] is not None:
        st.markdown("### T²EQ Equivalence Test")
        
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            st.metric("p-value", f"{r['t2eq'].p_value:.4f}")
        with col_t2:
            st.metric("Significance Level (α)", f"{r['t2eq'].alpha:.2f}")
        with col_t3:
            decision_text = "Equivalent ✓" if r['t2eq'].decision else "Not Equivalent ✗"
            st.metric("Decision", decision_text)
        
        if r['t2eq'].p_value < 0.05:
            st.success("✓ p-value < 0.05: Profiles are statistically equivalent (similar)")
        else:
            st.warning("✗ p-value ≥ 0.05: Cannot conclude equivalence")
        
        # Additional T2EQ statistics
        with st.expander("T²EQ Detailed Statistics"):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.write(f"**T² Statistic:** {r['t2eq'].T2:.4f}")
                st.write(f"**Mahalanobis Distance:** {r['t2eq'].MD:.4f}")
                st.write(f"**Test Statistic:** {r['t2eq'].test_stat:.4f}")
            with col_b:
                st.write(f"**Critical Value:** {r['t2eq'].crit_value:.4f}")
                st.write(f"**Non-centrality Param:** {r['t2eq'].ncp:.4f}")
                st.write(f"**Equivalence Margin:** {r['t2eq'].eq_margin:.4f}")
            with col_c:
                st.write(f"**df1:** {r['t2eq'].df1}")
                st.write(f"**df2:** {r['t2eq'].df2}")
                st.write(f"**Sample sizes:** m={r['t2eq'].m}, n={r['t2eq'].n}")
        
        st.markdown("---")
    
    # f2/E(f2) Results
    st.markdown("### f2 Similarity Metrics")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Point Estimate f2", f"{r['point_f2']:.2f}")
        if show_f2 and r['ci_f2']:
            st.metric("Bootstrap Mean f2", f"{np.mean(r['boot_f2']):.2f}")
            st.write(f"**{int(confidence_level*100)}% {r['ci_f2_label']} CI**: [{r['ci_f2'][0]:.2f}, {r['ci_f2'][1]:.2f}]")
            if r['ci_f2'][0] >= 50:
                st.success("✓ Lower bound ≥ 50 (profiles similar)")
            else:
                st.error("✗ Lower bound < 50 (profiles may differ)")
    
    with col_b:
        st.metric("Point Estimate E(f2)", f"{r['point_ef2']:.2f}")
        if show_ef2 and r['ci_ef2']:
            st.metric("Bootstrap Mean E(f2)", f"{np.mean(r['boot_ef2']):.2f}")
            st.write(f"**{int(confidence_level*100)}% Percentile CI**: [{r['ci_ef2'][0]:.2f}, {r['ci_ef2'][1]:.2f}]")
            if r['ci_ef2'][0] >= 50:
                st.success("✓ Lower bound ≥ 50 (profiles similar)")
            else:
                st.error("✗ Lower bound < 50 (profiles may differ)")
    
    # Histograms
    st.markdown("---")
    st.subheader("Bootstrap Distributions")
    
    # Calculate common x-axis limits for consistent comparison
    all_vals = []
    if show_f2 and r['boot_f2'] is not None:
        all_vals.extend(r['boot_f2'])
    if show_ef2 and r['boot_ef2'] is not None:
        all_vals.extend(r['boot_ef2'])
    
    if all_vals:
        x_min = np.min(all_vals)
        x_max = np.max(all_vals)
        x_margin = (x_max - x_min) * 0.05
        xlim = (x_min - x_margin, x_max + x_margin)
    else:
        xlim = None
    
    if show_f2 and r['boot_f2'] is not None:
        fig_f2, ax_f2 = plt.subplots(figsize=(11, 5))
        ax_f2.hist(r['boot_f2'], bins=50, density=True, alpha=0.7, color='skyblue', 
                   edgecolor='black', label='Bootstrap f2')
        ax_f2.axvline(r['point_f2'], color='red', linestyle='--', linewidth=2, 
                     label=f"Point f2 ({r['point_f2']:.2f})")
        ax_f2.axvline(np.mean(r['boot_f2']), color='blue', linestyle='-', linewidth=2,
                     label=f"Bootstrap Mean ({np.mean(r['boot_f2']):.2f})")
        ax_f2.axvline(50, color='green', linestyle=':', linewidth=3, 
                     label='Similarity Threshold (50)')
        if r['ci_f2']:
            ax_f2.axvspan(r['ci_f2'][0], r['ci_f2'][1], alpha=0.15, color='purple',
                         label=f"{int(confidence_level*100)}% {r['ci_f2_label']} CI")
        if xlim:
            ax_f2.set_xlim(xlim)
        ax_f2.set_xlabel("f2 Value", fontsize=12)
        ax_f2.set_ylabel("Density", fontsize=12)
        ax_f2.set_title(f"Bootstrap Distribution of f2 ({test_batch} vs Reference)", 
                       fontsize=14, fontweight='bold')
        ax_f2.legend(fontsize=10)
        ax_f2.grid(alpha=0.3)
        st.pyplot(fig_f2)
        plt.close()
    
    if show_ef2 and r['boot_ef2'] is not None:
        fig_ef2, ax_ef2 = plt.subplots(figsize=(11, 5))
        ax_ef2.hist(r['boot_ef2'], bins=50, density=True, alpha=0.7, color='lightcoral',
                    edgecolor='black', label='Bootstrap E(f2)')
        ax_ef2.axvline(r['point_ef2'], color='darkred', linestyle='--', linewidth=2,
                      label=f"Point E(f2) ({r['point_ef2']:.2f})")
        ax_ef2.axvline(np.mean(r['boot_ef2']), color='red', linestyle='-', linewidth=2,
                      label=f"Bootstrap Mean ({np.mean(r['boot_ef2']):.2f})")
        ax_ef2.axvline(50, color='green', linestyle=':', linewidth=3,
                      label='Similarity Threshold (50)')
        if r['ci_ef2']:
            ax_ef2.axvspan(r['ci_ef2'][0], r['ci_ef2'][1], alpha=0.15, color='orange',
                          label=f"{int(confidence_level*100)}% Percentile CI")
        if xlim:
            ax_ef2.set_xlim(xlim)
        ax_ef2.set_xlabel("E(f2) Value", fontsize=12)
        ax_ef2.set_ylabel("Density", fontsize=12)
        ax_ef2.set_title(f"Bootstrap Distribution of E(f2) ({test_batch} vs Reference)",
                        fontsize=14, fontweight='bold')
        ax_ef2.legend(fontsize=10)
        ax_ef2.grid(alpha=0.3)
        st.pyplot(fig_ef2)
        plt.close()
    
    # Formulas
    st.markdown("---")
    st.subheader("Formulas")
    st.markdown(
        r"""
        **f2 Similarity Factor:**
        $$f_2 = 50 \times \log_{10}\left[\frac{100}{\sqrt{1 + \frac{1}{n}\sum_{t=1}^{n}(R_t - T_t)^2}}\right]$$
        
        **E(f2) - Expected f2:**
        $$E(f_2) = 50 \times \log_{10}\left[\frac{100}{\sqrt{1 + \frac{1}{n}\sum_{t=1}^{n}\left[(R_t - T_t)^2 + \frac{s_R^2 + s_T^2}{N}\right]}}\right]$$
        
        where $R_t$, $T_t$ are mean % dissolved at time $t$, $s_R^2$, $s_T^2$ are variances, $N$ is sample size per batch, and $n$ is number of time points.
        
        **T²EQ Test (Hoffelder):**
        Multivariate equivalence test using Hotelling's T² statistic with Mahalanobis distance:
        $$MD = (\bar{R} - \bar{T})' S^{-1}_{\text{pooled}} (\bar{R} - \bar{T})$$
        $$T^2 = \frac{mn}{N} \times MD$$
        
        The test uses a non-central F distribution with equivalence margin based on acceptance vector D (default: 10% at each time point).
        
        **Acceptance Criteria:**
        - f2 ≥ 50 indicates profile similarity
        - T²EQ p-value < 0.05 indicates statistical equivalence
        """
    )

else:
    st.info("👈 Select a test batch and click **Run Analysis** to compare dissolution profiles.")
    st.markdown("""
    ### About Dissolution Profile Comparison
    
    **f2 Similarity Factor:**
    - Regulatory metric to compare dissolution profiles
    - f2 ≥ 50 indicates similarity (FDA/EMA standard)
    
    **E(f2):**
    - Accounts for within-batch variability
    - More conservative estimate
    
    **T²EQ Test:**
    - Multivariate equivalence test using Hotelling's T² statistic
    - Considers correlation structure between time points
    - p-value < 0.05 indicates statistical equivalence
    - Based on Mahalanobis distance with pooled covariance
    
    **Bootstrap Confidence Intervals:**
    - Quantify uncertainty in f2 estimates
    - Help determine if lower CI bound exceeds 50
    """)
