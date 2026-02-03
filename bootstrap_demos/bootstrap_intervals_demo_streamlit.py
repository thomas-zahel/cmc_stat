# Bootstrap CI/PI/TI Demo - Non-Normal Distributions

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
        app_title="Bootstrap Intervals Demo",
        header_title="Bootstrap Intervals",
        header_subtitle="CI / PI / TI for non-normal distributions",
    )
)

# Seed per session
if "random_seed" not in st.session_state:
    st.session_state.random_seed = 42


def generate_sample(dist_type, sample_size, **params):
    if dist_type == "Normal":
        return np.random.normal(params["mean"], params["std"], sample_size)
    if dist_type == "Log-Normal":
        return np.random.lognormal(params["mean"], params["std"], sample_size)
    if dist_type == "Exponential":
        return np.random.exponential(params["scale"], sample_size)
    if dist_type == "Gamma":
        return np.random.gamma(params["shape"], params["scale"], sample_size)
    if dist_type == "Chi-Square":
        return np.random.chisquare(params["df"], sample_size)
    if dist_type == "Beta (Skewed)":
        return np.random.beta(params["alpha"], params["beta"], sample_size)
    raise ValueError("Unsupported distribution")


def get_distribution_properties(dist_type, **params):
    if dist_type == "Normal":
        mean = params["mean"]
        mode = params["mean"]
        std = params["std"]
        pdf = lambda x: stats.norm.pdf(x, mean, std)
        x_range = (mean - 4 * std, mean + 4 * std)
    elif dist_type == "Log-Normal":
        mu = params["mean"]
        sigma = params["std"]
        mean = np.exp(mu + sigma**2 / 2)
        std = np.sqrt((np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2))
        mode = np.exp(mu - sigma**2)
        pdf = lambda x: stats.lognorm.pdf(x, sigma, scale=np.exp(mu))
        x_range = (0, mean + 4 * std)
    elif dist_type == "Exponential":
        scale = params["scale"]
        mean = scale
        std = scale
        mode = 0
        pdf = lambda x: stats.expon.pdf(x, scale=scale)
        x_range = (0, mean * 4)
    elif dist_type == "Gamma":
        shape = params["shape"]
        scale = params["scale"]
        mean = shape * scale
        std = np.sqrt(shape) * scale
        mode = (shape - 1) * scale if shape >= 1 else 0
        pdf = lambda x: stats.gamma.pdf(x, shape, scale=scale)
        x_range = (0, mean + 4 * std)
    elif dist_type == "Chi-Square":
        df = params["df"]
        mean = df
        std = np.sqrt(2 * df)
        mode = max(df - 2, 0)
        pdf = lambda x: stats.chi2.pdf(x, df)
        x_range = (0, mean + 4 * std)
    elif dist_type == "Beta (Skewed)":
        alpha = params["alpha"]
        beta = params["beta"]
        mean = alpha / (alpha + beta)
        var = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
        std = np.sqrt(var)
        mode = (alpha - 1) / (alpha + beta - 2) if (alpha > 1 and beta > 1) else (0 if alpha <= 1 else 1)
        pdf = lambda x: stats.beta.pdf(x, alpha, beta)
        x_range = (0, 1)
    else:
        raise ValueError("Unsupported distribution")
    return mean, mode, std, pdf, x_range


def get_true_quantiles(dist_type, coverage, **params):
    """Get true population quantiles for given coverage level"""
    lower_p = (1 - coverage) / 2
    upper_p = (1 + coverage) / 2
    
    if dist_type == "Normal":
        loc = params["mean"]
        scale = params["std"]
        lower_q = stats.norm.ppf(lower_p, loc, scale)
        upper_q = stats.norm.ppf(upper_p, loc, scale)
    elif dist_type == "Log-Normal":
        mu = params["mean"]
        sigma = params["std"]
        lower_q = stats.lognorm.ppf(lower_p, sigma, scale=np.exp(mu))
        upper_q = stats.lognorm.ppf(upper_p, sigma, scale=np.exp(mu))
    elif dist_type == "Exponential":
        scale = params["scale"]
        lower_q = stats.expon.ppf(lower_p, scale=scale)
        upper_q = stats.expon.ppf(upper_p, scale=scale)
    elif dist_type == "Gamma":
        shape = params["shape"]
        scale = params["scale"]
        lower_q = stats.gamma.ppf(lower_p, shape, scale=scale)
        upper_q = stats.gamma.ppf(upper_p, shape, scale=scale)
    elif dist_type == "Chi-Square":
        df = params["df"]
        lower_q = stats.chi2.ppf(lower_p, df)
        upper_q = stats.chi2.ppf(upper_p, df)
    elif dist_type == "Beta (Skewed)":
        alpha = params["alpha"]
        beta = params["beta"]
        lower_q = stats.beta.ppf(lower_p, alpha, beta)
        upper_q = stats.beta.ppf(upper_p, alpha, beta)
    else:
        raise ValueError("Unsupported distribution")
    
    return lower_q, upper_q

def analytical_intervals(sample, confidence=0.95, coverage=0.90):
    n = len(sample)
    mean = np.mean(sample)
    s = np.std(sample, ddof=1)
    nu = n - 1
    alpha = 1 - confidence

    # Confidence Interval
    t_val = stats.t.ppf(1 - alpha / 2, nu)
    se = s / np.sqrt(n)
    ci = (mean - t_val * se, mean + t_val * se)

    # Prediction Interval
    pi = (mean - t_val * s * np.sqrt(1 + 1 / n),
          mean + t_val * s * np.sqrt(1 + 1 / n))

    # Two-sided normal tolerance interval (Howe / NIST)
    z_p = stats.norm.ppf((1 + coverage) / 2)
    chi_term = stats.chi2.ppf(1 - confidence, nu)

    k = z_p * np.sqrt(nu * (1 + 1 / n) / chi_term)
    ti = (mean - k * s, mean + k * s)

    return ci, pi, ti, mean, s


import numpy as np
from scipy import stats

def bootstrap_intervals(
    sample,
    n_bootstrap=2000,
    confidence=0.95,
    coverage=0.90,
    random_state=None,
):
    rng = np.random.default_rng(random_state)

    sample = np.asarray(sample)
    n = len(sample)
    alpha = 1 - confidence

    bootstrap_means = []
    bootstrap_stds = []
    predictive_draws = []
    example_samples = []

    # For TI
    ti_lower = []
    ti_upper = []

    z_p = stats.norm.ppf((1 + coverage) / 2)

    for i in range(n_bootstrap):
        # Bootstrap resample
        resampled = rng.choice(sample, size=n, replace=True)

        if i < 3:
            example_samples.append(resampled)

        mu_b = np.mean(resampled)
        s_b = np.std(resampled, ddof=1)

        bootstrap_means.append(mu_b)
        bootstrap_stds.append(s_b)

        # ---- Bootstrap PI ----
        # One future observation from estimated population
        predictive_draws.append(rng.normal(mu_b, s_b))

        # ---- Bootstrap TI ----
        # Population quantiles for this bootstrap world
        ti_lower.append(mu_b - z_p * s_b)
        ti_upper.append(mu_b + z_p * s_b)

    # ---- CI for the mean ----
    ci = (
        np.percentile(bootstrap_means, 100 * alpha / 2),
        np.percentile(bootstrap_means, 100 * (1 - alpha / 2)),
    )

    # ---- PI for one future observation ----
    pi = (
        np.percentile(predictive_draws, 100 * alpha / 2),
        np.percentile(predictive_draws, 100 * (1 - alpha / 2)),
    )

    # ---- TI (p-coverage, gamma-confidence) ----
    ti = (
        np.percentile(ti_lower, 100 * alpha / 2),
        np.percentile(ti_upper, 100 * (1 - alpha / 2)),
    )

    # ---- Std diagnostics (NOT used for TI construction) ----
    std_mean = np.mean(bootstrap_stds)
    std_ci = (
        np.percentile(bootstrap_stds, 100 * alpha / 2),
        np.percentile(bootstrap_stds, 100 * (1 - alpha / 2)),
    )

    return (
        np.mean(bootstrap_means),
        ci,
        pi,
        ti,
        example_samples,
        bootstrap_means,
        bootstrap_stds,
        std_mean,
        std_ci,
        predictive_draws,
        ti_lower,
        ti_upper,
    )


def bca_interval(sample, bootstrap_stats, stat_hat, confidence=0.95):
    alpha = 1 - confidence
    prop_less = np.mean(np.array(bootstrap_stats) < stat_hat)
    prop_less = min(max(prop_less, 1e-6), 1 - 1e-6)
    z0 = stats.norm.ppf(prop_less)
    n = len(sample)
    if n <= 2:
        a = 0.0
    else:
        total = np.sum(sample)
        jackknife_stats = (total - sample) / (n - 1)
        mean_jack = np.mean(jackknife_stats)
        num = np.sum((mean_jack - jackknife_stats) ** 3)
        den = 6 * (np.sum((mean_jack - jackknife_stats) ** 2) ** 1.5)
        a = num / den if den != 0 else 0.0

    def adj(prob):
        z = stats.norm.ppf(prob)
        return stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))

    lower_pct = np.clip(adj(alpha / 2), 0, 1) * 100
    upper_pct = np.clip(adj(1 - alpha / 2), 0, 1) * 100
    return (
        np.percentile(bootstrap_stats, lower_pct),
        np.percentile(bootstrap_stats, upper_pct),
    )


# UI
st.title("Bootstrap CI / Prediction / Tolerance Intervals")
st.markdown("---")

with st.sidebar:
    st.header("Settings")
    dist_type = st.selectbox(
        "Distribution:",
        ["Normal", "Log-Normal", "Exponential", "Gamma", "Chi-Square", "Beta (Skewed)"],
    )
    if dist_type == "Normal":
        p1 = st.slider("Mean (μ)", -5.0, 5.0, 0.0, 0.1)
        p2 = st.slider("Std Dev (σ)", 0.1, 5.0, 1.0, 0.1)
        dist_params = {"mean": p1, "std": p2}
    elif dist_type == "Log-Normal":
        p1 = st.slider("Log Mean (μ)", -2.0, 2.0, 0.0, 0.1)
        p2 = st.slider("Log Std Dev (σ)", 0.1, 2.0, 0.5, 0.1)
        dist_params = {"mean": p1, "std": p2}
    elif dist_type == "Exponential":
        p1 = st.slider("Scale (λ⁻¹)", 0.1, 5.0, 1.0, 0.1)
        dist_params = {"scale": p1}
    elif dist_type == "Gamma":
        p1 = st.slider("Shape (k)", 0.5, 10.0, 2.0, 0.5)
        p2 = st.slider("Scale (θ)", 0.1, 5.0, 1.0, 0.1)
        dist_params = {"shape": p1, "scale": p2}
    elif dist_type == "Chi-Square":
        p1 = st.slider("Degrees of Freedom", 1, 20, 5, 1)
        dist_params = {"df": p1}
    elif dist_type == "Beta (Skewed)":
        p1 = st.slider("Alpha (α)", 0.5, 10.0, 2.0, 0.5)
        p2 = st.slider("Beta (β)", 0.5, 10.0, 5.0, 0.5)
        dist_params = {"alpha": p1, "beta": p2}

    st.markdown("---")
    sample_size = st.slider("Sample Size", 4, 1000, 30, 1)
    n_bootstrap = st.slider("Bootstrap Samples", 100, 10000, 1000, 100)
    confidence_level = st.slider("Confidence Level (CI/PI/TI)", 0.5, 0.99, 0.95, 0.01, format="%.2f")
    coverage_level = st.slider("Population Coverage for TI", 0.5, 0.99, 0.90, 0.01, format="%.2f")

    st.markdown("---")
    show_ci = st.checkbox("Show Confidence Interval", value=True)
    show_pi = st.checkbox("Show Prediction Interval", value=True)
    show_ti = st.checkbox("Show Tolerance Interval", value=True)
    
    st.markdown("---")
    show_bootstrap_intervals = st.checkbox("Show Bootstrap Intervals", value=False)
    use_bca = st.checkbox("Use BCa for CI", value=False)

    st.markdown("---")
    resample_button = st.button("Draw New Sample", type="secondary", use_container_width=True)
    bootstrap_button = st.button("Run Bootstrap", type="primary", use_container_width=True)

# Sampling
if "sample" not in st.session_state or resample_button:
    np.random.seed(st.session_state.random_seed)
    st.session_state.sample = generate_sample(dist_type, sample_size, **dist_params)
    st.session_state.random_seed += 1
    st.session_state.bootstrap_done = False

true_mean, true_mode, true_std, pdf, x_range = get_distribution_properties(dist_type, **dist_params)

if bootstrap_button:
    st.session_state.bootstrap_done = True
    ci_a, pi_a, ti_a, sample_mean, sample_std = analytical_intervals(
        st.session_state.sample, confidence_level, coverage_level
    )
    (boot_mean, ci_b, pi_b, ti_b, example_samples, boot_means, boot_stds,
     std_mean, std_ci, predictive_draws, ti_lower, ti_upper) = bootstrap_intervals(
        st.session_state.sample, n_bootstrap, confidence_level, coverage_level
    )
    ci_b_bca = None
    if use_bca:
        ci_b_bca = bca_interval(st.session_state.sample, boot_means, sample_mean, confidence_level)
    st.session_state.analytical = {
        "ci": ci_a,
        "pi": pi_a,
        "ti": ti_a,
        "mean": sample_mean,
        "std": sample_std,
    }
    st.session_state.bootstrap = {
        "mean": boot_mean,
        "ci": ci_b,
        "pi": pi_b,
        "ti": ti_b,
        "ci_bca": ci_b_bca,
        "example": example_samples,
        "boot_means": boot_means,
        "boot_stds": boot_stds,
        "std_mean": std_mean,
        "std_ci": std_ci,
        "predictive": predictive_draws,
        "ti_lower": ti_lower,
        "ti_upper": ti_upper,
    }

st.subheader("Population Distribution")
x = np.linspace(x_range[0], x_range[1], 800)
if x_range[0] >= 0:
    x = x[x >= 0]
y = pdf(x)
fig, ax = plt.subplots(figsize=(11, 5))
ax.fill_between(x, y, alpha=0.3, color="gray", label="True Distribution")
ax.plot(x, y, "k-", linewidth=2)
ax.axvline(true_mean, color="black", linestyle="-", linewidth=3, label=f"True Mean ({true_mean:.3f})")
if abs(true_mean - true_mode) > 0.01:
    ax.axvline(true_mode, color="gray", linestyle=":", linewidth=2, label=f"True Mode ({true_mode:.3f})")

# Add true population quantiles for TI coverage
if show_ti:
    true_lower_q, true_upper_q = get_true_quantiles(dist_type, coverage_level, **dist_params)
    ax.axvline(true_lower_q, color="purple", linestyle="-.", linewidth=2, label=f"True {int(coverage_level*100)}% Lower ({true_lower_q:.3f})")
    ax.axvline(true_upper_q, color="purple", linestyle="-.", linewidth=2, label=f"True {int(coverage_level*100)}% Upper ({true_upper_q:.3f})")

if "sample" in st.session_state:
    ax.axvline(np.mean(st.session_state.sample), color="red", linestyle="--", linewidth=2, label=f"Sample Mean ({np.mean(st.session_state.sample):.2f})")

# Add interval markers if bootstrap analysis done
if st.session_state.get("bootstrap_done", False):
    y_max = max(y)
    a = st.session_state.analytical
    b = st.session_state.bootstrap
    
    # CI intervals at different heights
    if show_ci:
        ax.plot(a["ci"], [0.15 * y_max, 0.15 * y_max], color="green", linewidth=4, marker="|", markersize=15, label="Analytical CI")
        if show_bootstrap_intervals:
            ci_boot = b["ci_bca"] if (use_bca and b["ci_bca"]) else b["ci"]
            ax.plot(ci_boot, [0.2 * y_max, 0.2 * y_max], color="lime", linewidth=4, marker="|", markersize=15, label="Bootstrap CI")
    
    # PI intervals
    if show_pi:
        ax.plot(a["pi"], [0.25 * y_max, 0.25 * y_max], color="blue", linewidth=4, marker="|", markersize=15, label="Analytical PI")
        if show_bootstrap_intervals:
            ax.plot(b["pi"], [0.3 * y_max, 0.3 * y_max], color="cyan", linewidth=4, marker="|", markersize=15, label="Bootstrap PI")
    
    # TI intervals
    if show_ti:
        ax.plot(a["ti"], [0.35 * y_max, 0.35 * y_max], color="orange", linewidth=4, marker="|", markersize=15, label="Analytical TI")
        if show_bootstrap_intervals:
            ax.plot(b["ti"], [0.4 * y_max, 0.4 * y_max], color="gold", linewidth=4, marker="|", markersize=15, label="Bootstrap TI")

ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)
st.pyplot(fig)
plt.close()

st.markdown("---")
st.subheader("Sample Snapshot")
if "sample" in st.session_state:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("n", len(st.session_state.sample))
    with col2:
        st.metric("Mean", f"{np.mean(st.session_state.sample):.3f}")
    with col3:
        st.metric("Std", f"{np.std(st.session_state.sample, ddof=1):.3f}")
    with col4:
        st.metric("Median", f"{np.median(st.session_state.sample):.3f}")
    st.dataframe(pd.DataFrame({"Value": st.session_state.sample}), height=180, use_container_width=True)

if st.session_state.get("bootstrap_done", False):
    st.markdown("---")
    st.subheader("Intervals")
    a = st.session_state.analytical
    b = st.session_state.bootstrap

    if show_ci:
        st.markdown("**Confidence Interval (mean)**")
        st.write(f"Analytical CI: [{a['ci'][0]:.4f}, {a['ci'][1]:.4f}] (t-based)")
        if show_bootstrap_intervals:
            ci_boot = b["ci_bca"] if (use_bca and b["ci_bca"]) else b["ci"]
            label = "BCa Bootstrap CI" if (use_bca and b["ci_bca"]) else "Percentile Bootstrap CI"
            st.write(f"{label}: [{ci_boot[0]:.4f}, {ci_boot[1]:.4f}]")

    if show_pi:
        st.markdown("**Prediction Interval (one future observation)**")
        st.write(f"Analytical PI: [{a['pi'][0]:.4f}, {a['pi'][1]:.4f}] (t-based)")
        if show_bootstrap_intervals:
            st.write(f"Bootstrap PI: [{b['pi'][0]:.4f}, {b['pi'][1]:.4f}] (using mean of bootstrap std: {b['std_mean']:.4f})")

    if show_ti:
        st.markdown("**Tolerance Interval (Howe 1969)**")
        st.write(f"Analytical TI: [{a['ti'][0]:.4f}, {a['ti'][1]:.4f}] (coverage={coverage_level:.2f})")
        if show_bootstrap_intervals:
            st.write(f"Bootstrap TI: [{b['ti'][0]:.4f}, {b['ti'][1]:.4f}] (coverage={coverage_level:.2f}, from bootstrap population quantiles)")

    st.markdown("---")
    st.subheader("Bootstrap Distributions")
    fig2, ax2 = plt.subplots(figsize=(11, 5))
    ax2.hist(b["boot_means"], bins=50, density=True, alpha=0.7, color="skyblue", edgecolor="black", label="Bootstrap Means")
    ax2.axvline(b["mean"], color="blue", linestyle="-", linewidth=2, label=f"Bootstrap Mean ({b['mean']:.3f})")
    ax2.axvline(a["mean"], color="red", linestyle="--", linewidth=2, label=f"Sample Mean ({a['mean']:.3f})")
    ax2.axvline(true_mean, color="black", linestyle="-", linewidth=3, label=f"True Mean ({true_mean:.3f})")
    if show_ci:
        ci_boot = b["ci_bca"] if (use_bca and b["ci_bca"]) else b["ci"]
        ax2.axvspan(ci_boot[0], ci_boot[1], alpha=0.15, color="purple", label="Bootstrap CI")
    ax2.set_xlabel("Mean Value")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)
    plt.close()

    # Plot bootstrap population quantiles for TI
    if show_ti:
        fig_quant, ax_quant = plt.subplots(figsize=(11, 5))
        ax_quant.hist(b["ti_lower"], bins=50, density=True, alpha=0.6, color="orange", edgecolor="black", label=f"Lower {int(coverage_level*100)}% Quantile")
        ax_quant.hist(b["ti_upper"], bins=50, density=True, alpha=0.6, color="gold", edgecolor="black", label=f"Upper {int(coverage_level*100)}% Quantile")
        ax_quant.axvline(b["ti"][0], color="darkorange", linestyle="-", linewidth=3, label=f"Bootstrap TI Lower ({b['ti'][0]:.3f})")
        ax_quant.axvline(b["ti"][1], color="darkgoldenrod", linestyle="-", linewidth=3, label=f"Bootstrap TI Upper ({b['ti'][1]:.3f})")
        ax_quant.axvline(a["ti"][0], color="blue", linestyle="--", linewidth=2, label=f"Analytical TI Lower ({a['ti'][0]:.3f})")
        ax_quant.axvline(a["ti"][1], color="blue", linestyle="--", linewidth=2, label=f"Analytical TI Upper ({a['ti'][1]:.3f})")
        ax_quant.set_xlabel("Quantile Value")
        ax_quant.set_ylabel("Density")
        ax_quant.set_title(f"Bootstrap Population Quantiles (p={coverage_level:.2f})")
        ax_quant.legend(fontsize=9)
        ax_quant.grid(alpha=0.3)
        st.pyplot(fig_quant)
        plt.close()

    fig3, ax3 = plt.subplots(figsize=(11, 5))
    ax3.hist(b["predictive"], bins=50, density=True, alpha=0.7, color="lightgreen", edgecolor="black", label="Predictive Draws")
    if show_pi:
        ax3.axvspan(b["pi"][0], b["pi"][1], alpha=0.2, color="green", label="Bootstrap PI")
    if show_ti:
        ax3.axvspan(b["ti"][0], b["ti"][1], alpha=0.15, color="orange", label="Bootstrap TI")
    ax3.set_xlabel("Predictive Value")
    ax3.set_ylabel("Density")
    ax3.legend()
    ax3.grid(alpha=0.3)
    st.pyplot(fig3)
    plt.close()

    st.markdown("---")
    st.subheader("Formulas (Analytical)")
    st.markdown(
        rf"""
        - CI for mean: $$\bar{{{{x}}}} \pm t_{{{{1-\alpha/2,\,\nu}}}} \; \frac{{{{s}}}}{{{{\sqrt{{{{n}}}}}}}}$$
        - PI for one future obs: $$\bar{{{{x}}}} \pm t_{{{{1-\alpha/2,\,\nu}}}} \; s\,\sqrt{{{{1+1/n}}}}$$
        - TI (Howe 1969, content={coverage_level:.2f}, confidence={confidence_level:.2f}): $$\bar{{{{x}}}} \pm k\,s$$ where $$k = z_{{{{(1+p)/2}}}} \sqrt{{{{\frac{{{{\nu(1+1/n)}}}}{{{{\chi^2_{{{{1-\alpha,\,\nu}}}}}}}}}}}}$$
        """
    )
else:
    st.info("Set parameters, draw a sample, then run bootstrap to see intervals.")
