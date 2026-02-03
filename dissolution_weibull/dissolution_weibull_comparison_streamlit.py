# Bayesian Weibull Model for Dissolution Profiles - Reference vs Test Comparison

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pymc as pm
import arviz as az

# Seed per session
if "random_seed" not in st.session_state:
    st.session_state.random_seed = 2026
    
# Initialize reference fit flag
if "reference_fitted" not in st.session_state:
    st.session_state.reference_fitted = False


def weibull_dissolution(t, f_max, td, beta):
    """
    Weibull dissolution model
    f(t) = f_max * (1 - exp(-(t/td)^beta))
    
    Parameters:
    - f_max: maximum dissolution (asymptote)
    - td: time scale parameter
    - beta: shape parameter
    """
    return f_max * (1 - np.exp(-(t / td) ** beta))


def calculate_rmsd(profile_ref, profile_test):
    """Calculate Root Mean Squared Difference between two profiles"""
    return np.sqrt(np.mean((profile_ref - profile_test) ** 2))


def fit_weibull_bayesian(time_points, dissolution_data, n_samples=1000, n_chains=2):
    """
    Fit Weibull model using Bayesian inference with PyMC
    
    Parameters:
    - time_points: array of time points
    - dissolution_data: array of shape (n_tablets, n_timepoints)
    - n_samples: number of MCMC samples per chain
    - n_chains: number of MCMC chains
    
    Returns:
    - trace: PyMC trace object with posterior samples
    - model: PyMC model object
    """
    # Flatten data for modeling
    n_tablets, n_times = dissolution_data.shape
    times_flat = np.tile(time_points, n_tablets)
    dissolutions_flat = dissolution_data.flatten()
    
    # Get data statistics for better priors
    max_diss = np.max(dissolutions_flat)
    mean_diss = np.mean(dissolutions_flat)
    max_time = np.max(time_points)
    
    with pm.Model() as model:
        # More informative priors based on data
        # f_max: maximum dissolution (use data maximum as guide)
        f_max = pm.Normal("f_max", mu=max_diss, sigma=10, 
                          initval=max_diss)
        
        # td: time scale parameter (use middle of time range)
        td = pm.Gamma("td", alpha=2, beta=2/max_time, 
                      initval=max_time/2)
        
        # beta: shape parameter (weakly informative around 1)
        beta = pm.Gamma("beta", alpha=2, beta=1, 
                        initval=1.0)
        
        # sigma: residual standard deviation
        sigma = pm.HalfNormal("sigma", sigma=5, 
                              initval=3.0)
        
        # Expected dissolution at each time point
        # Add small epsilon to avoid numerical issues
        mu = f_max * (1 - pm.math.exp(-((times_flat + 1e-6) / td) ** beta))
        
        # Likelihood: normally distributed residuals
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=dissolutions_flat)
        
        # Sample from posterior with better settings
        trace = pm.sample(
            draws=n_samples,
            chains=n_chains,
            tune=500,
            return_inferencedata=True,
            random_seed=st.session_state.random_seed,
            target_accept=0.9,
            cores=1  # Sequential for stability
        )
    
    return trace, model


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

time_points = np.array([30, 60, 90, 180])
batches = {'Test1': T1, 'Test2': T2, 'Test3': T3, 'Test4': T4, 'Test5': T5}


# UI
st.title("Bayesian Weibull: Reference vs Test Comparison")
st.markdown("---")

with st.sidebar:
    st.header("Settings")
    
    test_batch = st.selectbox(
        "Select Test Batch:",
        list(batches.keys())
    )
    
    st.markdown("---")
    st.subheader("MCMC Settings")
    
    n_samples = st.slider("MCMC Samples per Chain", 500, 5000, 1000, 500)
    n_chains = st.slider("Number of Chains", 2, 8, 2, 1)
    
    st.markdown("---")
    show_individual = st.checkbox("Show Individual Tablets", value=False)
    show_posterior_predictive = st.checkbox("Show Posterior Predictive", value=True)
    
    st.markdown("---")
    fit_button = st.button("Fit Test Batch", type="primary", use_container_width=True)

# Fit reference batch automatically on first load
if not st.session_state.reference_fitted:
    with st.spinner("Fitting reference batch... Please wait 30-60 seconds."):
        try:
            ref_trace, ref_model = fit_weibull_bayesian(time_points, ref, n_samples=1000, n_chains=2)
            st.session_state.ref_trace = ref_trace
            st.session_state.ref_model = ref_model
            st.session_state.reference_fitted = True
            st.success("✓ Reference batch fitted successfully!")
        except Exception as e:
            st.error(f"Error fitting reference: {str(e)}")
            st.stop()

# Get selected test batch
test_data = batches[test_batch]

# Display data combined in one plot
st.subheader("Dissolution Profiles: Reference vs Test")

ref_mean = ref.mean(axis=0)
ref_std = ref.std(axis=0, ddof=1)
test_mean = test_data.mean(axis=0)
test_std = test_data.std(axis=0, ddof=1)

fig_data, ax_data = plt.subplots(figsize=(11, 5))

if show_individual:
    for i in range(len(ref)):
        ax_data.plot(time_points, ref[i, :], 'o-', alpha=0.2, color='blue', linewidth=1, markersize=4)
    for i in range(len(test_data)):
        ax_data.plot(time_points, test_data[i, :], 's-', alpha=0.2, color='red', linewidth=1, markersize=4)

ax_data.errorbar(time_points, ref_mean, yerr=ref_std, marker='o', capsize=5,
                 linewidth=2, markersize=10, label='Reference Mean ± SD',
                 color='blue', ecolor='blue', capthick=2)
ax_data.errorbar(time_points, test_mean, yerr=test_std, marker='s', capsize=5,
                 linewidth=2, markersize=10, label=f'{test_batch} Mean ± SD',
                 color='red', ecolor='red', capthick=2)

ax_data.set_xlabel("Time (min)", fontsize=12)
ax_data.set_ylabel("% Dissolved", fontsize=12)
ax_data.set_title("Dissolution Profiles: Reference vs Test", fontsize=14, fontweight='bold')
ax_data.legend(fontsize=10)
ax_data.grid(alpha=0.3)
st.pyplot(fig_data)
plt.close()

# Data tables side by side
col_data1, col_data2 = st.columns(2)

with col_data1:
    st.markdown("**Reference Batch**")
    ref_df = pd.DataFrame(ref, columns=[f"{t} min" for t in time_points])
    ref_df.insert(0, "Tablet", range(1, len(ref)+1))
    st.dataframe(ref_df, height=200, use_container_width=True)

with col_data2:
    st.markdown(f"**{test_batch}**")
    test_df = pd.DataFrame(test_data, columns=[f"{t} min" for t in time_points])
    test_df.insert(0, "Tablet", range(1, len(test_data)+1))
    st.dataframe(test_df, height=200, use_container_width=True)

# Bayesian inference for test batch
if fit_button:
    with st.spinner("Running MCMC sampling for test batch... This may take 30-60 seconds."):
        try:
            test_trace, test_model = fit_weibull_bayesian(time_points, test_data, n_samples, n_chains)
            st.session_state.test_trace = test_trace
            st.session_state.test_model = test_model
            st.session_state.test_fitted = True
            st.success("✓ Test batch fitted successfully!")
        except Exception as e:
            st.error(f"Error fitting test batch: {str(e)}")
            st.session_state.test_fitted = False

# Display results if both reference and test are fitted
if st.session_state.get("test_fitted", False):
    ref_trace = st.session_state.ref_trace
    test_trace = st.session_state.test_trace
    
    st.markdown("---")
    st.subheader("Bayesian Fit Results: Reference vs Test")
    
    # Extract posterior samples
    ref_posterior = ref_trace.posterior
    ref_f_max = ref_posterior.f_max.values.flatten()
    ref_td = ref_posterior.td.values.flatten()
    ref_beta = ref_posterior.beta.values.flatten()
    ref_sigma = ref_posterior.sigma.values.flatten()
    
    test_posterior = test_trace.posterior
    test_f_max = test_posterior.f_max.values.flatten()
    test_td = test_posterior.td.values.flatten()
    test_beta = test_posterior.beta.values.flatten()
    test_sigma = test_posterior.sigma.values.flatten()
    
    # Generate predictions for both
    t_pred = np.linspace(0, max(time_points) * 1.2, 200)
    n_posterior_samples = min(500, len(ref_f_max))
    idx = np.random.choice(len(ref_f_max), n_posterior_samples, replace=False)
    
    ref_predictions = np.zeros((n_posterior_samples, len(t_pred)))
    for i, idx_i in enumerate(idx):
        ref_predictions[i, :] = weibull_dissolution(t_pred, ref_f_max[idx_i], 
                                                     ref_td[idx_i], ref_beta[idx_i])
    
    ref_pred_mean = np.mean(ref_predictions, axis=0)
    ref_pred_lower = np.percentile(ref_predictions, 2.5, axis=0)
    ref_pred_upper = np.percentile(ref_predictions, 97.5, axis=0)
    
    test_predictions = np.zeros((n_posterior_samples, len(t_pred)))
    for i, idx_i in enumerate(idx):
        test_predictions[i, :] = weibull_dissolution(t_pred, test_f_max[idx_i], 
                                                      test_td[idx_i], test_beta[idx_i])
    
    test_pred_mean = np.mean(test_predictions, axis=0)
    test_pred_lower = np.percentile(test_predictions, 2.5, axis=0)
    test_pred_upper = np.percentile(test_predictions, 97.5, axis=0)
    
    # Combined fit plot
    st.markdown("### Bayesian Weibull Fits: Reference vs Test")
    fig_fit, ax_fit = plt.subplots(figsize=(12, 6))
    
    if show_individual:
        for i in range(len(ref)):
            ax_fit.plot(time_points, ref[i, :], 'o', alpha=0.2, 
                       color='lightblue', markersize=5)
        for i in range(len(test_data)):
            ax_fit.plot(time_points, test_data[i, :], 's', alpha=0.2, 
                       color='lightcoral', markersize=5)
    
    # Reference data and fit
    ax_fit.errorbar(time_points, ref_mean, yerr=ref_std, 
                   marker='o', capsize=5, linewidth=0, elinewidth=2,
                   markersize=10, label='Reference Data Mean ± SD',
                   color='blue', ecolor='blue', capthick=2)
    ax_fit.plot(t_pred, ref_pred_mean, 'b-', linewidth=3, 
               label='Reference Posterior Mean')
    ax_fit.fill_between(t_pred, ref_pred_lower, ref_pred_upper, alpha=0.2, 
                        color='blue', label='Reference 95% CI')
    
    # Test data and fit
    ax_fit.errorbar(time_points, test_mean, yerr=test_std, 
                   marker='s', capsize=5, linewidth=0, elinewidth=2,
                   markersize=10, label=f'{test_batch} Data Mean ± SD',
                   color='red', ecolor='red', capthick=2)
    ax_fit.plot(t_pred, test_pred_mean, 'r-', linewidth=3, 
               label=f'{test_batch} Posterior Mean')
    ax_fit.fill_between(t_pred, test_pred_lower, test_pred_upper, alpha=0.2, 
                        color='red', label=f'{test_batch} 95% CI')
    
    if show_posterior_predictive:
        ref_pred_lower_obs = ref_pred_mean - 1.96 * np.mean(ref_sigma)
        ref_pred_upper_obs = ref_pred_mean + 1.96 * np.mean(ref_sigma)
        ax_fit.fill_between(t_pred, ref_pred_lower_obs, ref_pred_upper_obs, 
                           alpha=0.1, color='blue', linestyle='--')
        test_pred_lower_obs = test_pred_mean - 1.96 * np.mean(test_sigma)
        test_pred_upper_obs = test_pred_mean + 1.96 * np.mean(test_sigma)
        ax_fit.fill_between(t_pred, test_pred_lower_obs, test_pred_upper_obs, 
                           alpha=0.1, color='red', linestyle='--')
    
    ax_fit.set_xlabel("Time (min)", fontsize=12)
    ax_fit.set_ylabel("% Dissolved", fontsize=12)
    ax_fit.set_title("Bayesian Weibull Fits: Reference vs Test", fontsize=14, fontweight='bold')
    ax_fit.legend(fontsize=9, loc='lower right')
    ax_fit.grid(alpha=0.3)
    st.pyplot(fig_fit)
    plt.close()
    
    # Parameter estimates side by side
    st.markdown("---")
    st.markdown("### Parameter Estimates")
    col_param1, col_param2 = st.columns(2)
    
    with col_param1:
        st.markdown("**Reference Parameters:**")
        ref_params_df = pd.DataFrame({
            'Parameter': ['f_max (%)', 'td (min)', 'beta', 'sigma (%)'],
            'Mean': [np.mean(ref_f_max), np.mean(ref_td), 
                     np.mean(ref_beta), np.mean(ref_sigma)],
            '2.5%': [np.percentile(ref_f_max, 2.5), np.percentile(ref_td, 2.5),
                     np.percentile(ref_beta, 2.5), np.percentile(ref_sigma, 2.5)],
            '97.5%': [np.percentile(ref_f_max, 97.5), np.percentile(ref_td, 97.5),
                      np.percentile(ref_beta, 97.5), np.percentile(ref_sigma, 97.5)]
        })
        st.dataframe(ref_params_df.style.format({
            'Mean': '{:.2f}',
            '2.5%': '{:.2f}',
            '97.5%': '{:.2f}'
        }), use_container_width=True)
    
    with col_param2:
        st.markdown(f"**{test_batch} Parameters:**")
        test_params_df = pd.DataFrame({
            'Parameter': ['f_max (%)', 'td (min)', 'beta', 'sigma (%)'],
            'Mean': [np.mean(test_f_max), np.mean(test_td), 
                     np.mean(test_beta), np.mean(test_sigma)],
            '2.5%': [np.percentile(test_f_max, 2.5), np.percentile(test_td, 2.5),
                     np.percentile(test_beta, 2.5), np.percentile(test_sigma, 2.5)],
            '97.5%': [np.percentile(test_f_max, 97.5), np.percentile(test_td, 97.5),
                      np.percentile(test_beta, 97.5), np.percentile(test_sigma, 97.5)]
        })
        st.dataframe(test_params_df.style.format({
            'Mean': '{:.2f}',
            '2.5%': '{:.2f}',
            '97.5%': '{:.2f}'
        }), use_container_width=True)
    
    # Parameter uncertainty histograms
    st.markdown("---")
    st.markdown("### Parameter Posterior Distributions")
    
    fig_params, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # f_max
    axes[0, 0].hist(ref_f_max, bins=40, alpha=0.6, color='blue', label='Reference', density=True)
    axes[0, 0].hist(test_f_max, bins=40, alpha=0.6, color='red', label=test_batch, density=True)
    axes[0, 0].axvline(np.mean(ref_f_max), color='blue', linestyle='--', linewidth=2)
    axes[0, 0].axvline(np.mean(test_f_max), color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('f_max (%)', fontsize=11)
    axes[0, 0].set_ylabel('Density', fontsize=11)
    axes[0, 0].set_title('Maximum Dissolution (f_max)', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)
    
    # td
    axes[0, 1].hist(ref_td, bins=40, alpha=0.6, color='blue', label='Reference', density=True)
    axes[0, 1].hist(test_td, bins=40, alpha=0.6, color='red', label=test_batch, density=True)
    axes[0, 1].axvline(np.mean(ref_td), color='blue', linestyle='--', linewidth=2)
    axes[0, 1].axvline(np.mean(test_td), color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('td (min)', fontsize=11)
    axes[0, 1].set_ylabel('Density', fontsize=11)
    axes[0, 1].set_title('Time Scale Parameter (td)', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(alpha=0.3)
    
    # beta
    axes[1, 0].hist(ref_beta, bins=40, alpha=0.6, color='blue', label='Reference', density=True)
    axes[1, 0].hist(test_beta, bins=40, alpha=0.6, color='red', label=test_batch, density=True)
    axes[1, 0].axvline(np.mean(ref_beta), color='blue', linestyle='--', linewidth=2)
    axes[1, 0].axvline(np.mean(test_beta), color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('beta', fontsize=11)
    axes[1, 0].set_ylabel('Density', fontsize=11)
    axes[1, 0].set_title('Shape Parameter (beta)', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(alpha=0.3)
    
    # sigma
    axes[1, 1].hist(ref_sigma, bins=40, alpha=0.6, color='blue', label='Reference', density=True)
    axes[1, 1].hist(test_sigma, bins=40, alpha=0.6, color='red', label=test_batch, density=True)
    axes[1, 1].axvline(np.mean(ref_sigma), color='blue', linestyle='--', linewidth=2)
    axes[1, 1].axvline(np.mean(test_sigma), color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('sigma (%)', fontsize=11)
    axes[1, 1].set_ylabel('Density', fontsize=11)
    axes[1, 1].set_title('Residual Standard Deviation (sigma)', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig_params)
    plt.close()
    
    st.markdown("""
    **Interpretation:**
    - Dashed vertical lines show posterior means for each parameter
    - Overlapping distributions suggest similar parameter values
    - Non-overlapping distributions indicate significant differences
    - The width of each distribution represents the uncertainty in that parameter
    """)

    # Residual diagnostics to check homoscedasticity
    st.markdown("---")
    st.subheader("Residuals vs Time (Homoscedasticity Check)")

    # Use posterior mean parameters as point predictions for residuals
    ref_mean_params = {
        "f_max": np.mean(ref_f_max),
        "td": np.mean(ref_td),
        "beta": np.mean(ref_beta),
    }
    test_mean_params = {
        "f_max": np.mean(test_f_max),
        "td": np.mean(test_td),
        "beta": np.mean(test_beta),
    }

    def compute_residuals(observed, params):
        preds = weibull_dissolution(time_points, params["f_max"], params["td"], params["beta"])
        return observed - preds

    ref_residuals = compute_residuals(ref, ref_mean_params)
    test_residuals = compute_residuals(test_data, test_mean_params)

    fig_res, ax_res = plt.subplots(figsize=(11, 5))
    jitter_scale = 0.8
    rng = np.random.default_rng(st.session_state.random_seed)

    # Scatter residuals per time point with slight jitter for visibility
    for i, t in enumerate(time_points):
        jitter_ref = rng.normal(0, jitter_scale, size=ref_residuals.shape[0])
        jitter_test = rng.normal(0, jitter_scale, size=test_residuals.shape[0])
        ax_res.scatter(t + jitter_ref, ref_residuals[:, i], color="blue", alpha=0.4, label="Reference" if i == 0 else None)
        ax_res.scatter(t + jitter_test, test_residuals[:, i], color="red", alpha=0.4, label=test_batch if i == 0 else None, marker="s")

    ax_res.axhline(0, color="black", linewidth=1.5, linestyle="--", label="Zero Residual")
    ax_res.set_xlabel("Time (min)", fontsize=12)
    ax_res.set_ylabel("Residual (% dissolved)", fontsize=12)
    ax_res.set_title("Residuals vs Time Points", fontsize=14, fontweight="bold")
    ax_res.legend(fontsize=10)
    ax_res.grid(alpha=0.3)

    st.pyplot(fig_res)
    plt.close()
    st.markdown("Residuals should be roughly centered around zero with similar spread across time points. Fanning patterns or time-dependent spread suggest heteroscedasticity.")
    
    # Calculate RMSD distribution
    st.markdown("---")
    st.subheader("Similarity Analysis: Posterior RMSD Distribution")
    
    st.write("Calculating RMSD by sampling from both reference and test posteriors...")
    
    # Sample RMSD from posteriors
    n_rmsd_samples = min(2000, len(ref_f_max))
    rmsd_samples = np.zeros(n_rmsd_samples)
    
    for i in range(n_rmsd_samples):
        # Sample parameters from posteriors
        idx_ref = np.random.randint(len(ref_f_max))
        idx_test = np.random.randint(len(test_f_max))
        
        # Calculate profiles at observation time points
        ref_profile = weibull_dissolution(time_points, ref_f_max[idx_ref], 
                                          ref_td[idx_ref], ref_beta[idx_ref])
        test_profile = weibull_dissolution(time_points, test_f_max[idx_test], 
                                           test_td[idx_test], test_beta[idx_test])
        
        # Calculate RMSD
        rmsd_samples[i] = calculate_rmsd(ref_profile, test_profile)
    
    # Plot RMSD distribution
    fig_rmsd, ax_rmsd = plt.subplots(figsize=(11, 5))
    
    ax_rmsd.hist(rmsd_samples, bins=50, density=True, alpha=0.7, 
                 color='purple', edgecolor='black', label='Posterior RMSD')
    ax_rmsd.axvline(np.mean(rmsd_samples), color='blue', linestyle='-', 
                    linewidth=3, label=f"Mean RMSD: {np.mean(rmsd_samples):.2f}")
    ax_rmsd.axvline(np.percentile(rmsd_samples, 5), color='orange', 
                    linestyle='--', linewidth=2, label='90% Credible Interval')
    ax_rmsd.axvline(np.percentile(rmsd_samples, 95), color='orange', 
                    linestyle='--', linewidth=2)
    ax_rmsd.axvline(9.95, color='red', linestyle=':', linewidth=3, 
                    label='Similarity Threshold (10%)')
    
    # Shade region below threshold
    rmsd_range = np.linspace(rmsd_samples.min(), rmsd_samples.max(), 1000)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(rmsd_samples)
    density = kde(rmsd_range)
    ax_rmsd.fill_between(rmsd_range[rmsd_range <= 9.95], 
                         density[rmsd_range <= 9.95], 
                         alpha=0.3, color='green', label='Similar Region')
    
    ax_rmsd.set_xlabel("RMSD (%)", fontsize=12)
    ax_rmsd.set_ylabel("Density", fontsize=12)
    ax_rmsd.set_title("Posterior Distribution of RMSD (accounting for parameter uncertainty)", 
                     fontsize=14, fontweight='bold')
    ax_rmsd.legend(fontsize=10)
    ax_rmsd.grid(alpha=0.3)
    st.pyplot(fig_rmsd)
    plt.close()
    
    # Similarity metrics
    prob_similar = np.mean(rmsd_samples < 9.95)
    
    col_sim1, col_sim2, col_sim3 = st.columns(3)
    with col_sim1:
        st.metric("Mean RMSD", f"{np.mean(rmsd_samples):.2f}%")
    with col_sim2:
        st.metric("90% CI for RMSD", 
                 f"[{np.percentile(rmsd_samples, 5):.2f}, {np.percentile(rmsd_samples, 95):.2f}]")
    with col_sim3:
        st.metric("P(RMSD < 9.95%)", f"{prob_similar:.3f}")
    
    if prob_similar > 0.95:
        st.success(f"✓ High confidence of similarity (P = {prob_similar:.3f} > 0.95)")
    elif prob_similar > 0.80:
        st.warning(f"⚠ Moderate confidence of similarity (P = {prob_similar:.3f})")
    else:
        st.error(f"✗ Low confidence of similarity (P = {prob_similar:.3f} < 0.80)")
    
    st.markdown("""
    **Interpretation:**
    - **Mean RMSD**: Average difference between reference and test profiles across posterior samples
    - **P(RMSD < 9.95%)**: Probability that profiles are similar (within 10% threshold), accounting for uncertainty in both reference and test parameters
    - **Decision rule**: P > 0.95 suggests high confidence of similarity
    """)

else:
    st.info("👈 The reference batch has been fitted. Select a test batch and click **Fit Test Batch** to compare.")
    st.markdown("""
    ### About Bayesian Similarity Analysis
    
    This app automatically fits the **reference batch** when opened, then allows you to:
    1. Select and fit a **test batch**
    2. Compare fitted Weibull parameters between reference and test
    3. Calculate **posterior RMSD distribution** by sampling from both posteriors
    
    **Key Advantage:**
    Instead of using point estimates, this approach accounts for **uncertainty in both reference and test parameters**,
    providing a probability statement about similarity: P(RMSD < 10% threshold).
    
    **Weibull Model**: f(t) = f_max × (1 - exp(-(t/td)^β))
    - f_max: maximum dissolution
    - td: time scale parameter
    - β: shape parameter
    """)
