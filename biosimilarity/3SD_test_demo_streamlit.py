# Biosimilarity Range Test Interactive Demo - Streamlit Version

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from scipy.stats import norm

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.branding import BrandingConfig, apply_branding

apply_branding(
    BrandingConfig(
        app_title="Simple Range Test Interactive Demo",
        header_title="Analytical Biosimilarity — Simple Range Test",
        header_subtitle="Interactive 3SD-style criteria exploration",
    )
)

# Function to run the test once
def run_test(mean_diff, sd_diff, n, x_sd=3.0, proportion=0.99):
    # reference population ground truth
    mu_ref = 0
    sd_ref = 1

    # biosimilar ground truth
    mu_bio = mean_diff * sd_ref
    sd_bio = sd_diff * sd_ref

    # simulate samples
    ref_sample = np.random.normal(mu_ref, sd_ref, n)
    bio_sample = np.random.normal(mu_bio, sd_bio, n)

    # test criterion 1: at least X% of biosimilar runs within Y SD of reference runs
    ref_mean = np.mean(ref_sample)
    ref_std = np.std(ref_sample, ddof=1)

    lower_ref = ref_mean - x_sd * ref_std
    upper_ref = ref_mean + x_sd * ref_std

    within = np.mean((bio_sample >= lower_ref) & (bio_sample <= upper_ref))

    result1 = "Yes" if within >= proportion else "No"
    
    # test criterion 2: bio mean ±3SD fully contained within ref mean ±x_sd*SD
    bio_mean = np.mean(bio_sample)
    bio_std = np.std(bio_sample, ddof=1)
    
    lower_bio = bio_mean - 3 * bio_std
    upper_bio = bio_mean + 3 * bio_std
    
    result2 = "Yes" if (lower_bio >= lower_ref and upper_bio <= upper_ref) else "No"
    
    return result1, result2, ref_sample, bio_sample


# Streamlit App
st.title("Simple Range Test Interactive Demo for Analytical Biosimilarity")
st.markdown("---")

# Custom CSS to increase sidebar font size
st.markdown("""
    <style>
    /* Increase font size for all sidebar content */
    section[data-testid="stSidebar"] {
        font-size: 18px !important;
    }
    
    /* Increase slider labels */
    section[data-testid="stSidebar"] label {
        font-size: 18px !important;
    }
    
    /* Increase slider values */
    section[data-testid="stSidebar"] [data-baseweb="slider"] {
        font-size: 18px !important;
    }
    
    /* Increase header text */
    section[data-testid="stSidebar"] h2 {
        font-size: 24px !important;
    }
    
    /* Increase button text */
    section[data-testid="stSidebar"] button {
        font-size: 18px !important;
    }
    
    /* Increase all text in sidebar */
    section[data-testid="stSidebar"] * {
        font-size: 18px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.header("Parameters")
    
    mean_diff = st.slider(
        "Mean difference (relative to SD ref):",
        min_value=0.0,
        max_value=5.0,
        value=1.2,
        step=0.1
    )
    
    sd_diff = st.slider(
        "SD ratio (SD test to SD ref):",
        min_value=0.0,
        max_value=4.0,
        value=0.9,
        step=0.1
    )
    
    n = st.slider(
        "Number of batches (TP and RP):",
        min_value=10,
        max_value=100,
        value=15,
        step=1
    )
    
    p = st.slider(
        "Repeats to calculate Acceptance Rate:",
        min_value=1,
        max_value=1000,
        value=1000,
        step=1
    )
    
    st.markdown("---")
    st.subheader("Test Criterion Parameters")
    
    x_sd = st.slider(
        "Number of Standard Deviations (x SD):",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.1
    )
    
    proportion = st.slider(
        "Proportion of test runs within range:",
        min_value=0.80,
        max_value=1.00,
        value=0.99,
        step=0.01,
        format="%.2f"
    )
    
    st.markdown("---")
    show_test2 = st.checkbox("Show Test 2 (Containment Criterion)", value=False, 
                             help="Check if Test Mean ±3SD is fully contained within Ref Mean ±x·SD")
    
    st.markdown("---")
    calculate_button = st.button("Calculate", type="primary", use_container_width=True)

# Initialize session state for results
if 'results_ready' not in st.session_state:
    st.session_state.results_ready = False

# Only calculate when button is clicked
if calculate_button:
    st.session_state.results_ready = True
    
    # Run test once and get samples
    result1, result2, ref_sample, bio_sample = run_test(mean_diff, sd_diff, n, x_sd, proportion)
    
    # Store in session state
    st.session_state.result1 = result1
    st.session_state.result2 = result2
    st.session_state.ref_sample = ref_sample
    st.session_state.bio_sample = bio_sample
    st.session_state.mean_diff = mean_diff
    st.session_state.sd_diff = sd_diff
    st.session_state.n = n
    st.session_state.p = p
    st.session_state.x_sd = x_sd
    st.session_state.proportion = proportion
    
    # Run repeated tests for both criteria
    yes_count1 = 0
    yes_count2 = 0
    for _ in range(p):
        r1, r2, _, _ = run_test(mean_diff, sd_diff, n, x_sd, proportion)
        if r1 == "Yes":
            yes_count1 += 1
        if r2 == "Yes":
            yes_count2 += 1
    
    no_count1 = p - yes_count1
    no_count2 = p - yes_count2
    acceptance_rate1 = yes_count1 / p
    acceptance_rate2 = yes_count2 / p
    
    st.session_state.yes_count1 = yes_count1
    st.session_state.no_count1 = no_count1
    st.session_state.acceptance_rate1 = acceptance_rate1
    st.session_state.yes_count2 = yes_count2
    st.session_state.no_count2 = no_count2
    st.session_state.acceptance_rate2 = acceptance_rate2

# Always show population distributions when parameters are set
st.subheader("Population Distributions")

# Calculate true population parameters
mu_ref = 0
sd_ref = 1
mu_bio = mean_diff * sd_ref
sd_bio = sd_diff * sd_ref

# Calculate 99% bounds using z-values
z_99 = norm.ppf(0.995)  # z-value for 99% (0.5% in each tail)
ref_lower_99 = mu_ref - z_99 * sd_ref
ref_upper_99 = mu_ref + z_99 * sd_ref
bio_lower_99 = mu_bio - z_99 * sd_bio
bio_upper_99 = mu_bio + z_99 * sd_bio

# Create plot
fig_dist, ax_dist = plt.subplots(figsize=(11, 5))

# Generate x range for plotting
x_min = min(mu_ref - 4*sd_ref, mu_bio - 4*sd_bio)
x_max = max(mu_ref + 4*sd_ref, mu_bio + 4*sd_bio)
x = np.linspace(x_min, x_max, 500)

# Calculate probability densities
pdf_ref = norm.pdf(x, mu_ref, sd_ref)
pdf_bio = norm.pdf(x, mu_bio, sd_bio)

# Plot distributions
ax_dist.plot(x, pdf_ref, 'r-', linewidth=2, label=f'Reference: N(μ={mu_ref:.1f}, σ={sd_ref:.1f})')
ax_dist.fill_between(x, pdf_ref, alpha=0.3, color='red')
ax_dist.plot(x, pdf_bio, 'b-', linewidth=2, label=f'Test: N(μ={mu_bio:.2f}, σ={sd_bio:.2f})')
ax_dist.fill_between(x, pdf_bio, alpha=0.3, color='blue')

# Add vertical lines for means
ax_dist.axvline(mu_ref, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Reference Mean')
ax_dist.axvline(mu_bio, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'Test Mean')

# Add 99% bounds for reference (using z-value)
ax_dist.axvline(ref_lower_99, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
ax_dist.axvline(ref_upper_99, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
ax_dist.axvspan(ref_lower_99, ref_upper_99, alpha=0.1, color='red', 
                label=f'Reference 99% region (z={z_99:.3f})')

# Add 99% bounds for biosimilar
ax_dist.axvline(bio_lower_99, color='blue', linestyle=':', linewidth=1.5, alpha=0.5)
ax_dist.axvline(bio_upper_99, color='blue', linestyle=':', linewidth=1.5, alpha=0.5)
ax_dist.axvspan(bio_lower_99, bio_upper_99, alpha=0.1, color='blue', 
                label=f'Test 99% region')

ax_dist.set_xlabel("Value", fontsize=12)
ax_dist.set_ylabel("Probability Density", fontsize=12)
ax_dist.set_title("Population Distributions: Reference vs Test", fontsize=14, fontweight='bold')
ax_dist.legend(fontsize=10)
ax_dist.grid(True, alpha=0.3)
st.pyplot(fig_dist)
plt.close()

st.markdown("---")

# Display results if available
if st.session_state.results_ready:
    
    # Create two columns for the main display
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parameter Space")
        
        # Parameter space plot
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        ax1.set_xlim(0, 5)
        ax1.set_ylim(0, 4)
        ax1.set_xlabel("Difference in means (relative to SD ref)")
        ax1.set_ylabel("Ratio of SDs (SD test / SD ref)")
        
        # Orange line between (0,1) and (3,0)
        ax1.plot([0, 3], [1, 0], color='orange', label='Boundary')
        
        # Current point
        ax1.scatter([st.session_state.mean_diff], [st.session_state.sd_diff], 
                   color='blue', s=100, zorder=5, label='Current parameters')
        
        ax1.legend()
        ax1.set_title("Parameter Space")
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        plt.close()
    
    with col2:
        st.subheader(f"Reference ±{st.session_state.x_sd:.1f}SD and Test Samples")
        
        # Reference ±x_sd plot
        ref_mean = np.mean(st.session_state.ref_sample)
        ref_std = np.std(st.session_state.ref_sample, ddof=1)
        lower_ref = ref_mean - st.session_state.x_sd * ref_std
        upper_ref = ref_mean + st.session_state.x_sd * ref_std
        
        # Bio ±3SD bounds
        bio_mean = np.mean(st.session_state.bio_sample)
        bio_std = np.std(st.session_state.bio_sample, ddof=1)
        lower_bio = bio_mean - 3 * bio_std
        upper_bio = bio_mean + 3 * bio_std
        
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.axvline(lower_ref, color='red', linestyle='--', linewidth=2, label=f'Ref ±{st.session_state.x_sd:.1f}SD')
        ax2.axvline(upper_ref, color='red', linestyle='--', linewidth=2)
        ax2.axvline(lower_bio, color='blue', linestyle=':', linewidth=2, label='Test ±3SD')
        ax2.axvline(upper_bio, color='blue', linestyle=':', linewidth=2)
        ax2.scatter(st.session_state.bio_sample, 
                   np.zeros_like(st.session_state.bio_sample), 
                   alpha=0.6, color='blue', s=30, label='Test samples')
        ax2.set_title(f"Acceptance Bounds and Test Distribution")
        ax2.set_yticks([])
        ax2.set_xlabel("Value")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='x')
        st.pyplot(fig2)
        plt.close()
    
    # Results summary
    st.markdown("---")
    st.subheader("Test Results")
    
    # Test 1: Proportion criterion
    st.markdown(f"**Test 1:** At least {st.session_state.proportion*100:.0f}% of test samples within Ref ±{st.session_state.x_sd:.1f}SD")
    metric_col1a, metric_col1b, metric_col1c = st.columns(3)
    
    with metric_col1a:
        st.metric("Single Run Result", st.session_state.result1)
    
    with metric_col1b:
        st.metric("Yes/No Counts", 
                 f"{st.session_state.yes_count1}/{st.session_state.no_count1}")
    
    with metric_col1c:
        st.metric("Acceptance Rate", 
                 f"{st.session_state.acceptance_rate1*100:.1f}%")
    
    # Test 2: Containment criterion (only show if checkbox is activated)
    if show_test2:
        st.markdown("---")
        st.markdown(f"**Test 2:** Test Mean ±3SD fully contained within Ref Mean ±{st.session_state.x_sd:.1f}SD")
        metric_col2a, metric_col2b, metric_col2c = st.columns(3)
        
        with metric_col2a:
            st.metric("Single Run Result", st.session_state.result2)
        
        with metric_col2b:
            st.metric("Yes/No Counts", 
                     f"{st.session_state.yes_count2}/{st.session_state.no_count2}")
        
        with metric_col2c:
            st.metric("Acceptance Rate", 
                     f"{st.session_state.acceptance_rate2*100:.1f}%")
    
    # Sample data table
    st.markdown("---")
    st.subheader("Sample Data")
    
    # Build table
    ref_sample_df = pd.DataFrame({'Reference Sample': st.session_state.ref_sample})
    bio_sample_df = pd.DataFrame({'Test Sample': st.session_state.bio_sample})
    df = pd.concat([ref_sample_df, bio_sample_df], axis=1)
    
    # Add summary statistics at the bottom
    summary_data = {
        'Reference Sample': [
            np.mean(st.session_state.ref_sample), 
            np.std(st.session_state.ref_sample, ddof=1)
        ],
        'Test Sample': [
            np.mean(st.session_state.bio_sample), 
            np.std(st.session_state.bio_sample, ddof=1)
        ]
    }
    
    # Display the main data table
    st.dataframe(df, use_container_width=True, height=300)
    
    # Display summary statistics
    st.subheader("Summary Statistics")
    summary_df = pd.DataFrame(summary_data, index=['Mean', 'Std Dev'])
    st.dataframe(summary_df, use_container_width=True)

else:
    # Instructions when no results yet
    st.info("👈 Adjust the parameters in the sidebar and click **Calculate** to run the simulation.")
    
    st.markdown("""
    ### About this demo
    
    This interactive demo allows you to explore the biosimilarity range test with adjustable criteria.
    
    **Population Parameters:**
    - **Mean difference**: The difference in means between the reference and biosimilar products
    - **SD difference**: The absolute difference in standard deviations
    - **Number of batches**: Sample size for both reference and biosimilar products
    - **Repeats**: Number of simulations to calculate the acceptance rate
    
    **Test Criterion Parameters:**
    - **Number of Standard Deviations (x SD)**: The multiplier for the reference standard deviation to define the acceptance range
    - **Proportion of test runs**: The minimum proportion of biosimilar runs that must fall within the acceptance range
    
    **Test criterion:** At least X% of biosimilar runs must fall within Y standard deviations 
    of the reference runs to be considered biosimilar. You can adjust both X and Y using the sliders.
    """)
