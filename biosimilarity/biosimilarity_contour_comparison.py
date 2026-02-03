"""
Biosimilarity Test Comparison - Contour Plot Analysis
Compares 4 different biosimilarity tests across parameter space
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm, t
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'common'))
from bootstrap_biosimilarity_test import biosimilarity_bootstrap_test
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# TEST IMPLEMENTATIONS
# ============================================================================

def test_3sd(ref_sample, test_sample, p_tp, p_rp):
    """
    3SD-type test: Check if X% (p_tp) of test products are within 
    the Y% (p_rp) quantile range of reference.
    
    Returns True if biosimilar (passes test).
    """
    # Convert p_rp to z-score (e.g., 99% -> ~2.576 SD)
    z_score = norm.ppf((1 + p_rp) / 2)  # Two-sided quantile
    
    ref_mean = np.mean(ref_sample)
    ref_std = np.std(ref_sample, ddof=1)
    
    lower_bound = ref_mean - z_score * ref_std
    upper_bound = ref_mean + z_score * ref_std
    
    # Calculate proportion of test samples within bounds
    within = np.mean((test_sample >= lower_bound) & (test_sample <= upper_bound))
    
    # Pass if at least p_tp proportion is within bounds
    return within >= p_tp


def test_tost(ref_sample, test_sample, equivalence_margin_sd=1.5, alpha=0.05):
    """
    Two One-Sided Tests (TOST) for equivalence.
    
    Equivalence margin is specified in SD units of the reference population.
    Uses the reference sample SD to estimate the margin.
    
    Returns True if biosimilar (equivalence established).
    """
    ref_mean = np.mean(ref_sample)
    test_mean = np.mean(test_sample)
    
    n_ref = len(ref_sample)
    n_test = len(test_sample)
    
    # Pooled standard error
    ref_var = np.var(ref_sample, ddof=1)
    test_var = np.var(test_sample, ddof=1)
    se_pooled = np.sqrt(ref_var / n_ref + test_var / n_test)
    
    # Equivalence margin based on reference SD
    ref_std = np.std(ref_sample, ddof=1)
    delta = equivalence_margin_sd * ref_std
    
    # Test statistics
    diff = test_mean - ref_mean
    df = n_ref + n_test - 2
    
    # TOST: Two one-sided t-tests
    # H0: diff <= -delta  vs  H1: diff > -delta
    t1 = (diff + delta) / se_pooled
    p1 = t.cdf(t1, df)
    
    # H0: diff >= delta  vs  H1: diff < delta
    t2 = (diff - delta) / se_pooled
    p2 = t.cdf(t2, df)
    
    # Equivalence if both nulls rejected
    # For lower tail: p1 > alpha means reject H0: diff <= -delta
    # For upper tail: p2 < alpha means reject H0: diff >= delta
    # Combined: max(1-p1, p2) < alpha
    p_value = max(1 - p1, p2)
    
    return p_value < alpha


def test_minmax(ref_sample, test_sample, p_tp):
    """
    Min-Max test: Check if X% (p_tp) of test products are within 
    the min-max range of reference samples.
    
    Returns True if biosimilar.
    """
    ref_min = np.min(ref_sample)
    ref_max = np.max(ref_sample)
    
    within = np.mean((test_sample >= ref_min) & (test_sample <= ref_max))
    
    return within >= p_tp


def test_bootstrap(ref_sample, test_sample, p_tp, p_rp, n_boot=1000, alpha=0.05):
    """
    Bootstrap biosimilarity test (Zahel, 2022).
    
    Returns True if biosimilar (reject null of non-similarity).
    """
    try:
        result = biosimilarity_bootstrap_test(
            tp=test_sample,
            rp=ref_sample,
            p_tp=p_tp,
            p_rp=p_rp,
            n_boot=n_boot,
            alpha=alpha,
            random_state=None
        )
        # reject_null=True means we reject H0 (non-similarity), so products ARE similar
        return result["reject_null"]
    except Exception as e:
        # In case of numerical issues, return False (conservative)
        return False


# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================

def simulate_single_test(test_func, mean_diff, sd_ratio, n_ref, n_test, 
                         mu_ref=0, sd_ref=1, **test_params):
    """
    Simulate a single test at given parameter values.
    
    Returns True if test passes (biosimilar).
    """
    mu_test = mu_ref + mean_diff * sd_ref
    sd_test = sd_ratio * sd_ref
    
    ref_sample = np.random.normal(mu_ref, sd_ref, n_ref)
    test_sample = np.random.normal(mu_test, sd_test, n_test)
    
    return test_func(ref_sample, test_sample, **test_params)


def compute_acceptance_rate(test_func, mean_diff, sd_ratio, n_ref, n_test,
                            n_repeats, **test_params):
    """
    Compute acceptance rate at a single point in parameter space.
    """
    successes = 0
    for _ in range(n_repeats):
        if simulate_single_test(test_func, mean_diff, sd_ratio, 
                               n_ref, n_test, **test_params):
            successes += 1
    
    return successes / n_repeats


def compute_contour_grid(test_func, mean_diffs, sd_ratios, n_ref, n_test,
                         n_repeats, **test_params):
    """
    Compute acceptance rates over the entire grid.
    """
    grid = np.zeros((len(sd_ratios), len(mean_diffs)))
    
    total_points = len(mean_diffs) * len(sd_ratios)
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    
    for i, sd_ratio in enumerate(sd_ratios):
        for j, mean_diff in enumerate(mean_diffs):
            grid[i, j] = compute_acceptance_rate(
                test_func, mean_diff, sd_ratio, n_ref, n_test, 
                n_repeats, **test_params
            )
            
            # Update progress
            completed = i * len(mean_diffs) + j + 1
            progress = completed / total_points
            progress_bar.progress(progress)
            status_text.text(f"Computing grid point {completed}/{total_points}")
    
    progress_bar.empty()
    status_text.empty()
    
    return grid


def compute_boundary_acceptance_rates(test_func, n_ref, n_test, 
                                     n_repeats, p_tp, p_rp, n_boundary_points=20, **test_params):
    """
    Compute acceptance rates at exact points on the decision boundary.
    
    Returns:
        boundary_mean_diffs: array of mean differences on boundary
        boundary_sd_ratios: array of SD ratios on boundary
        boundary_rates: acceptance rates at each boundary point
    """
    z_tp = norm.ppf((1 + p_tp) / 2)  # Two-sided
    z_rp = norm.ppf((1 + p_rp) / 2)  # Two-sided
    
    # Generate boundary points
    # From test statistic: sd_ratio = (z_rp/z_tp) - mean_diff/z_tp
    # Generate more points initially since we'll filter to valid range
    boundary_mean_diffs_full = np.linspace(0, 5, n_boundary_points * 3)
    boundary_sd_ratios_full = (z_rp / z_tp) - boundary_mean_diffs_full / z_tp
    
    # Filter to valid SD ratios (positive and reasonable range)
    valid_mask = (boundary_sd_ratios_full > 0.1) & (boundary_sd_ratios_full < 2.5)
    boundary_mean_diffs_filtered = boundary_mean_diffs_full[valid_mask]
    boundary_sd_ratios_filtered = boundary_sd_ratios_full[valid_mask]
    
    # Sample n_boundary_points from the filtered set
    if len(boundary_mean_diffs_filtered) > n_boundary_points:
        indices = np.linspace(0, len(boundary_mean_diffs_filtered) - 1, n_boundary_points, dtype=int)
        boundary_mean_diffs = boundary_mean_diffs_filtered[indices]
        boundary_sd_ratios = boundary_sd_ratios_filtered[indices]
    else:
        boundary_mean_diffs = boundary_mean_diffs_filtered
        boundary_sd_ratios = boundary_sd_ratios_filtered
    
    # Compute acceptance rates at boundary points
    boundary_rates = np.zeros(len(boundary_mean_diffs))
    
    # Add p_tp and p_rp to test_params only for tests that need them
    test_params_with_probs = test_params.copy()
    test_name = test_func.__name__
    if test_name == 'test_3sd':
        test_params_with_probs['p_tp'] = p_tp
        test_params_with_probs['p_rp'] = p_rp
    elif test_name == 'test_minmax':
        test_params_with_probs['p_tp'] = p_tp
    elif test_name == 'test_bootstrap':
        test_params_with_probs['p_tp'] = p_tp
        test_params_with_probs['p_rp'] = p_rp
    # test_tost doesn't need p_tp or p_rp
    
    status_text = st.empty()
    for i, (mean_diff, sd_ratio) in enumerate(zip(boundary_mean_diffs, boundary_sd_ratios)):
        status_text.text(f"Computing boundary point {i+1}/{len(boundary_mean_diffs)}")
        boundary_rates[i] = compute_acceptance_rate(
            test_func, mean_diff, sd_ratio, n_ref, n_test, 
            n_repeats, **test_params_with_probs
        )
    
    status_text.empty()
    
    return boundary_mean_diffs, boundary_sd_ratios, boundary_rates


def is_theoretically_similar(mean_diff, sd_ratio, p_tp, p_rp):
    """
    Check if a parameter combination theoretically satisfies the similarity condition.
    
    Based on the test statistic from Zahel (2022):
    T = (σ_test/σ_ref) - (z_Y/z_X) + |μ_test - μ_ref|/(z_X·σ_ref)
    
    Products are similar if T < 0 (approximately).
    """
    z_tp = norm.ppf((1 + p_tp) / 2)  # Two-sided
    z_rp = norm.ppf((1 + p_rp) / 2)  # Two-sided
    
    # Test statistic (with σ_ref = 1, μ_ref = 0)
    test_stat = sd_ratio - (z_rp / z_tp) + abs(mean_diff) / z_tp
    
    # Similar if test statistic < 0
    return test_stat < 0


def compute_decision_boundary(p_tp, p_rp):
    """
    Compute the theoretical decision boundary for the similarity region.
    
    Based on: X% of test within Y% of reference
    For normal distributions, this gives an approximate boundary.
    """
    z_tp = norm.ppf((1 + p_tp) / 2)  # Two-sided
    z_rp = norm.ppf((1 + p_rp) / 2)  # Two-sided
    
    # Approximate boundary: relationship between mean diff and SD ratio
    # From test statistic: sd_ratio - (z_rp/z_tp) + |mean_diff|/z_tp = 0
    mean_diffs = np.linspace(0, 5, 100)
    sd_ratios = (z_rp / z_tp) - mean_diffs / z_tp
    sd_ratios = np.maximum(0.1, sd_ratios)  # Keep positive
    
    return mean_diffs, sd_ratios


def calculate_error_metrics(grid, mean_diffs, sd_ratios, p_tp, p_rp, boundary_rates=None):
    """
    Calculate Type I and Type II error from the grid.
    
    Type I error: Acceptance rate at exact boundary (if provided) or approximated from grid
    Type II error: 1 - acceptance rate INSIDE similarity region (false negative)
    
    Uses the theoretical similarity condition based on p_tp and p_rp.
    
    Parameters:
        boundary_rates: Optional array of acceptance rates at exact boundary points.
                       If provided, used for more accurate Type I error.
    """
    # Classify each grid point as inside or outside similarity region
    inside_mask = np.zeros_like(grid, dtype=bool)
    
    for i, sd_ratio in enumerate(sd_ratios):
        for j, mean_diff in enumerate(mean_diffs):
            is_similar = is_theoretically_similar(mean_diff, sd_ratio, p_tp, p_rp)
            
            if is_similar:
                inside_mask[i, j] = True
    
    # Type II error: rejection rate inside similarity region
    if np.any(inside_mask):
        type_ii_error = 1.0 - np.mean(grid[inside_mask])
    else:
        # No points inside - use origin as fallback
        closest_i = np.argmin(np.abs(sd_ratios - 1.0))
        closest_j = np.argmin(np.abs(mean_diffs - 0.0))
        type_ii_error = 1.0 - grid[closest_i, closest_j]
    
    # Type I error: acceptance rate at boundary
    if boundary_rates is not None and len(boundary_rates) > 0:
        # Use exact boundary acceptance rates
        type_i_error = np.mean(boundary_rates)
    else:
        # Fallback: approximate from grid (find boundary points)
        boundary_mask = np.zeros_like(grid, dtype=bool)
        for i in range(len(sd_ratios)):
            for j in range(len(mean_diffs)):
                if not inside_mask[i, j]:
                    # Check if any neighbor is inside
                    has_inside_neighbor = False
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < len(sd_ratios) and 0 <= nj < len(mean_diffs):
                                if inside_mask[ni, nj]:
                                    has_inside_neighbor = True
                                    break
                        if has_inside_neighbor:
                            break
                    if has_inside_neighbor:
                        boundary_mask[i, j] = True
        
        if np.any(boundary_mask):
            type_i_error = np.mean(grid[boundary_mask])
        else:
            # No boundary points found
            far_from_origin = ~inside_mask
            if np.any(far_from_origin):
                type_i_error = np.mean(grid[far_from_origin])
            else:
                type_i_error = 0.0
    
    return type_i_error, type_ii_error


# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(layout="wide", page_title="Biosimilarity Test Comparison")
    
    st.title("Biosimilarity Test Comparison - Contour Plot Analysis")
    st.markdown("---")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Simulation Parameters")
        
        st.subheader("Sample Sizes")
        n_ref = st.slider(
            "Number of Reference Batches:",
            min_value=5,
            max_value=100,
            value=20,
            step=5
        )
        
        n_test = st.slider(
            "Number of Test Batches:",
            min_value=5,
            max_value=100,
            value=20,
            step=5
        )
        
        st.markdown("---")
        st.subheader("Similarity Condition")
        
        p_tp = st.slider(
            "X%: Proportion of Test Product:",
            min_value=0.80,
            max_value=1.00,
            value=0.99,
            step=0.01,
            format="%.2f",
            help="X% of test product must be within Y% of reference"
        )
        
        p_rp = st.slider(
            "Y%: Proportion of Reference Range:",
            min_value=0.80,
            max_value=1.00,
            value=0.99,
            step=0.01,
            format="%.2f",
            help="Defines the reference range (Y% quantile)"
        )
        
        st.markdown("---")
        st.subheader("Computation Settings")
        
        grid_size = st.slider(
            "Grid Size (points per dimension):",
            min_value=5,
            max_value=50,
            value=15,
            step=5,
            help="Higher values = more detail but slower computation"
        )
        
        n_repeats = st.slider(
            "Simulations per Grid Point:",
            min_value=50,
            max_value=1000,
            value=200,
            step=50,
            help="Number of simulations to estimate acceptance rate"
        )
        
        n_bootstrap = st.slider(
            "Bootstrap Samples (for Bootstrap Test):",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help="Number of bootstrap resamples"
        )
        
        st.markdown("---")
        st.subheader("TOST Settings")
        
        tost_margin = st.slider(
            "TOST Equivalence Margin (× SD_ref):",
            min_value=0.5,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="Equivalence margin as multiple of reference SD"
        )
        
        st.markdown("---")
        st.subheader("Optional Tests")
        
        include_bootstrap = st.checkbox(
            "Include Bootstrap Test",
            value=True,
            help="Bootstrap test is computationally intensive. Uncheck to skip it for faster results."
        )
        
        st.markdown("---")
        compute_button = st.button(
            "🚀 Run Comparison",
            type="primary",
            use_container_width=True
        )
    
    # Main display
    if not compute_button:
        st.info("👈 Configure parameters in the sidebar and click **Run Comparison** to start.")
        
        st.markdown("""
        ### About this Application
        
        This tool compares four different biosimilarity tests by computing acceptance rates
        across the parameter space of:
        - **Mean difference** (normalized by reference SD) on X-axis
        - **SD ratio** (Test SD / Reference SD) on Y-axis
        
        #### The Four Tests:
        
        1. **3SD Test**: Classic range-based test
        2. **TOST**: Two One-Sided Tests for equivalence
        3. **Min-Max Test**: Test range within reference min-max
        4. **Bootstrap Test**: Novel bootstrap-based test (Zahel, 2022)
        
        #### Output:
        
        - **Contour plots**: Show acceptance rate (0-100%) across parameter space
        - **Type I Error**: False positive rate (accepting non-similar products)
        - **Type II Error**: False negative rate (rejecting similar products)
        
        #### Similarity Condition:
        
        Set X% and Y% to define: "X% of test product must be within Y% of reference product"
        """)
        
        return
    
    # ========================================================================
    # COMPUTATION
    # ========================================================================
    
    st.header("Computing Acceptance Rates...")
    
    # Define parameter grid
    mean_diffs = np.linspace(0, 5, grid_size)
    sd_ratios = np.linspace(0.1, 2.5, grid_size)
    
    # Container for results
    results = {}
    boundary_results = {}
    
    # Compute grids for each test
    st.subheader("1️⃣ Computing 3SD Test...")
    results['3SD'] = compute_contour_grid(
        test_3sd, mean_diffs, sd_ratios, n_ref, n_test, n_repeats,
        p_tp=p_tp, p_rp=p_rp
    )
    st.text("Computing boundary points for 3SD test...")
    boundary_results['3SD'] = compute_boundary_acceptance_rates(
        test_3sd, n_ref, n_test, n_repeats, p_tp, p_rp,
        n_boundary_points=20
    )
    
    st.subheader("2️⃣ Computing TOST...")
    results['TOST'] = compute_contour_grid(
        test_tost, mean_diffs, sd_ratios, n_ref, n_test, n_repeats,
        equivalence_margin_sd=tost_margin, alpha=0.05
    )
    st.text("Computing boundary points for TOST...")
    boundary_results['TOST'] = compute_boundary_acceptance_rates(
        test_tost, n_ref, n_test, n_repeats, p_tp, p_rp,
        n_boundary_points=20, equivalence_margin_sd=tost_margin, alpha=0.05
    )
    
    st.subheader("3️⃣ Computing Min-Max Test...")
    results['MinMax'] = compute_contour_grid(
        test_minmax, mean_diffs, sd_ratios, n_ref, n_test, n_repeats,
        p_tp=p_tp
    )
    st.text("Computing boundary points for Min-Max test...")
    boundary_results['MinMax'] = compute_boundary_acceptance_rates(
        test_minmax, n_ref, n_test, n_repeats, p_tp, p_rp,
        n_boundary_points=20
    )
    
    if include_bootstrap:
        st.subheader("4️⃣ Computing Bootstrap Test...")
        results['Bootstrap'] = compute_contour_grid(
            test_bootstrap, mean_diffs, sd_ratios, n_ref, n_test, n_repeats,
            p_tp=p_tp, p_rp=p_rp, n_boot=n_bootstrap, alpha=0.05
        )
        st.text("Computing boundary points for Bootstrap test...")
        boundary_results['Bootstrap'] = compute_boundary_acceptance_rates(
            test_bootstrap, n_ref, n_test, n_repeats, p_tp, p_rp,
            n_boundary_points=20, n_boot=n_bootstrap, alpha=0.05
        )
    else:
        st.info("⏭️ Bootstrap test skipped (enable in sidebar to include)")
        results['Bootstrap'] = None
        boundary_results['Bootstrap'] = None
    
    st.success("✅ All computations complete!")
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    st.markdown("---")
    st.header("Contour Plot Comparison")
    
    # Determine which tests to plot
    test_names = ['3SD', 'TOST', 'MinMax']
    test_titles = [
        f'3SD Test ({p_rp*100:.0f}% → {norm.ppf((1+p_rp)/2):.2f} SD)',
        f'TOST Test (Margin = {tost_margin:.1f} × SD_ref)',
        f'Min-Max Test ({p_tp*100:.0f}% within sample range)'
    ]
    
    if include_bootstrap:
        test_names.append('Bootstrap')
        test_titles.append(f'Bootstrap Test (Zahel 2022) ({n_bootstrap} bootstrap samples)')
    
    # Create subplot grid
    n_tests = len(test_names)
    if n_tests == 4:
        rows, cols = 2, 2
    else:
        rows, cols = 1, 3
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=test_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # Add overall title
    fig.update_layout(
        title={
            'text': f'Biosimilarity Test Comparison<br>Similarity Condition: {p_tp*100:.0f}% of Test within {p_rp*100:.0f}% of Reference<br>n_ref={n_ref}, n_test={n_test}, {n_repeats} simulations/point',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 14}
        },
        height=800 if n_tests == 4 else 500,
        showlegend=False
    )
    
    # Plot each test
    for idx, (test_name, title) in enumerate(zip(test_names, test_titles)):
        grid = results[test_name]
        
        if grid is None:
            continue
        
        row = (idx // cols) + 1
        col = (idx % cols) + 1
        
        # Add contour plot (with colorbar only for the first plot in each row for cleaner layout)
        show_colorbar = (col == cols)  # Show colorbar on rightmost column only
        fig.add_trace(
            go.Contour(
                x=mean_diffs,
                y=sd_ratios,
                z=grid * 100,
                colorscale='RdYlGn',
                contours=dict(
                    start=0,
                    end=100,
                    size=5,
                    showlines=True,
                    showlabels=True
                ),
                colorbar=dict(
                    title='Acceptance<br>Rate (%)',
                    len=0.4 if n_tests == 4 else 0.9,
                    y=0.75 - (row - 1) * 0.5 if n_tests == 4 else 0.5,
                ) if show_colorbar else None,
                showscale=show_colorbar,
                hovertemplate='Mean Diff: %{x:.2f}<br>SD Ratio: %{y:.2f}<br>Acceptance: %{z:.1f}%<extra></extra>',
                name=''
            ),
            row=row, col=col
        )
        
        # Add bold 5% contour line
        fig.add_trace(
            go.Contour(
                x=mean_diffs,
                y=sd_ratios,
                z=grid * 100,
                contours=dict(
                    start=5,
                    end=5,
                    size=1,
                    showlines=True,
                    coloring='lines'
                ),
                line=dict(color='red', width=3),
                showscale=False,
                hoverinfo='skip',
                name='5% Isoline'
            ),
            row=row, col=col
        )
        
        # Add decision boundary
        bound_x, bound_y = compute_decision_boundary(p_tp, p_rp)
        valid_mask = (bound_y >= sd_ratios.min()) & (bound_y <= sd_ratios.max())
        fig.add_trace(
            go.Scatter(
                x=bound_x[valid_mask],
                y=bound_y[valid_mask],
                mode='lines',
                line=dict(color='blue', width=3, dash='dash'),
                name='Decision Boundary',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row, col=col
        )
        
        # Plot boundary points if available (commented out - not needed for display)
        # if boundary_results[test_name] is not None:
        #     bnd_mean, bnd_sd, bnd_rates = boundary_results[test_name]
        #     fig.add_trace(
        #         go.Scatter(
        #             x=bnd_mean,
        #             y=bnd_sd,
        #             mode='markers',
        #             marker=dict(color='blue', size=8, symbol='x', line=dict(width=2)),
        #             name='Boundary Points',
        #             hovertemplate='Mean Diff: %{x:.2f}<br>SD Ratio: %{y:.2f}<extra></extra>'
        #         ),
        #         row=row, col=col
        #     )
        
        # Calculate error metrics
        boundary_rates_for_calc = boundary_results[test_name][2] if boundary_results[test_name] is not None else None
        type_i, type_ii = calculate_error_metrics(
            grid, mean_diffs, sd_ratios, p_tp, p_rp, boundary_rates=boundary_rates_for_calc
        )
        
        # Add error metrics as annotation
        fig.add_annotation(
            x=0.02, y=0.98,
            xref=f'x{idx+1} domain' if idx > 0 else 'x domain',
            yref=f'y{idx+1} domain' if idx > 0 else 'y domain',
            text=f'Type I error at decision boundary: {type_i*100:.1f}%<br>Type II Error: {type_ii*100:.1f}%',
            showarrow=False,
            bgcolor='white',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=10),
            align='left',
            xanchor='left',
            yanchor='top'
        )
        
        # Update axes
        fig.update_xaxes(title_text='Mean Difference (× SD_ref)', row=row, col=col, gridcolor='lightgray')
        fig.update_yaxes(title_text='SD Ratio (SD_test / SD_ref)', row=row, col=col, gridcolor='lightgray')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    
    st.markdown("---")
    st.header("Summary Statistics")
    
    cols = st.columns(len(test_names))
    
    for col, test_name in zip(cols, test_names):
        with col:
            st.subheader(test_name)
            grid = results[test_name]
            
            if grid is None:
                st.info("Test not run")
                continue
            
            boundary_rates_for_calc = boundary_results[test_name][2] if boundary_results[test_name] is not None else None
            type_i, type_ii = calculate_error_metrics(
                grid, mean_diffs, sd_ratios, p_tp, p_rp, boundary_rates=boundary_rates_for_calc
            )
            
            st.metric("Mean Acceptance Rate", f"{np.mean(grid)*100:.1f}%")
            st.metric("Type I Error (α)", f"{type_i*100:.1f}%")
            st.metric("Type II Error (β)", f"{type_ii*100:.1f}%")
            st.metric("Power (1-β)", f"{(1-type_ii)*100:.1f}%")
    
    # ========================================================================
    # TEST EXPLANATIONS
    # ========================================================================
    
    st.markdown("---")
    st.header("Test Descriptions")
    
    st.markdown(f"""
    ### How Each Test Works
    
    **Current Similarity Condition:** {p_tp*100:.0f}% of test product must be within {p_rp*100:.0f}% of reference product
    
    ---
    
    #### 1️⃣ 3SD Test (Range-Based Test)
    
    **Principle:** Classic approach using standard deviations to define acceptance range.
    
    **Implementation:**
    - Convert Y% ({p_rp*100:.0f}%) to z-score: z = {norm.ppf((1+p_rp)/2):.3f} SD
    - Calculate reference range: [μ_ref - z·σ_ref, μ_ref + z·σ_ref]
    - Count proportion of test samples within this range
    - **Accept if** ≥ {p_tp*100:.0f}% of test samples are within the range
    
    **Characteristics:**
    - Simple and intuitive
    - Based on sample statistics (mean and SD)
    - No formal hypothesis testing
    - Does not control Type I error rate
    
    ---
    
    #### 2️⃣ TOST (Two One-Sided Tests)
    
    **Principle:** Statistical equivalence testing using confidence intervals.
    
    **Implementation:**
    - Equivalence margin: δ = {tost_margin:.1f} × σ_ref (independent of X% and Y% sliders)
    - Null hypothesis: |μ_test - μ_ref| ≥ δ (products are different)
    - Perform two one-sided t-tests:
      - H0₁: μ_test - μ_ref ≤ -δ vs H1₁: μ_test - μ_ref > -δ
      - H0₂: μ_test - μ_ref ≥ δ vs H1₂: μ_test - μ_ref < δ
    - **Accept if** both null hypotheses are rejected at α = 0.05
    
    **Characteristics:**
    - Gold standard for equivalence testing
    - Controls Type I error at 5%
    - Based on formal statistical theory
    - Requires larger sample sizes for adequate power
    
    ---
    
    #### 3️⃣ Min-Max Test
    
    **Principle:** Simple range test using observed min and max of reference.
    
    **Implementation:**
    - Calculate reference range: [min(reference samples), max(reference samples)]
    - Count proportion of test samples within this range
    - **Accept if** ≥ {p_tp*100:.0f}% of test samples are within [min, max]
    - Note: Only uses X% parameter; Y% is not applicable
    
    **Characteristics:**
    - Very simple to implement
    - Conservative (tight range)
    - Sensitive to outliers in reference
    - Sample-dependent (not based on population parameters)
    
    ---
    
    #### 4️⃣ Bootstrap Test (Zahel 2022)
    
    **Principle:** Novel bootstrap-based approach directly testing the similarity condition.
    
    **Implementation:**
    - Uses both X% ({p_tp*100:.0f}%) and Y% ({p_rp*100:.0f}%) directly
    - Convert percentiles to z-scores: z_X = {norm.ppf(p_tp):.3f}, z_Y = {norm.ppf(p_rp):.3f}
    - Test statistic: T = (σ_test/σ_ref) - (z_Y/z_X) + |μ_test - μ_ref|/(z_X·σ_ref)
    - Bootstrap resampling ({n_bootstrap} samples) to estimate null distribution
    - **Accept if** p-value < 0.05 (reject H0: products are not similar)
    
    **Characteristics:**
    - Directly tests the "{p_tp*100:.0f}% within {p_rp*100:.0f}%" condition
    - Controls Type I error through bootstrap p-value
    - Accounts for uncertainty in both mean and variance
    - More computationally intensive
    - Specifically designed for analytical biosimilarity
    
    ---
    
    ### Interpretation
    
    - **Green regions**: High acceptance rate (products declared similar)
    - **Red regions**: Low acceptance rate (products declared different)
    - **Type I Error (α)**: Probability of accepting when products are at the boundary (false positive)
    - **Type II Error (β)**: Probability of rejecting when products are truly similar (false negative)
    - **Power (1-β)**: Probability of correctly accepting truly similar products
    
    **Ideal test characteristics:**
    - Low Type I error (< 5%) at decision boundary
    - Low Type II error (high power) within similarity region
    - Clear separation between acceptance and rejection regions
    """)
    
    st.markdown("---")
    st.caption(f"Generated on {st.session_state.get('timestamp', 'unknown')}")


if __name__ == "__main__":
    if 'timestamp' not in st.session_state:
        from datetime import datetime
        st.session_state.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    main()
