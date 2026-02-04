import numpy as np
from scipy.stats import norm

def biosimilarity_bootstrap_test(
    tp,
    rp,
    p_tp=0.99,
    p_rp=0.99,
    n_boot=1000,
    alpha=0.05,
    random_state=None
):
    """
    Bootstrap test for analytical biosimilarity (Zahel, 2022).

    Parameters
    ----------
    tp : array-like
        Test product (TP) batch data
    rp : array-like
        Reference product (RP) batch data
    p_tp : float
        Fraction of TP distribution required within RP (default 0.99)
    p_rp : float
        Fraction of RP distribution defining the range (default 0.99)
    n_boot : int
        Number of bootstrap resamples
    alpha : float
        Significance level
    random_state : int or None
        RNG seed

    Returns
    -------
    dict with keys:
        p_value
        reject_null
        rejection_rate
        test_statistic_observed
    """

    rng = np.random.default_rng(random_state)

    tp = np.asarray(tp)
    rp = np.asarray(rp)

    n_tp = len(tp)
    n_rp = len(rp)

    # Quantiles of standard normal (two-sided)
    z_tp = norm.ppf((1 + p_tp) / 2)
    z_rp = norm.ppf((1 + p_rp) / 2)

    def test_statistic(mu_tp, sd_tp, mu_rp, sd_rp):
        return (
            sd_tp / sd_rp
            - (z_rp / z_tp)
            + abs(mu_tp - mu_rp) / (z_tp * sd_rp)
        )

    # Observed statistic
    mu_tp_obs = tp.mean()
    sd_tp_obs = tp.std(ddof=1)
    mu_rp_obs = rp.mean()
    sd_rp_obs = rp.std(ddof=1)

    t_obs = test_statistic(mu_tp_obs, sd_tp_obs, mu_rp_obs, sd_rp_obs)

    # Bootstrap
    reject_count = 0

    for _ in range(n_boot):
        tp_b = rng.choice(tp, size=n_tp, replace=True)
        rp_b = rng.choice(rp, size=n_rp, replace=True)

        mu_tp_b = tp_b.mean()
        sd_tp_b = tp_b.std(ddof=1)
        mu_rp_b = rp_b.mean()
        sd_rp_b = rp_b.std(ddof=1)

        t_b = test_statistic(mu_tp_b, sd_tp_b, mu_rp_b, sd_rp_b)

        # Reject H0 if statistic < 0
        if t_b < 0:
            reject_count += 1

    rejection_rate = reject_count / n_boot
    p_value = 1 - rejection_rate

    return {
        "p_value": p_value,
        "reject_null": p_value < alpha,
        "rejection_rate": rejection_rate,
        "test_statistic_observed": t_obs
    }
