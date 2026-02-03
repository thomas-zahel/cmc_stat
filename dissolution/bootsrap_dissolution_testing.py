
import numpy as np
import pandas as pd
from math import log10
from scipy.stats import norm

# ---------- f2 and E(f2) ----------
def f2_stat(ref, test):
    R = ref.mean(axis=0)
    T = test.mean(axis=0)
    msd = np.mean((T - R)**2)
    return 100.0 - 25.0 * log10(1.0 + msd)

def Ef2_stat(ref, test):
    R = ref.mean(axis=0)
    T = test.mean(axis=0)
    n = ref.shape[0]
    sR2 = ref.var(axis=0, ddof=1)
    sT2 = test.var(axis=0, ddof=1)
    term = np.mean((T - R)**2 + (sR2 + sT2) / n)
    return 100.0 - 25.0 * log10(1.0 + term)

# ---------- Bootstrap helpers ----------
def bootstrap_f2(ref, test, B=1000, rng=None):
    if rng is None:
        rng = np.random.default_rng(2026)
    n = ref.shape[0]
    out = np.empty(B)
    for b in range(B):
        idx_r = rng.integers(0, n, size=n)
        idx_t = rng.integers(0, n, size=n)
        out[b] = f2_stat(ref[idx_r, :], test[idx_t, :])
    return out

def bootstrap_Ef2(ref, test, B=1000, rng=None):
    if rng is None:
        rng = np.random.default_rng(2026)
    n = ref.shape[0]
    out = np.empty(B)
    for b in range(B):
        idx_r = rng.integers(0, n, size=n)
        idx_t = rng.integers(0, n, size=n)
        out[b] = Ef2_stat(ref[idx_r, :], test[idx_t, :])
    return out

def jackknife_f2(ref, test):
    n = ref.shape[0]
    vals = []
    for i in range(n):
        vals.append(f2_stat(np.delete(ref, i, axis=0), test))
    for i in range(n):
        vals.append(f2_stat(ref, np.delete(test, i, axis=0)))
    return np.array(vals)

def percentile_ci(vals, alpha=0.10):
    vals = np.asarray(vals)
    return float(np.quantile(vals, alpha/2)), float(np.quantile(vals, 1 - alpha/2))

def bca_ci(boot_vals, t0, jack_vals, alpha=0.10):
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

batches = {'Test1': T1, 'Test2': T2, 'Test3': T3, 'Test4': T4, 'Test5': T5}

# ---------- Run comparisons ----------
replicates_list = [500, 1000, 5000]
rows = []
for name, T in batches.items():
    t0_f2 = f2_stat(ref, T)
    t0_Ef2 = Ef2_stat(ref, T)
    jack = jackknife_f2(ref, T)
    for B in replicates_list:
        rng1 = np.random.default_rng(2026 + B)
        rng2 = np.random.default_rng(3030 + B)
        boot_f2  = bootstrap_f2(ref, T, B=B, rng=rng1)
        boot_Ef2 = bootstrap_Ef2(ref, T, B=B, rng=rng2)
        L_pi, U_pi   = percentile_ci(boot_Ef2, alpha=0.10)  # 90% PI(E(f2))
        L_bca, U_bca = bca_ci(boot_f2, t0_f2, jack, alpha=0.10)  # 90% BCa(f2)
        rows.append({
            'Batch': name, 'B': B,
            'Point_f2': round(t0_f2, 2), 'Point_Ef2': round(t0_Ef2, 2),
            'PI_Ef2_L': round(L_pi, 2),  'PI_Ef2_U': round(U_pi, 2),
            'BCa_f2_L': round(L_bca, 2), 'BCa_f2_U': round(U_bca, 2),
        })

result_df = pd.DataFrame(rows)
print(result_df.to_string(index=False))
``
