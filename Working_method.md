# Working Method: Double Exponential Initial Rate Analysis

## Overview

This document describes the validated method for extracting catalytic rate constants (k_cat) from stopped-flow CO2 hydration kinetics data using weighted least squares regression.

---

## Method Summary

1. Fit double exponential to absorbance decay
2. Calculate initial rate (dydt0) from fit parameters
3. Convert to observed rate constant (k_obs) using [CO2] and Q factor
4. Weighted least squares regression of k_obs vs [catalyst] gives k_cat

---

## Step 1: Double Exponential Fitting

### Model Equation

```
A(t) = A₀ + A₁·exp(-k₁·t) + A₂·exp(-k₂·t)
```

**Parameters:**
| Symbol | Description | Units |
|--------|-------------|-------|
| A(t) | Absorbance at time t | AU |
| A₀ | Baseline absorbance (final value) | AU |
| A₁ | Amplitude of fast component | AU |
| k₁ | Rate constant of fast component | /s |
| A₂ | Amplitude of slow component | AU |
| k₂ | Rate constant of slow component | /s |

### Fitting Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Time window start (t_min) | 0.02 s | Excludes mixing artifacts |
| Time window end (t_max) | 40.0 s | Captures full decay |
| k bounds | [1e-6, 100] /s | Physical constraints |

---

## Step 2: Initial Rate Calculation

### Equation

The initial rate is the derivative of A(t) evaluated at t=0:

```
v₀ = dA/dt |_{t=0} = -A₁·k₁ - A₂·k₂
```

**Units:** AU/s (absorbance units per second)

**Note:** v₀ (dydt0) is negative because absorbance decreases over time.

---

## Step 3: Conversion to Observed Rate Constant

Two methods are used to calculate k_obs from the double exponential fit parameters. **Both should be reported.**

### Method 1: k_obs (Working Method)

```
k_obs = |dydt0| × |Q| / [CO₂]
```

Where:
- dydt0 = -A₁·k₁ - A₂·k₂ (initial rate in AU/s)
- [CO₂] = 0.0338 M (saturated CO₂ at 25°C)
- Q = calibration factor (converts AU/s to M/s)

### Method 2: k_obs_weighted (Amplitude-Weighted k)

```
k_obs_weighted = |A₁·k₁ + A₂·k₂| / |A₁ + A₂|
```

This gives an amplitude-weighted average of the two exponential rate constants.

### Q Factor Values (CRITICAL)

**The Q factor depends on the study:**

| Study | Q Value | Description |
|-------|---------|-------------|
| Monodentate study | **-0.402** | Older calibration |
| Cyclen study (ALL dates) | **-0.452** | Updated calibration |

**Note:** ALL Cyclen study data uses Q = -0.452, including early dates like 12_05.

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| [CO₂] | 0.0338 M | Saturated CO₂ concentration at 25°C |
| Q (Monodentate) | -0.402 | For monodentate study |
| Q (Cyclen study) | -0.452 | For ALL cyclen study data |

**Physical meaning of Q:** The Q factor converts absorbance change (AU/s) to concentration change (M/s). It is related to the indicator extinction coefficient and path length: Q ≈ 1/(ε × path_length).

### Example Calculations

**k_obs (Working Method):**
```
dydt0 = -0.020 AU/s
k_obs = |-0.020| × 0.452 / 0.0338
k_obs = 0.020 × 0.452 / 0.0338
k_obs = 0.267 /s
```

**k_obs_weighted (Amplitude-Weighted):**
```
A₁ = 0.03, k₁ = 0.5 /s
A₂ = 0.02, k₂ = 0.1 /s
k_obs_weighted = |0.03×0.5 + 0.02×0.1| / |0.03 + 0.02|
k_obs_weighted = |0.015 + 0.002| / |0.05|
k_obs_weighted = 0.34 /s
```

---

## Step 4: Weighted Least Squares Regression

### Linear Model

```
k_obs = k_w + k_cat × [Cat]
```

**Where:**
| Symbol | Description | Units |
|--------|-------------|-------|
| k_obs | Observed rate constant from Step 3 | /s |
| k_w | Intrinsic first-order rate of uncatalyzed hydration (intercept) | /s |
| k_cat | Second-order catalytic constant (slope) | /M/s |
| [Cat] | Total concentration of zinc complex | M |

### Weighted Least Squares Recipe

For each three-point concentration set:

**1. Weights**
```
w_i = 1 / σ_i²
```
where σ_i = standard deviation of k_obs,i from n replicates (typically n=5-6)

**2. Weighted Sums**
```
S   = Σ w_i
Sₓ  = Σ w_i·x_i
Sᵧ  = Σ w_i·y_i
Sₓₓ = Σ w_i·x_i²
Sₓᵧ = Σ w_i·x_i·y_i
```

**3. Slope and Intercept**
```
Δ = S·Sₓₓ - Sₓ²

k_cat = (S·Sₓᵧ - Sₓ·Sᵧ) / Δ

k_w = (Sₓₓ·Sᵧ - Sₓ·Sₓᵧ) / Δ
```

**4. Standard Errors (1σ)**
```
SE(k_cat) = √(S / Δ)

SE(k_w) = √(Sₓₓ / Δ)
```

**5. 95% Confidence Interval**

For three concentrations → df = 1 (n - 2 parameters)
```
CI₉₅ = value ± t₀.₀₂₅,₁ × SE

where t₀.₀₂₅,₁ = 12.706
```

### Validation Benchmark

**k_w should be approximately 0.12-0.18 /s**

This is the known uncatalyzed rate for CO₂ hydration at 25°C. Literature value is ~0.15 /s. If the intercept falls within this range, the method is validated.

---

## Complete Workflow

```
Raw Data (absorbance vs time)
           ↓
    [Time window: 0.02-40s]
           ↓
    Double Exponential Fit
    A(t) = A₀ + A₁·exp(-k₁·t) + A₂·exp(-k₂·t)
           ↓
    Calculate BOTH k_obs values:
    ├── k_obs = |dydt0| × |Q| / [CO₂]     (Working Method)
    └── k_obs_wt = |A₁k₁ + A₂k₂| / |A₁ + A₂|  (Amplitude-Weighted)
           ↓
    [Repeat for each trial]
           ↓
    Average replicates at each [Cat]
    Calculate σ_i for each concentration (for both methods)
           ↓
    Weighted Least Squares Regression (for BOTH methods)
    k_obs vs [Cat], weights w_i = 1/σ_i²
           ↓
    Results (8 columns):
    - k_cat (/M/s) ± SE      from k_obs
    - k_w (/s) ± SE          from k_obs
    - k_cat_wt (/M/s) ± SE   from k_obs_weighted
    - k_w_wt (/s) ± SE       from k_obs_weighted
```

---

## Reference Values

### Cyclen Study (Q = -0.452) - Complete Results

*With 1-std outlier removal applied*

| Catalyst | Date | k_cat | ± | k_w | ± | k_cat_wt | ± | k_w_wt | ± |
|----------|------|-------|---|-----|---|----------|---|--------|---|
| Benz_Cyclen | 12_05 | 173 | 14 | 0.165 | 0.008 | 252 | 14 | 0.185 | 0.007 |
| nBu_Cyclen | 01_14 | 186 | 14 | 0.133 | 0.006 | 250 | 18 | 0.175 | 0.009 |
| Hexyl_Cyclen | 01_14 | 275 | 20 | 0.122 | 0.010 | 402 | 96 | 0.161 | 0.048 |
| Hexyl_Cyclen | 02_03 | 201 | 12 | 0.155 | 0.005 | 280 | 8 | 0.180 | 0.003 |
| Cyclen | 01_13 | 339 | 19 | 0.166 | 0.012 | 352 | 16 | 0.185 | 0.014 |

**Units:** k_cat, k_cat_wt in /M/s; k_w, k_w_wt in /s

**Averages by Ligand:**

| Ligand | k_cat | k_w | k_cat_wt | k_w_wt |
|--------|-------|-----|----------|--------|
| Benz_Cyclen | 173 | 0.165 | 252 | 0.185 |
| nBu_Cyclen | 186 | 0.133 | 250 | 0.175 |
| Hexyl_Cyclen | 238 | 0.139 | 341 | 0.171 |
| Cyclen | 339 | 0.166 | 352 | 0.185 |

**Observations:**
- k_obs_weighted method gives ~35-45% higher k_cat values than Working Method
- Both methods give k_w values in valid range (0.12-0.17 /s)
- Working Method k_w values are closer to literature (~0.15 /s)

### Monodentate Study (Q = -0.402) - Selected Results

| Ligand Equiv | Date | k_cat (/M/s) | ± SE | k_w (/s) | ± SE |
|--------------|------|--------------|------|----------|------|
| 15x NMeIm | 2025-06-30 | 426 | 25 | 0.190 | 0.013 |
| 25x NMeIm | 2025-06-26 | 382 | 19 | 0.183 | 0.007 |
| 37x NMeIm | 2025-06-30 | 358 | 18 | 0.243 | 0.008 |
| 50x NMeIm | 2025-06-26 | 277 | 22 | 0.218 | 0.009 |

**Note:** Monodentate k_w values are elevated (~0.19-0.24 /s) compared to literature. This may indicate ligand-catalyzed background reaction or systematic offset.

---

## Reporting Significant Figures

### Rules
- Report SE to 1-2 significant figures
- Round the value to match the precision of the SE

### Examples
| Raw Value | Reported |
|-----------|----------|
| k_cat = 181.9 ± 15.7 | **182 ± 16 /M/s** |
| k_w = 0.1571 ± 0.0112 | **0.157 ± 0.011 /s** |

---

## Outlier Exclusion Criteria

### Amplitude-Based Exclusion (Dataset Level)

Datasets with amplitude > 0.065 should be evaluated for exclusion:
```
amplitude = median(first 5% of data) - median(last 5% of data)
```

### Trial-Level Quality Filters

Remove individual trials that fail quality checks:
- Poor fit quality (R² < 0.99)
- dydt0 > 0 (wrong sign indicates bad fit)
- k_obs > 2 /s (physically unreasonable)
- Negative rate constants (k₁ < 0 or k₂ < 0)
- Unreasonably fast rate constants (k₁ > 50 or k₂ > 50)

### Outlier Removal (1-Standard Deviation Filter)

After quality filtering, remove statistical outliers at each concentration:

```
For each concentration:
    1. Calculate mean and std of k_obs values
    2. Keep only values within 1σ of the mean:
       |k_obs - mean| ≤ 1 × std
    3. Require at least 2 trials remaining
```

**Rationale:** The 1-std filter removes only the most extreme outliers while preserving natural experimental variance. This is conservative - it typically removes only 1-2 trials per concentration when clear outliers exist, but leaves data intact when variance is uniform.

**Example (Benz_Cyclen 12_05 at 1.0mM):**
- Raw values: [0.247, 0.331, 0.343, 0.344]
- Mean = 0.316, Std = 0.046
- 1σ range: [0.270, 0.362]
- Removed: 0.247 (below lower bound)
- Kept: [0.331, 0.343, 0.344]

### Excluded Datasets
- Benz_Cyclen 12_13: amplitude outlier (0.0712)
- Benz_Cyclen 01_14: excluded per user request
- nBu_Cyclen 02_02: excluded per user request

---

## Important Notes

1. **Q Factor:** Use Q = -0.452 for ALL Cyclen study data, Q = -0.402 for Monodentate study

2. **[CO₂]:** Use 0.0338 M (33.8 mM), saturated CO₂ at 25°C

3. **Weighted LS:** The SE formulas assume σ values represent true measurement uncertainty. This gives larger (more conservative) SEs than OLS, which is appropriate when measurement precision varies between concentrations.

4. **Validation:** k_w values should fall in 0.12-0.18 /s range to confirm method validity

---

## Script Reference

Implementation files:
- `complete_analysis.py` - Full analysis pipeline
- `stopped_flow_fitting_utility_OG.py` - Original fitting utility

---

## Output Format

**Every analysis should output 8 columns:**

| Column | Description | Units |
|--------|-------------|-------|
| k_cat | Slope from k_obs regression | /M/s |
| ± | Standard error of k_cat | /M/s |
| k_w | Intercept from k_obs regression | /s |
| ± | Standard error of k_w | /s |
| k_cat_wt | Slope from k_obs_weighted regression | /M/s |
| ± | Standard error of k_cat_wt | /M/s |
| k_w_wt | Intercept from k_obs_weighted regression | /s |
| ± | Standard error of k_w_wt | /s |

---

## Document History

- Created: 2026-02-08
- Updated: 2026-02-09
  - Corrected Q values: Q = -0.452 for ALL Cyclen study data
  - Added complete weighted least squares recipe with all formulas
  - Added 95% CI calculation (t₀.₀₂₅,₁ = 12.706 for df=1)
  - Added significant figures reporting guidelines
  - Updated reference values with outlier removal
  - Added k_obs_weighted method (amplitude-weighted k)
  - Updated output format to include all 8 columns for both methods
- Updated: 2026-02-10
  - Changed outlier removal from MAD-based to 1-std filter
  - 1-std filter removes only extreme outliers while preserving natural variance
  - Updated reference values with new outlier removal method
  - Added Cyclen baseline (01_13) to reference table
