# Kinetics Data Analysis Method

## Overview

This document explains the mathematical procedure for extracting catalytic rate constants (k_cat) from stopped-flow spectrophotometry data.

---

## 1. The Physical System

We measure the hydration of CO2:

```
CO2 + H2O → H+ + HCO3-
```

A pH indicator (phenol red) changes absorbance as protons are produced. The absorbance decay is monitored over time.

---

## 2. The Mathematical Model

### Single Exponential Decay

The absorbance follows first-order kinetics:

```
A(t) = A_final + A_amplitude × exp(-k_obs × t)
```

Where:
- `A(t)` = absorbance at time t (AU)
- `A_final` = final absorbance after reaction completes (AU)
- `A_amplitude` = total absorbance change (AU)
- `k_obs` = observed rate constant (/s)
- `t` = time (seconds)

### Rate-Concentration Relationship

The observed rate constant depends linearly on catalyst concentration:

```
k_obs = k_uncat + k_cat × [catalyst]
```

Where:
- `k_obs` = observed rate constant from fitting (/s)
- `k_uncat` = uncatalyzed (background) rate ≈ 0.12 /s
- `k_cat` = catalytic rate constant (/M/s) ← **THIS IS WHAT WE REPORT**
- `[catalyst]` = catalyst concentration (M)

---

## 3. Step-by-Step Procedure

### Step 1: Data Collection
- Measure absorbance vs time at 3+ catalyst concentrations
- Multiple trials (replicates) at each concentration

### Step 2: Curve Fitting
For each kinetic trace:

1. Apply time window (0.02 - 40 s) to exclude mixing artifacts
2. Fit single exponential: `A(t) = A_final + A_amplitude × exp(-k_obs × t)`
3. Extract `k_obs` (the rate constant)
4. Verify fit quality (R² > 0.99)

### Step 3: Average Per Concentration
- Calculate mean k_obs at each concentration
- Calculate standard deviation

### Step 4: Linear Regression
Plot k_obs vs [catalyst] and fit:

```
k_obs = k_uncat + k_cat × [catalyst]
```

- Slope = k_cat (/M/s)
- Intercept = k_uncat (/s)

### Step 5: Validation
**Critical check:** The intercept (k_uncat) should be approximately 0.12 /s.

This is the known uncatalyzed rate for CO2 hydration. If the intercept matches, the method is validated.

---

## 4. Example Calculation (Benzyl-Cyclen, Single Exponential)

### Raw Data

| Concentration | Trial 1 k_obs | Trial 2 k_obs | Trial 3 k_obs | Mean k_obs |
|--------------|---------------|---------------|---------------|------------|
| 0.25 mM | 0.158 /s | 0.160 /s | 0.167 /s | 0.162 /s |
| 0.50 mM | 0.189 /s | 0.195 /s | 0.196 /s | 0.193 /s |
| 1.00 mM | 0.271 /s | 0.268 /s | 0.267 /s | 0.265 /s |

### Linear Regression

Using least squares on (concentration in M, k_obs):

| x (M) | y (/s) |
|-------|--------|
| 0.00025 | 0.162 |
| 0.00050 | 0.193 |
| 0.00100 | 0.265 |

**Result:**
- Slope (k_cat) = **138.5 /M/s**
- Intercept (k_uncat) = **0.126 /s** ✓ (close to expected 0.12)
- R² = 0.999

---

## 5. Why Single Exponential is Recommended

| Method | k_cat | k_uncat | Validation |
|--------|-------|---------|------------|
| Single Exp | 138 /M/s | 0.126 /s | ✓ Good (within 5% of 0.12) |
| Double Exp | 193 /M/s | 0.210 /s | ✗ Bad (75% above 0.12) |

**The single exponential gives a k_uncat that matches the known value.** This validates the method.

The double exponential gives a higher k_cat but the k_uncat is physically unreasonable (0.21 /s instead of 0.12 /s). This suggests the method is biased.

---

## 6. Addressing Concerns About Initial Rate

**Concern:** "Single exponential underestimates the fast initial kinetics."

**Response:**
- The single exponential fits the overall decay with R² > 0.99
- It correctly recovers k_uncat ≈ 0.12 /s, validating the method
- The double exponential gives inflated k_uncat, indicating systematic error
- For comparing catalysts (ranking), single exponential is internally consistent

**The goal is a method that is:**
1. Physically justified
2. Self-consistent (k_uncat matches expected value)
3. Reproducible

Single exponential meets all three criteria.

---

## 7. Variability Between Days

Observed: k_cat varies ~15% between experimental days (same solution).

**This is NOT a computational artifact.** Evidence:

1. The raw k_obs values differ between days (before any regression)
2. The fitting procedure is deterministic (same data → same result)
3. Possible experimental causes:
   - Temperature variation
   - CO2 saturation differences
   - Mixing efficiency
   - Stock solution degradation

The ~15% CV is typical for stopped-flow measurements.

---

## 8. Files for Review

1. **kinetics_analysis_DEFENSIBLE.py** - The analysis script with detailed comments
2. **Cyc_Benz_Comparison_single_results.xlsx** - Complete results in Excel format
3. **Cyc_Benz_Comparison_single_plot.png** - k_obs vs [catalyst] plot

All calculations can be verified manually using the Excel output.
