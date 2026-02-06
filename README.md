# Kinetics Data Processing

Processing and analysis pipeline for stopped-flow spectrophotometry data measuring CO2 hydration kinetics.

## Overview

This project processes absorbance vs. time data from a stopped-flow spectrometer to extract rate constants for CO2 hydration catalysis. The method tracks the color change of a pH-sensitive buffer indicator as CO2 hydrates to carbonic acid/bicarbonate.

## Data Files

### `Data/Benz_Cyclen_Summary/Cyc_Benz_Comparison.xlsx`

Excel workbook containing stopped-flow kinetics data for benzyl-cyclen ligand experiments.

#### Sheet Structure

| Sheet | Description | Q Factor | Purpose |
|-------|-------------|----------|---------|
| `12_05` | Catalyzed runs from Dec 5 | -0.402 | Precision benchmark |
| `12_13` | Catalyzed runs from Dec 13 | -0.402 | Precision benchmark |
| `01_14` | Catalyzed runs from Jan 14 | -0.402 | Precision benchmark |
| `Uncatalyzed` | No catalyst control | -0.607 | Accuracy benchmark (expected k_uncat = 0.12 /s) |

**Note:** Q is the instrument extinction coefficient factor used to convert absorbance change to concentration change.

#### Catalyzed Sheet Format (12_05, 12_13, 01_14)

Each sheet contains data for three catalyst concentrations organized in column groups:

```
| concentration = 0.25 mM | concentration = 0.5 mM | concentration = 1 mM |
|-------------------------|------------------------|----------------------|
| x | trial1 | trial2 ...| x | trial1 | trial2 ...| x | trial1 | trial2...|
```

- **Row 0**: Concentration headers (marks start of each group)
- **Row 1**: Column identifiers (`x` for time, filename for absorbance trials)
- **Rows 2+**: Numerical data
  - First column of each group: time in seconds
  - Remaining columns: absorbance values (AU)

**Trial counts vary by date and concentration:**
- 12_05: 4, 4, 6 trials for 0.25, 0.5, 1.0 mM respectively
- 12_13: 5, 5, 4 trials
- 01_14: 7, 7, 6 trials

#### Uncatalyzed Sheet Format

Simpler structure without concentration groups:

```
| Time | Trial 1 | Trial 2 | Trial 3 | Trial 4 | (blank) | Q = -0.607... |
```

- **Row 0**: Empty
- **Row 1**: Column headers
- **Rows 2+**: Time (s) and absorbance data for 4 trials

#### Data Characteristics

- **Time range recorded**: 0 to ~120 seconds (catalyzed), 0 to ~200 seconds (uncatalyzed)
- **Time resolution**: 1 ms (0.001 s)
- **Baseline shift**: Y-axis baseline shifts between successive trials within a series (instrument software artifact for overlay display; does not affect kinetic analysis)
- **CO2 concentration**: Saturated at 0.0338 M

---

## Analysis Pipeline Context

### Data Processing Protocol

#### 1. Time Window Selection
Use only data from **t = 0.02 s to t = 40 s**:
- **Early cutoff (0.02 s)**: Excludes mixing dead time artifacts and initial noise
- **Late cutoff (40 s)**: Excludes trailing region where reaction approaches completion and signal-to-noise degrades

#### 2. Exponential Fitting
Fit absorbance traces to exponential decay model(s):
- **Single exponential**: `A(t) = A0 + A * exp(-k * t)`
- **Double exponential**: `A(t) = A0 + A1 * exp(-k1 * t) + A2 * exp(-k2 * t)`

Extract initial rate: `dA/dt|_{t=0}` (slope at time zero)

#### 3. Initial Rate Extraction
The initial rate `v0 = dA/dt|_{t=0}` is calculated from fitted parameters:
- Single: `v0 = -A * k`
- Double: `v0 = -A1 * k1 - A2 * k2`

#### 4. Rate Constant Determination
For each experimental date:
1. Calculate mean initial rate for each catalyst concentration
2. Plot initial rate vs. [catalyst]
3. Linear regression: `v0 = k_cat * [catalyst] + v_uncat`
   - **Slope** = catalytic rate constant `k_cat` (M⁻¹ s⁻¹)
   - **Intercept** = uncatalyzed rate `v_uncat` (s⁻¹)

#### 5. Concentration Reordering
The three catalyst concentrations are always 0.25, 0.5, and 1.0 mM, but may be mislabeled in raw data. After obtaining initial rates, reorder by rate magnitude (rate should increase with concentration) if the order appears incorrect.

### Quality Benchmarks

| Metric | Expected Value | Source |
|--------|----------------|--------|
| Uncatalyzed rate | 0.12 /s | Literature/independent measurement |
| Replicate precision | Same k_cat across dates | Commercial ligand purity |

### Instrument Parameters

- **Technique**: Stopped-flow UV-Vis spectrophotometry
- **Detection**: pH indicator dye absorbance shift
- **Q factors**:
  - Catalyzed experiments: -0.402
  - Uncatalyzed control: -0.607

---

## Scripts

### `stopped_flow_fitting_utility.py`
Fits exponential decay models to absorbance traces. Outputs fitted parameters and dA/dt at t=0.

**Usage:**
```bash
python stopped_flow_fitting_utility.py data.csv --model single
python stopped_flow_fitting_utility.py data.xlsx --model double --cols 1 2 3
```

### `MLR.py`
Multiple linear regression for structure-activity relationships using Zn-ligand descriptors.

---

---

## Critical Analysis Findings

### Root Cause of Precision Issues

Sensitivity analysis on the Cyc_Benz_Comparison dataset revealed the following:

#### 1. Model Choice: Single vs Double Exponential

| Model | k_cat CV | k_uncat accuracy |
|-------|----------|------------------|
| Double exponential | 27% | Poor |
| Single exponential | 16% | Excellent (within 3% of expected) |

**Recommendation: Use single exponential for CO2 hydration kinetics.**

The double exponential model is numerically unstable because:
- k1/k2 ratios vary from 4.8 to 10.4 between trials
- Parameters trade off (A1↔A2, k1↔k2) giving different dA/dt values
- The optimizer finds different local minima depending on initial conditions

#### 2. Direct k_obs Extraction

For pseudo-first-order kinetics, the single exponential rate constant `k` directly gives the observed rate:

```
A(t) = A0 + A * exp(-k_obs * t)
```

Then:
```
k_obs = k_uncat + k_cat * [catalyst]
```

This approach:
- Bypasses amplitude normalization issues
- Gives units of /s directly
- Is insensitive to signal intensity variations between dates

#### 3. Validated Results

Using single exponential with t=[0.02, 40] s:

| Date | k_cat (/M/s) | k_uncat (/s) | R² |
|------|--------------|--------------|-----|
| 12_05 | 138.5 | 0.126 | 0.999 |
| 12_13 | 122.1 | 0.114 | 1.000 |
| 01_14 | 166.0 | 0.109 | 0.997 |
| **Mean** | **142 ± 22** | **0.116 ± 0.009** | |

Uncatalyzed control (direct measurement): **0.118 /s** (expected: 0.12 /s)

---

## Dependencies

```
numpy
pandas
scipy
matplotlib
openpyxl
scikit-learn
```
