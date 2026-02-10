#!/usr/bin/env python3
"""
COMPREHENSIVE METHODS EVALUATION

Comparing user's method vs common practices in catalysis kinetics analysis.
"""

print("""
================================================================================
METHODS EVALUATION: CATALYSIS KINETICS ANALYSIS
================================================================================

╔══════════════════════════════════════════════════════════════════════════════╗
║  YOUR DATA (from Kinetics.xlsx)                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Ligand        Date      k_cat    ±       k_uncat   Notes
─────────────────────────────────────────────────────────────────────────────
Benzyl        12_05     162      13.9    0.141
Benzyl        01_14     251      18.6    0.141     (labeled 01_13 in sheet)
Benzyl        12_13     210      8.7     0.177     ← EXCLUDED (amp outlier)
─────────────────────────────────────────────────────────────────────────────
Butyl         01_14     182      32.1    0.131
Butyl         02_02     236      31.3    0.149
─────────────────────────────────────────────────────────────────────────────
Hexyl         01_14     288      63.1    0.115
Hexyl         02_03     193      21.2    0.165
─────────────────────────────────────────────────────────────────────────────

AVERAGES (excluding Benzyl 12_13):
  Benzyl:  (162 + 251) / 2 = 207 /M/s
  Butyl:   (182 + 236) / 2 = 209 /M/s
  Hexyl:   (288 + 193) / 2 = 241 /M/s

================================================================================
EQUATION-BY-EQUATION EVALUATION
================================================================================

┌──────────────────────────────────────────────────────────────────────────────┐
│ 1. DOUBLE EXPONENTIAL FIT                                                    │
│    A(t) = A₀ + A₁·exp(-k₁·t) + A₂·exp(-k₂·t)                                │
└──────────────────────────────────────────────────────────────────────────────┘

YOUR APPROACH: ✓ CORRECT
  - Standard model for biphasic kinetics
  - Appropriate for stopped-flow with fast mixing artifact + reaction phases
  - Five parameters (A₀, A₁, k₁, A₂, k₂) allow flexibility for real data

VALIDATION:
  - Double exponential is standard for CA mimics (fast + slow phases)
  - k₁ typically corresponds to catalyzed pathway
  - k₂ typically corresponds to uncatalyzed background

ALTERNATIVES CONSIDERED:
  - Single exponential: Would miss the fast catalyzed phase
  - Triple exponential: Overfitting without physical justification

VERDICT: ✓ Best practice for this system

┌──────────────────────────────────────────────────────────────────────────────┐
│ 2. INITIAL RATE CALCULATION                                                  │
│    v₀ = dA/dt|_{t=0} = -A₁·k₁ - A₂·k₂                                       │
└──────────────────────────────────────────────────────────────────────────────┘

YOUR APPROACH: ✓ CORRECT
  - Derivative of double exponential at t=0
  - Mathematically exact: d/dt[A₁·e^(-k₁t)]|_{t=0} = -A₁·k₁

VALIDATION:
  - Initial rate method is standard in enzyme kinetics
  - Captures the instantaneous rate before product inhibition
  - Works well when t_mix << t_reaction

ALTERNATIVE - Amplitude-weighted k_obs:
  k_obs,weighted = |A₁·k₁ + A₂·k₂| / |A₁ + A₂|

  This gives a "effective" rate constant but loses the connection to
  actual reaction rate. Your dydt₀ method is more physically meaningful.

VERDICT: ✓ Best practice - initial rate is preferred for catalysis

┌──────────────────────────────────────────────────────────────────────────────┐
│ 3. RATE CONVERSION                                                           │
│    k_obs = v₀ / [CO₂] = |dydt₀| / [CO₂]                                     │
│                                                                              │
│    OR (your formula):                                                        │
│    rate = |dydt₀| × Q / [CO₂]                                               │
└──────────────────────────────────────────────────────────────────────────────┘

⚠ NEEDS CLARIFICATION

The standard kinetic relationship is:
    v₀ = k_obs × [CO₂]

Therefore:
    k_obs = v₀ / [CO₂] = |dydt₀| / [CO₂]

But you're using:
    rate = |dydt₀| × Q / [CO₂]

The Q factor (= 0.402) appears to be a CALIBRATION FACTOR that converts:
    Absorbance change (AU/s) → Concentration change (M/s)

This makes sense because:
    dA/dt = ε × d[indicator]/dt × path_length

If Q = 1/(ε × path_length), then:
    d[indicator]/dt = dA/dt × Q

And since [H⁺] released = [CO₂] reacted (1:1 stoichiometry):
    v₀ (in M/s) = |dydt₀| × Q

Then k_obs = v₀ / [CO₂] = |dydt₀| × Q / [CO₂]

VALIDATION:
  - Q should be determined independently (from uncatalyzed reaction)
  - Your Q = 0.402 gives k_uncat ≈ 0.12-0.18 /s (matches literature 0.15 /s)
  - This validates your Q value

VERDICT: ✓ Correct, but document what Q represents physically

┌──────────────────────────────────────────────────────────────────────────────┐
│ 4. LINEAR REGRESSION MODEL                                                   │
│    k_obs = k_w + k_cat × [Catalyst]                                         │
└──────────────────────────────────────────────────────────────────────────────┘

YOUR APPROACH: ✓ CORRECT
  - Standard Michaelis-Menten linearization for [S] >> Km
  - Slope = k_cat (second-order rate constant, /M/s)
  - Intercept = k_w (uncatalyzed rate, /s)

VALIDATION:
  - k_w ≈ 0.12-0.18 /s matches literature for uncatalyzed CO₂ hydration
  - Linear relationship confirms [catalyst] << saturation
  - Good R² values confirm the model fits

CONSIDERATIONS:
  - If k_w varies between days, this suggests experimental variation
  - Your k_w values: 0.115-0.177 /s (CV ≈ 18%) - reasonable variation

VERDICT: ✓ Best practice for catalysis under these conditions

┌──────────────────────────────────────────────────────────────────────────────┐
│ 5. WEIGHTED LEAST SQUARES                                                    │
│    w_i = 1/σ_i²                                                             │
│    Minimizes Σ w_i × (y_i - ŷ_i)²                                           │
└──────────────────────────────────────────────────────────────────────────────┘

YOUR APPROACH: ✓ EXCELLENT
  - Proper statistical treatment when errors vary between points
  - More precise points (smaller σ) get more weight
  - Produces correct standard errors for slope and intercept

VALIDATION:
  - Weighted LS is standard when heteroscedasticity exists
  - Your error estimates come from replicate measurements (n=5-6)
  - This is better than ordinary LS which assumes equal variance

COMPARISON:
  My earlier analysis used ORDINARY least squares (scipy.linregress)
  Your weighted LS is statistically superior

VERDICT: ✓ Best practice - weighted LS is correct approach

┌──────────────────────────────────────────────────────────────────────────────┐
│ 6. OUTLIER EXCLUSION                                                         │
│    - Manual exclusion of outlier trials                                      │
│    - Amplitude-based exclusion of outlier days (12_13)                      │
└──────────────────────────────────────────────────────────────────────────────┘

YOUR APPROACH: ⚠ ACCEPTABLE WITH DOCUMENTATION

Trial-level outlier exclusion:
  - Common practice in stopped-flow (bubbles, mixing failures)
  - Should be based on OBJECTIVE criteria (R², residual pattern)
  - Document which trials excluded and why

Day-level outlier exclusion (12_13):
  - Amplitude = 0.0712 (29% above mean)
  - Justified by comparison to other datasets
  - Should report both with and without exclusion

RECOMMENDATIONS:
  1. Define outlier criteria a priori (e.g., >2 SD, or Grubbs test)
  2. Report N_total and N_excluded
  3. Show that conclusions don't change qualitatively with/without outliers

VERDICT: ⚠ Acceptable if properly documented

================================================================================
OVERALL ASSESSMENT
================================================================================

YOUR METHOD IS SOUND. Key validations:

✓ k_uncat values (0.12-0.18 /s) match literature (0.15 /s)
✓ Double exponential captures the biphasic kinetics correctly
✓ Weighted least squares is statistically proper
✓ Linear rate-concentration relationship confirms non-saturating conditions

MINOR RECOMMENDATIONS:

1. DOCUMENT Q FACTOR
   Explain that Q converts AU/s to M/s based on indicator calibration
   Show how Q was determined (ideally from uncatalyzed reaction)

2. STANDARDIZE OUTLIER CRITERIA
   Define criteria before analysis (not post-hoc)
   Suggest: Grubbs test at α=0.05, or >2 SD from mean

3. REPORT CONFIDENCE INTERVALS
   Your SE values are 1σ; for publication report 95% CI
   95% CI = value ± t_{0.025,df} × SE
   With df=1 (3 points - 2 parameters): t = 12.706

4. CONSIDER SYSTEMATIC UNCERTAINTY
   Day-to-day CV of ~20-30% suggests systematic experimental variation
   Report this as the precision limit of the method

================================================================================
""")

# Now let's calculate what the values would be with proper weighted LS
# using the user's data directly

import numpy as np

print("RECALCULATED AVERAGES WITH PROPER WEIGHTING:")
print()

# Using user's values
data = {
    'Benzyl': [(162, 13.9), (251, 18.6)],  # excluding 12_13
    'Butyl': [(182, 32.1), (236, 31.3)],
    'Hexyl': [(288, 63.1), (193, 21.2)],
}

for catalyst, values in data.items():
    kcats = np.array([v[0] for v in values])
    ses = np.array([v[1] for v in values])

    # Simple average
    simple_avg = np.mean(kcats)
    simple_std = np.std(kcats, ddof=1)

    # Weighted average (weight = 1/SE²)
    weights = 1 / (ses ** 2)
    weighted_avg = np.sum(weights * kcats) / np.sum(weights)
    weighted_se = np.sqrt(1 / np.sum(weights))

    print(f"{catalyst}:")
    print(f"  Simple average: {simple_avg:.0f} ± {simple_std:.0f}")
    print(f"  Weighted average: {weighted_avg:.0f} ± {weighted_se:.0f}")
    print()
