# Noise Analysis SNR Plot — Design

## Overview

Modify `noise_analysis_rp.py` to plot estimation error vs **natural SNR** instead of absolute noise level. Each Monte Carlo sample produces a scatter point; all points are plotted together with a power-law fit line.

## Motivation

The original design used fixed noise levels as X-axis, but SNR (Signal-to-Noise Ratio) is the dimensionless quantity that actually governs estimation quality. Since `|b_local|` varies naturally across random poses and magnetic moment magnitudes, the SNR values themselves are continuously distributed — making the plot richer and more representative of real conditions.

## Design

### Data Generation

```
for m in [50, 250, 500]:
    for i in range(100):              # random poses
        clean_B_meas, |b_local| = generate()
        for noise in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            SNR = |b_local| / noise
            inject noise → run estimator → record (SNR, pos_error, ori_error)
```

- Total: **3 × 100 × 6 = 1800** scatter points per error type
- All magnitudes share the same color — no distinction by `m`
- SNR is computed per-sample (not a grid), giving natural X-axis distribution

### Plot

- **Layout**: 1 row × 2 columns (Position | Orientation)
- **Axes**: `ax.set_xscale('log')`, `ax.set_yscale('log')` (double log)
- **Scatter**: all 1800 points plotted with transparency (`alpha=0.3`) to show density
- **Fit line**: power-law `error = a × SNR^b` fitted via linear regression on log-log data, overplotted as dashed line
- **Style**: academic (Computer Modern font, 2-column ~17.8cm wide, same height 12cm), same color scheme per subplot

### SNR Fit

Power-law in log-log space:
```
log10(error) = log10(a) + b × log10(SNR)
```
Fitted using `np.polyfit` on log10-transformed data. Line drawn across the full SNR range of the data.

### Output

- Single PNG file: `noise_analysis_rp_snr_cycle_{id}.png`
- Console: prints fit parameters `a` and `b` for both position and orientation errors
- Console: summary table with per-magnitude statistics (for verification only)

## Implementation Notes

- `run_multi_magnitude_analysis()` renamed scope: it now collects ALL (SNR, error) pairs regardless of magnitude, then plots them together
- `generate_synthetic_B_meas_for_pose()`: returns `|b_local|` alongside `B_meas_cell` so SNR can be computed per-sample
- `run_noise_analysis_for_magnitude()`: collects individual sample results, not just statistics
- No change to `MaPS_Estimator`, `mag_dipole_model`, or data flow — only plotting logic changes
- `noise_levels` parameter unchanged: still `[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]`
- Backward compatible: `--magnitudes`, `--samples`, `--sensors`, `--output` flags remain
