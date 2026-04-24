# Noise Analysis SNR Plot — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modify `noise_analysis_rp.py` to plot estimation error vs natural SNR with scatter points and power-law fit, replacing the current fixed-noise-level errorbar plot.

**Architecture:** Change data collection to return per-sample SNR+error pairs, then change the plotting stage to scatter+fit. Core estimator pipeline unchanged.

**Tech Stack:** numpy, matplotlib, existing MaPS_Estimator pipeline

---

## File Changes

- Modify: `src/gels_localization/scripts/noise_analysis_rp.py`

---

## Task 1: Update `generate_synthetic_B_meas_for_pose` to return `|b_local|`

**Files:**
- Modify: `src/gels_localization/scripts/noise_analysis_rp.py:60-103`

- [ ] **Step 1: Modify function signature and return value**

Replace the function body (lines 60-103) so it returns `B_meas_cell` AND `|b_local|` (float, the clean field magnitude at array center).

The clean field at center is computed from `p_sensor_array` (no offset) and `m_Ci` for each source. Take the **norm of the sum** across all sources, or the **average norm** across sources — either is acceptable for SNR definition. Use the total RSS (root-sum-square) of the clean field across all sources at the center point:

```python
def generate_synthetic_B_meas_for_pose(p_sensor_array, R_sensor_array, D_LIST,
                                        gs_to_tesla,
                                        sources, sensor_ids, noise_level=0.0):
    """
    ...
    Returns:
        B_meas_cell: list of 3 x N matrices (one per source/slot)
        b_local_norm: float, |b_local| in Gs at array center (no offset)
    """
    B_meas_cell = []
    b_local_norms = []  # per-source clean field norms at center

    for src in sources:
        p_Ci = src['p_Ci']
        m_Ci = src['m_Ci']

        # Clean field at center (no offset)
        b_global_center, _ = mag_dipole_model(p_sensor_array, m_Ci, p_Ci, order=1)
        b_sensor_center = R_sensor_array.T @ b_global_center
        b_local_norms.append(np.linalg.norm(b_sensor_center))

        B_meas = np.zeros((3, len(sensor_ids)))
        for col_idx, sid in enumerate(sensor_ids):
            sensor_idx = sid - 1
            d_j = D_LIST[sensor_idx]
            p_sensor_global = p_sensor_array + R_sensor_array @ d_j

            b_global, _ = mag_dipole_model(p_sensor_global, m_Ci, p_Ci, order=1)
            b_sensor = R_sensor_array.T @ b_global

            b_sensor_gs = b_sensor / gs_to_tesla
            if noise_level > 0:
                b_sensor_gs = b_sensor_gs + noise_level * np.random.randn(3)

            B_meas[:, col_idx] = b_sensor_gs

        B_meas_cell.append(B_meas)

    b_local_norm = np.linalg.norm(b_local_norms)  # RSS across sources, in Gs
    return B_meas_cell, b_local_norm
```

- [ ] **Step 2: Verify all call sites updated**

Find all places that call `generate_synthetic_B_meas_for_pose` and update to unpack the second return value:

```python
B_meas_cell, b_local_norm = generate_synthetic_B_meas_for_pose(...)
```

Only `run_noise_analysis_for_magnitude` calls it (line ~179).

- [ ] **Step 3: Commit**

```bash
git add src/gels_localization/scripts/noise_analysis_rp.py
git commit -m "refactor: return b_local_norm from generate_synthetic_B_meas_for_pose"
```

---

## Task 2: Update `run_noise_analysis_for_magnitude` to collect per-sample (SNR, error) pairs

**Files:**
- Modify: `src/gels_localization/scripts/noise_analysis_rp.py:133-216`

- [ ] **Step 1: Rewrite function to collect individual samples**

Change the function to return a list of `(SNR, pos_error, ori_error)` tuples instead of statistics dict. Keep `num_samples`, `noise_levels`, etc. as parameters.

```python
def run_noise_analysis_for_magnitude(json_path, moment_magnitude, D_LIST, gs_to_tesla,
                                      num_samples=100,
                                      radius=0.05, noise_levels=None, sensor_ids=None,
                                      rng_seed=42):
    """
    Returns:
        list of (SNR, pos_error, ori_error) tuples for ALL samples
    """
    if noise_levels is None:
        noise_levels = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    sources, req = build_sources_from_json(json_path, moment_magnitude)
    if sensor_ids is None:
        sensor_ids = list(req.sensor_ids)

    gt = req.ground_truth_pose
    p_gt_original = np.array([gt.position.x, gt.position.y, gt.position.z])
    R_gt_original = quaternion_to_rotation_matrix(np.array([
        gt.orientation.w, gt.orientation.x, gt.orientation.y, gt.orientation.z
    ]))

    M = len(sources)
    D_cal = build_D_cal(sensor_ids)

    rng = np.random.default_rng(rng_seed)
    random_poses = []
    for i in range(num_samples):
        p_rand = sample_random_position(p_gt_original, radius)
        R_rand = sample_random_rotation()
        random_poses.append((p_rand, R_rand))

    all_results = []  # (SNR, pos_error, ori_error)

    for nl_idx, noise_level in enumerate(noise_levels):
        for sample_idx, (p_gt, R_gt) in enumerate(random_poses):
            B_meas_cell, b_local_norm = generate_synthetic_B_meas_for_pose(
                p_gt, R_gt, D_LIST, gs_to_tesla, sources, sensor_ids, noise_level=noise_level
            )

            SNR = b_local_norm / noise_level if noise_level > 0 else np.inf

            try:
                R_est, p_est, details = MaPS_Estimator(D_cal, sources, B_meas_cell)
            except Exception:
                continue

            pos_error = np.linalg.norm(p_est - p_gt)
            ori_error = compute_rotation_error(R_est, R_gt)
            all_results.append((SNR, pos_error, ori_error))

    return all_results
```

- [ ] **Step 2: Verify function signature consistent**

The function is called from `run_multi_magnitude_analysis` — signature is unchanged (same parameters), only return type changes.

- [ ] **Step 3: Commit**

```bash
git add src/gels_localization/scripts/noise_analysis_rp.py
git commit -m "refactor: run_noise_analysis_for_magnitude returns per-sample (SNR, error) pairs"
```

---

## Task 3: Rewrite `run_multi_magnitude_analysis` for scatter + fit plotting

**Files:**
- Modify: `src/gels_localization/scripts/noise_analysis_rp.py:219-380`

- [ ] **Step 1: Rewrite to collect all results then plot scatter + fit**

Replace the body of `run_multi_magnitude_analysis` (lines 219-380) so it:
1. Collects `all_snr`, `all_pos_err`, `all_ori_err` from ALL magnitudes (no grouping)
2. Plots scatter on log-log axes
3. Fits power-law `error = a * SNR^b` via `np.polyfit` on log10 data
4. Overplots fit line

```python
def run_multi_magnitude_analysis(json_path, magnitudes, num_samples=100, radius=0.05,
                                  noise_levels=None, sensor_ids=None, rng_seed=42,
                                  output_path=None):
    """
    Collect all (SNR, error) pairs across all magnitudes, then plot scatter + fit.
    """
    if noise_levels is None:
        noise_levels = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    from sensor_array_config import get_config
    config = get_config('QMC6309')
    D_LIST_RAW = np.array(config.hardware.d_list)
    GS_TO_TESLA_VAL = config.gs_to_si

    import localization_service_node as lsn
    lsn.D_LIST = D_LIST_RAW
    lsn.GS_TO_TESLA = GS_TO_TESLA_VAL

    all_snr = []
    all_pos_err = []
    all_ori_err = []

    for mag in magnitudes:
        print(f"\n=== Moment magnitude = {mag} A·m² ===")
        results = run_noise_analysis_for_magnitude(
            json_path=json_path,
            moment_magnitude=mag,
            D_LIST=D_LIST_RAW,
            gs_to_tesla=GS_TO_TESLA_VAL,
            num_samples=num_samples,
            radius=radius,
            noise_levels=noise_levels,
            sensor_ids=sensor_ids,
            rng_seed=rng_seed
        )
        for snr, pos_err, ori_err in results:
            all_snr.append(snr)
            all_pos_err.append(pos_err)
            all_ori_err.append(ori_err)

    all_snr = np.array(all_snr)
    all_pos_err = np.array(all_pos_err)
    all_ori_err = np.array(all_ori_err)

    # Filter valid (non-nan, positive) points
    valid_pos = ~(np.isnan(all_pos_err) | (all_pos_err <= 0))
    valid_ori = ~(np.isnan(all_ori_err) | (all_ori_err <= 0))

    # Plot — academic style
    fig, axes = plt.subplots(1, 2, figsize=(17.8 / 2.54, 12.0 / 2.54))

    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 16
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['axes.formatter.use_mathtext'] = True

    # ---- Position error subplot ----
    ax = axes[0]
    snr_v = all_snr[valid_pos]
    pos_v = all_pos_err[valid_pos] * 1000  # convert to mm

    ax.scatter(snr_v, pos_v, alpha=0.3, s=10, color='steelblue', label='Samples')

    # Power-law fit on log-log
    log_snr = np.log10(snr_v)
    log_pos = np.log10(pos_v)
    coeffs = np.polyfit(log_snr, log_pos, 1)
    b_pos, log_a_pos = coeffs[0], coeffs[1]
    a_pos = 10**log_a_pos

    # Fit line
    snr_fit = np.logspace(np.log10(snr_v.min()), np.log10(snr_v.max()), 200)
    pos_fit = a_pos * snr_fit**b_pos
    ax.plot(snr_fit, pos_fit, 'r--', linewidth=1.5, label=f'Fit: $a={a_pos:.2e}$, $b={b_pos:.2f}$')

    ax.set_xlabel(r'$\mathrm{SNR}$', fontsize=16)
    ax.set_ylabel(r'$\mathrm{Position\ Error}$ $\mathrm{[mm]}$', fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=11, loc='upper right')
    ax.tick_params(labelsize=14)

    print(f"Position fit: error = {a_pos:.4e} * SNR^({b_pos:.4f})")

    # ---- Orientation error subplot ----
    ax = axes[1]
    snr_v = all_snr[valid_ori]
    ori_v = np.degrees(all_ori_err[valid_ori])

    ax.scatter(snr_v, ori_v, alpha=0.3, s=10, color='darkorange', label='Samples')

    log_snr = np.log10(snr_v)
    log_ori = np.log10(ori_v)
    coeffs = np.polyfit(log_snr, log_ori, 1)
    b_ori, log_a_ori = coeffs[0], coeffs[1]
    a_ori = 10**log_a_ori

    ori_fit = a_ori * snr_fit**b_ori
    ax.plot(snr_fit, ori_fit, 'r--', linewidth=1.5, label=f'Fit: $a={a_ori:.2e}$, $b={b_ori:.2f}$')

    ax.set_xlabel(r'$\mathrm{SNR}$', fontsize=16)
    ax.set_ylabel(r'$\mathrm{Orientation\ Error}$ $[^{\circ}]$', fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=11, loc='upper right')
    ax.tick_params(labelsize=14)

    print(f"Orientation fit: error = {a_ori:.4e} * SNR^({b_ori:.4f})")

    plt.tight_layout(pad=0.5)

    out_path = output_path
    if out_path is None:
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(json_path)),
            f'noise_analysis_rp_snr_cycle_{0:04d}.png'
        )
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {out_path}")
    plt.close()
```

- [ ] **Step 2: Commit**

```bash
git add src/gels_localization/scripts/noise_analysis_rp.py
git commit -m "feat: scatter+fit SNR plot replacing errorbar noise-level plot"
```

---

## Task 4: Run and verify

**Files:**
- Test: `src/gels_localization/scripts/noise_analysis_rp.py`

- [ ] **Step 1: Run the script**

```bash
cd /home/zhang/embedded_array_ws
python src/gels_localization/scripts/noise_analysis_rp.py
```

Expected: finishes without error, prints fit parameters, saves PNG to `sensor_data_collection/result/`

- [ ] **Step 2: Verify plot output exists and is non-empty**

Check the PNG file was created and has reasonable size (>50KB).

- [ ] **Step 3: Commit final state**

```bash
git add src/gels_localization/scripts/noise_analysis_rp.py
git commit -m "feat: SNR-based noise analysis with scatter+power-law fit"
```
