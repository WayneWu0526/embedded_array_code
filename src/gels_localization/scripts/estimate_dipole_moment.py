#!/usr/bin/env python3
"""
Estimate permanent magnet dipole moment from measured magnetic field data.

Uses the first-order magnetic dipole model to estimate the magnetic moment magnitude,
then compares model predictions with actual measurements.
"""

import json
import numpy as np
import os

# Add scripts directory to path for imports
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, SCRIPTS_DIR)

from mag_dipole_model import mag_dipole_model


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """Convert quaternion to rotation matrix."""
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R


def quaternion_y_axis(q):
    """
    Extract the y-axis direction from a quaternion.

    Args:
        q: quaternion [w, x, y, z]

    Returns:
        np.ndarray: y-axis direction vector (3,)
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    y_axis = np.array([
        2 * (x * y - z * w),
        1 - 2 * (x**2 + z**2),
        2 * (y * z + x * w)
    ])
    return y_axis / np.linalg.norm(y_axis)


def load_permanent_magnet_data():
    """Load permanent magnet measurement data."""
    result_dir = os.path.join(
        os.path.dirname(os.path.dirname(SCRIPTS_DIR)),
        'sensor_data_collection', 'result'
    )
    data_path = os.path.join(result_dir, 'permanent_magnet_data.json')

    with open(data_path, 'r') as f:
        data = json.load(f)

    return data


def estimate_dipole_moment(measurements):
    """
    Estimate the dipole moment magnitude by minimizing prediction error.

    The magnetization direction is along local -y axis.
    We find |m| that best fits all measurements.

    Transforms each measurement to its own source frame (per slot):
    - p_in_source = R_source^T @ (p_sensor - p_source)
    - b_in_source = R_source^T @ b_in_world

    Args:
        measurements: list of measurement dicts

    Returns:
        m_magnitude: estimated magnetic moment magnitude (A·m²)
        residuals: list of residual vectors (b_pred - b_meas)
        results: list of dicts with prediction details
    """
    # Stack all measurements
    n = len(measurements)
    p_in_source_list = []
    p_magnets = []
    m_directions = []  # Unit vectors for dipole moment direction in source frame
    b_in_source_list = []

    print("\n  Data transformation (divide by 4, transform to source frame):")

    for idx, m in enumerate(measurements):
        # Sensor position in world frame
        p_sensor_world = np.array([
            m['sensor_array_pose']['position']['x'],
            m['sensor_array_pose']['position']['y'],
            m['sensor_array_pose']['position']['z']
        ])

        # Magnet (source) position in world frame
        p_magnet_world = np.array([
            m['permanent_magnet_pose']['position']['x'],
            m['permanent_magnet_pose']['position']['y'],
            m['permanent_magnet_pose']['position']['z']
        ])

        # Magnet orientation quaternion
        ori = m['permanent_magnet_pose']['orientation']
        q = np.array([ori['w'], ori['x'], ori['y'], ori['z']])
        R_source = quaternion_to_rotation_matrix(q[1], q[2], q[3], q[0])

        # Magnetization direction in SOURCE/LOCAL frame: local -y (fixed)
        # (magnetization along local -y NEGATIVE direction, per user specification)
        # In source frame: magnet is at origin, so local frame = source frame
        # Thus m_direction = (0, -1, 0) in source frame, NOT rotated to world
        m_direction = np.array([0.0, -1.0, 0.0])  # local -y direction (user specified)

        # Measured magnetic field (world frame) - DIVIDE BY 4 as requested
        b_in_world = np.array([
            m['b_in_world']['x'] / 4.0,
            m['b_in_world']['y'] / 4.0,
            m['b_in_world']['z'] / 4.0
        ])

        # Transform to source frame:
        # p_in_source = R_source^T @ (p_sensor - p_source)
        p_in_source = R_source.T @ (p_sensor_world - p_magnet_world)

        # b_in_source = R_source^T @ b_in_world
        b_in_source = R_source.T @ b_in_world

        print(f"    [{idx:2d}] Cycle {m['cycle_id']}, Slot {m['slot']}: "
              f"|b_in_world| = {np.linalg.norm(b_in_world):.3e} T -> |b_in_source| = {np.linalg.norm(b_in_source):.3e} T, "
              f"|p_in_source| = {np.linalg.norm(p_in_source):.4f} m")

        p_in_source_list.append(p_in_source)
        p_magnets.append(p_magnet_world)  # Keep original magnet position for reference
        m_directions.append(m_direction)
        b_in_source_list.append(b_in_source)

    p_sensors = np.array(p_in_source_list)
    p_magnets = np.array(p_magnets)
    m_directions = np.array(m_directions)
    b_measured = np.array(b_in_source_list)

    mu0 = 4 * np.pi * 1e-7
    coeff = mu0 / (4 * np.pi)

    def compute_prediction(m_mag, p_sensor, m_dir):
        """
        Compute dipole model prediction in source frame.

        In source frame: magnet is at origin (0,0,0)
        p_sensor is the sensor position relative to magnet (source frame origin)
        """
        m_vec = m_mag * m_dir  # magnetic moment vector in source frame
        r_vec = p_sensor  # in source frame, magnet is at origin, so r = p_sensor
        r = np.linalg.norm(r_vec)
        r_hat = r_vec / r if r > 1e-10 else np.array([0, 0, 1])

        # b = coeff * (3*(m·r̂)*r̂ - m) / r^3
        b_pred = coeff * (3 * np.dot(m_vec, r_hat) * r_hat - m_vec) / r**3
        return b_pred, r

    def objective(m_mag):
        """Objective function: sum of squared errors."""
        total_error = 0.0
        for i in range(n):
            b_pred, _ = compute_prediction(m_mag, p_sensors[i], m_directions[i])
            error = b_pred - b_measured[i]
            total_error += np.dot(error, error)
        return total_error

    # Optimize |m| around 220 A·m² (NdFeB N62 typical)
    from scipy.optimize import minimize_scalar

    print(f"\n  Optimizing magnetic moment magnitude around 220 A·m²...")

    result = minimize_scalar(objective, bounds=(100.0, 500.0),
                            method='bounded', options={'xatol': 1e-8, 'maxiter': 1000})
    m_optimal = result.x

    print(f"  Optimized: m = {m_optimal:.4f} A·m², error = {result.fun:.6e}")

    # Compute all predictions and residuals
    results = []
    residuals = []
    for i in range(n):
        b_pred, r = compute_prediction(m_optimal, p_sensors[i], m_directions[i])
        residual = b_pred - b_measured[i]
        residuals.append(residual)

        results.append({
            'cycle_id': measurements[i]['cycle_id'],
            'slot': measurements[i]['slot'],
            'p_sensor': p_sensors[i],
            'm_direction': m_directions[i],
            'b_measured': b_measured[i],
            'b_predicted': b_pred,
            'residual': residual,
            'r': r
        })

    return m_optimal, objective, p_sensors, m_directions, b_measured, results


def main():
    print("="*80)
    print("PERMANENT MAGNET DIPOLE MOMENT ESTIMATION")
    print("="*80)

    # Load data
    data = load_permanent_magnet_data()
    measurements = data['measurements']
    print(f"\nLoaded {len(measurements)} measurements")

    # Estimate dipole moment
    print("\nEstimating dipole moment magnitude...")
    m_optimal, objective, p_sensors, m_directions, b_measured, results = estimate_dipole_moment(measurements)

    print(f"\nEstimated magnetic moment magnitude: |m| = {m_optimal:.6f} A·m²")
    print(f"         (= {m_optimal * 1e3:.3f} mA·m²)")

    # Print magnet dimensions used in model
    L = 240.0e-3  # Length in mm -> m
    D = 40.0e-3   # Diameter in mm -> m
    volume = np.pi * (D/2)**2 * L  # Cylindrical magnet volume
    print(f"\nFor reference (model uses these dimensions):")
    print(f"  L = {L*1e3:.1f} mm, D = {D*1e3:.1f} mm")
    print(f"  Volume = {volume*1e6:.2f} cm³")

    # Compute statistics
    residuals_array = np.array([r['residual'] for r in results])
    print("\n" + "="*80)
    print("MODEL VS MEASUREMENT COMPARISON")
    print("="*80)

    print(f"\n{'Idx':<4} {'Cycle':<6} {'Slot':<4} {'r(m)':<10} {'|B_meas|(G)':<14} {'|B_pred|(G)':<14} {'|Error|(G)':<12} {'Error%':<10}")
    print("-"*90)

    total_error_norm = 0.0
    total_b_norm = 0.0

    for i, r in enumerate(results):
        b_meas_norm = np.linalg.norm(r['b_measured']) * 1e4  # T -> G
        b_pred_norm = np.linalg.norm(r['b_predicted']) * 1e4  # T -> G
        error_norm = np.linalg.norm(r['residual']) * 1e4  # T -> G
        error_pct = error_norm / b_meas_norm * 100 if b_meas_norm > 0 else 0

        total_error_norm += (error_norm / 1e4)**2  # back to T^2 for calculation
        total_b_norm += (b_meas_norm / 1e4)**2  # back to T^2 for calculation

        print(f"{i:<4} {r['cycle_id']:<6} {r['slot']:<4} {r['r']:<10.4f} {b_meas_norm:<14.6f} {b_pred_norm:<14.6f} {error_norm:<12.6f} {error_pct:<10.2f}%")

    rmse = np.sqrt(total_error_norm / len(results)) * 1e4  # T -> G
    rms_b = np.sqrt(total_b_norm / len(results)) * 1e4  # T -> G
    relative_rmse = rmse / rms_b * 100

    print("-"*90)
    print(f"\nSUMMARY STATISTICS:")
    print(f"  Overall RMSE:                    {rmse:.6f} G")
    print(f"  RMS of measured field magnitude: {rms_b:.6f} G")
    print(f"  Relative RMSE:                   {relative_rmse:.2f}%")

    # Per-component analysis
    print(f"\nPER-COMPONENT ANALYSIS:")
    print(f"  Bx: RMSE = {np.sqrt(np.mean(residuals_array[:,0]**2)) * 1e4:.6f} G")
    print(f"  By: RMSE = {np.sqrt(np.mean(residuals_array[:,1]**2)) * 1e4:.6f} G")
    print(f"  Bz: RMSE = {np.sqrt(np.mean(residuals_array[:,2]**2)) * 1e4:.6f} G")

    # Show detailed comparison for first few measurements
    print("\n" + "="*80)
    print("DETAILED COMPARISON (first 5 measurements)")
    print("="*80)

    for i in range(min(5, len(results))):
        r = results[i]
        print(f"\nMeasurement {i} (Cycle {r['cycle_id']}, Slot {r['slot']}):")
        print(f"  Sensor position in source frame: ({r['p_sensor'][0]:.4f}, {r['p_sensor'][1]:.4f}, {r['p_sensor'][2]:.4f}) m")
        print(f"  Distance r (sensor to origin): {r['r']:.4f} m")
        print(f"  Magnetic moment direction:      ({r['m_direction'][0]:.4f}, {r['m_direction'][1]:.4f}, {r['m_direction'][2]:.4f})")
        print(f"  B_measured (G):   ({r['b_measured'][0]*1e4:.6f}, {r['b_measured'][1]*1e4:.6f}, {r['b_measured'][2]*1e4:.6f})")
        print(f"  B_predicted (G):  ({r['b_predicted'][0]*1e4:.6f}, {r['b_predicted'][1]*1e4:.6f}, {r['b_predicted'][2]*1e4:.6f})")
        print(f"  Residual (G):    ({r['residual'][0]*1e4:.6f}, {r['residual'][1]*1e4:.6f}, {r['residual'][2]*1e4:.6f})")

    # Generate error analysis plot
    print("\n" + "="*80)
    print("GENERATING ERROR ANALYSIS PLOT")
    print("="*80)

    import matplotlib.pyplot as plt

    # Error vs magnetic moment magnitude
    m_range = np.linspace(50, 600, 200)
    errors = [objective(m) for m in m_range]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Subplot 1: Error vs m
    ax1 = axes[0, 0]
    ax1.plot(m_range, errors, 'b-', linewidth=2)
    ax1.axvline(m_optimal, color='r', linestyle='--', linewidth=2, label=f'Optimal m = {m_optimal:.2f}')
    ax1.axvline(220, color='g', linestyle=':', linewidth=2, label='Initial m = 220')
    ax1.set_xlabel('Magnetic Moment |m| (A·m²)', fontsize=11)
    ax1.set_ylabel('Sum of Squared Errors', fontsize=11)
    ax1.set_title('Objective Function vs Magnetic Moment', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Measured vs Predicted magnetic field magnitude
    ax2 = axes[0, 1]
    b_meas_norms = [np.linalg.norm(r['b_measured']) * 1e4 for r in results]  # T -> G
    b_pred_norms = [np.linalg.norm(r['b_predicted']) * 1e4 for r in results]  # T -> G
    indices = range(len(results))

    ax2.scatter(b_meas_norms, b_pred_norms, c='blue', s=60, alpha=0.7, label='Measurements')
    max_val = max(max(b_meas_norms), max(b_pred_norms)) * 1.1
    ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect match')
    ax2.set_xlabel('|B_measured| (G)', fontsize=11)
    ax2.set_ylabel('|B_predicted| (G)', fontsize=11)
    ax2.set_title('Measured vs Predicted Field Magnitude', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Error percentage per measurement
    ax3 = axes[1, 0]
    error_pcts = [np.linalg.norm(r['residual']) / np.linalg.norm(r['b_measured']) * 100
                  for r in results]
    colors = ['green' if e < 100 else 'orange' if e < 150 else 'red' for e in error_pcts]
    bars = ax3.bar(indices, error_pcts, color=colors, alpha=0.7)
    ax3.axhline(np.mean(error_pcts), color='black', linestyle='--', linewidth=2,
                label=f'Mean = {np.mean(error_pcts):.1f}%')
    ax3.set_xlabel('Measurement Index', fontsize=11)
    ax3.set_ylabel('Relative Error (%)', fontsize=11)
    ax3.set_title('Relative Error per Measurement', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Subplot 4: Residual vector components
    ax4 = axes[1, 1]
    residuals_array = np.array([r['residual'] for r in results])
    x_pos = np.arange(len(results))
    width = 0.25

    ax4.bar(x_pos - width, residuals_array[:, 0] * 1e4, width, label='Bx residual (G)', alpha=0.8)
    ax4.bar(x_pos, residuals_array[:, 1] * 1e4, width, label='By residual (G)', alpha=0.8)
    ax4.bar(x_pos + width, residuals_array[:, 2] * 1e4, width, label='Bz residual (G)', alpha=0.8)
    ax4.set_xlabel('Measurement Index', fontsize=11)
    ax4.set_ylabel('Residual (G)', fontsize=11)
    ax4.set_title('Residual Components per Measurement', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.suptitle(f'Permanent Magnet Dipole Moment Estimation\n|m| = {m_optimal:.2f} A·m², Local -y magnetization',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(SCRIPTS_DIR)),
                             'sensor_data_collection', 'result')
    plot_path = os.path.join(output_dir, 'dipole_moment_error_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved to: {plot_path}")
    plt.close()

    return m_optimal, results


if __name__ == '__main__':
    main()