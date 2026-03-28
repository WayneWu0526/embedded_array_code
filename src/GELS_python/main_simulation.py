"""
MaPS Simulation: 6-DoF Pose Estimation Verification
Main simulation script - Python port of main_simulation.m
Compares MaPS estimator with Nonlinear Least Squares (LM) baseline.
Uses experimental configuration from Exp_data_202602.
"""

import numpy as np
import time
from mag_dipole_model import mag_dipole_model, eul2rotm
from maps_estimator import MaPS_Estimator
from nlls_lm_solver import nlls_lm_solver_from_maps


def run_maps_estimator(D_cal, sources, B_meas_cell, n_runs=10):
    """Run MaPS estimator and return timing statistics."""
    t_exec = np.zeros(n_runs)
    R_est_all = []
    p_est_all = np.zeros((3, n_runs))

    for k in range(n_runs):
        start = time.perf_counter()
        R_est, p_est, details = MaPS_Estimator(D_cal, sources, B_meas_cell)
        t_exec[k] = time.perf_counter() - start
        R_est_all.append(R_est)
        p_est_all[:, k] = p_est

    return R_est_all, p_est_all, t_exec, details


def run_nlls_lm(D_cal, sources, B_meas_cell, R_init=None, p_init=None, n_runs=10):
    """Run NLLS-LM solver and return timing statistics."""
    t_exec = np.zeros(n_runs)
    R_est_all = []
    p_est_all = np.zeros((3, n_runs))

    for k in range(n_runs):
        start = time.perf_counter()
        R_est, p_est, result = nlls_lm_solver_from_maps(
            D_cal, sources, B_meas_cell,
            R_init=R_init if k == 0 else R_est,
            p_init=p_init if k == 0 else p_est,
            max_nfev=100
        )
        t_exec[k] = time.perf_counter() - start
        R_est_all.append(R_est)
        p_est_all[:, k] = p_est

    return R_est_all, p_est_all, t_exec


def main():
    # 1. Sensor Array Configuration (Centered layout)
    d = 2e-3  # 2mm offset
    D_cal = np.array([
        [d, -d, 0, 0],
        [0,  0, d, -d],
        [0,  0, 0,  0]
    ])  # 4 sensors in a cross pattern on XY plane

    # 2. Ground Truth Pose (from experimental data)
    p_true = np.array([0.5, 0.5, 0.5])  # Sensor array center
    euler_true = np.array([0, 0, 0])  # No rotation for simplicity
    R_true = eul2rotm(np.deg2rad(euler_true), 'ZYX')

    # 3. Magnetic Sources (M = 3) - from experimental data
    # Extract from earliest experimental record (calibrated_record_20260207_215323.json)
    # Using real robot poses: diana7, arm1, arm2
    # Using Y-axis of rotation matrix for magnetic moment direction
    sources = [
        {
            'name': 'diana7',
            'p_Ci': np.array([0.61874425, -0.03768031, 1.13160486]),
            'm_Ci': np.array([0.00554931, 0.06544571, -0.49566729]) * 0.5
        },
        {
            'name': 'arm1',
            'p_Ci': np.array([1.00828728, 0.17619624, 1.11044773]),
            'm_Ci': np.array([-0.19945896, -0.18818124, 0.41809562]) * 0.5
        },
        {
            'name': 'arm2',
            'p_Ci': np.array([0.96907553, -0.23677016, 1.12432226]),
            'm_Ci': np.array([0.22931434, 0.32077883, 0.30743434]) * 0.5
        },
    ]

    M = len(sources)

    # 4. Generate Synthetic Measurements
    N = D_cal.shape[1]
    B_meas_cell = []

    print('Generating synthetic measurements from experimental source configuration...')
    print(f'Sources: {[s["name"] for s in sources]}')
    print()

    for i in range(M):
        B_meas = np.zeros((3, N))
        for j in range(N):
            # Sensor position in global frame: p_s = p + R @ d_j
            p_sensor_global = p_true + R_true @ D_cal[:, j]

            # Generate field at sensor (using dipole model)
            b_global, _ = mag_dipole_model(
                p_sensor_global,
                sources[i]['m_Ci'],
                sources[i]['p_Ci']
            )

            # Transform to sensor frame: b_sensor = R.T @ b_global
            B_meas[:, j] = R_true.T @ b_global

        B_meas_cell.append(B_meas)

    # 5. Run MaPS Estimator
    n_runs = 10
    print(f'{"="*60}')
    print('Running MaPS Estimator Performance Test...')
    print(f'{"="*60}')

    R_est_maps, p_est_maps, t_exec_maps, details = run_maps_estimator(
        D_cal, sources, B_meas_cell, n_runs
    )

    # MaPS statistics
    avg_time_maps = np.mean(t_exec_maps) * 1e6
    median_time_maps = np.median(t_exec_maps) * 1e6
    std_time_maps = np.std(t_exec_maps) * 1e6
    min_time_maps = np.min(t_exec_maps) * 1e6
    max_time_maps = np.max(t_exec_maps) * 1e6

    print(f'\n--- MaPS Estimator: Computational Performance (over {n_runs} runs) ---')
    print(f'Average Execution Time: {avg_time_maps:.2f} us')
    print(f'Median Execution Time:  {median_time_maps:.2f} us')
    print(f'Standard Deviation:     {std_time_maps:.2f} us')
    print(f'Min Execution Time:     {min_time_maps:.2f} us')
    print(f'Max Execution Time:     {max_time_maps:.2f} us')

    # MaPS final results
    R_est = R_est_maps[-1]
    p_est = p_est_maps[:, -1]
    pos_error_maps = np.linalg.norm(p_est - p_true) * 1000  # mm
    rot_error_mat = R_est.T @ R_true
    angle_error_maps = np.rad2deg(np.arccos(np.clip((np.trace(rot_error_mat) - 1) / 2, -1, 1)))

    print(f'\n--- MaPS Estimator: Results ---')
    print(f'Position Error: {pos_error_maps:.4f} mm')
    print(f'Orientation Error: {angle_error_maps:.4f} deg')

    # 6. Run NLLS-LM Solver (starting from MaPS estimate)
    print(f'\n{"="*60}')
    print('Running NLLS-LM Solver (starting from MaPS estimate)...')
    print(f'{"="*60}')

    R_est_nlls, p_est_nlls, t_exec_nlls = run_nlls_lm(
        D_cal, sources, B_meas_cell,
        R_init=R_est, p_init=p_est, n_runs=n_runs
    )

    # NLLS-LM statistics
    avg_time_nlls = np.mean(t_exec_nlls) * 1e6
    median_time_nlls = np.median(t_exec_nlls) * 1e6
    std_time_nlls = np.std(t_exec_nlls) * 1e6
    min_time_nlls = np.min(t_exec_nlls) * 1e6
    max_time_nlls = np.max(t_exec_nlls) * 1e6

    print(f'\n--- NLLS-LM Solver: Computational Performance (over {n_runs} runs) ---')
    print(f'Average Execution Time: {avg_time_nlls:.2f} us')
    print(f'Median Execution Time:  {median_time_nlls:.2f} us')
    print(f'Standard Deviation:     {std_time_nlls:.2f} us')
    print(f'Min Execution Time:     {min_time_nlls:.2f} us')
    print(f'Max Execution Time:     {max_time_nlls:.2f} us')

    # NLLS-LM final results
    R_est_lm = R_est_nlls[-1]
    p_est_lm = p_est_nlls[:, -1]
    pos_error_nlls = np.linalg.norm(p_est_lm - p_true) * 1000  # mm
    rot_error_mat_lm = R_est_lm.T @ R_true
    angle_error_nlls = np.rad2deg(np.arccos(np.clip((np.trace(rot_error_mat_lm) - 1) / 2, -1, 1)))

    print(f'\n--- NLLS-LM Solver: Results ---')
    print(f'Position Error: {pos_error_nlls:.4f} mm')
    print(f'Orientation Error: {angle_error_nlls:.4f} deg')

    # 7. Summary Comparison
    print(f'\n{"="*60}')
    print('COMPARISON SUMMARY')
    print(f'{"="*60}')
    print(f'\n{"Method":<15} {"Avg Time (us)":<15} {"Median (us)":<15} {"Min (us)":<12} {"Pos Error (mm)":<15} {"Ori Error (deg)":<15}')
    print('-' * 90)
    print(f'{"MaPS":<15} {avg_time_maps:<15.2f} {median_time_maps:<15.2f} {min_time_maps:<12.2f} {pos_error_maps:<15.4f} {angle_error_maps:<15.4f}')
    print(f'{"NLLS-LM":<15} {avg_time_nlls:<15.2f} {median_time_nlls:<15.2f} {min_time_nlls:<12.2f} {pos_error_nlls:<15.4f} {angle_error_nlls:<15.4f}')

    speedup = avg_time_nlls / avg_time_maps if avg_time_maps > 0 else float('inf')
    print(f'\nMaPS is {speedup:.1f}x faster than NLLS-LM on average')

    # 8. Visualization
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.patches import FancyArrowPatch
        from mpl_toolkits.mplot3d import proj3d

        class Arrow3D(FancyArrowPatch):
            def __init__(self, xs, ys, zs, *args, **kwargs):
                FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
                self._verts3d = xs, ys, zs

            def do_3d_projection(self):
                xs = self._verts3d[0]
                ys = self._verts3d[1]
                zs = self._verts3d[2]
                xs, ys, zs = proj3d.proj_transform(xs, ys, zs, self.axes.M)
                self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
                return np.min(zs)

        fig = plt.figure(figsize=(16, 6))

        # Plot 1: 3D visualization with magnetic sources
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('MaPS: Pose Estimation with Magnetic Sources')

        # Plot sensor array center (estimated vs true)
        ax1.scatter(p_true[0], p_true[1], p_true[2],
                   c='green', marker='o', s=200, label='True Sensor Center', edgecolors='black')
        ax1.scatter(p_est[0], p_est[1], p_est[2],
                   c='red', marker='x', s=200, label='Est Sensor Center')

        # Plot sensor positions (true)
        for j in range(N):
            p_sensor = p_true + R_true @ D_cal[:, j]
            ax1.scatter(p_sensor[0], p_sensor[1], p_sensor[2],
                        c='blue', marker='^', s=50)

        # Plot magnetic sources as points + arrows
        colors = ['red', 'green', 'blue']
        for i, source in enumerate(sources):
            # Source position
            ax1.scatter(source['p_Ci'][0], source['p_Ci'][1], source['p_Ci'][2],
                       c=colors[i], marker='o', s=150, edgecolors='black')

            # Magnetic moment direction as arrow
            m_norm = source['m_Ci'] / np.linalg.norm(source['m_Ci'])
            arrow_length = 0.15  # 15cm arrow
            arrow = Arrow3D(
                [source['p_Ci'][0], source['p_Ci'][0] + m_norm[0] * arrow_length],
                [source['p_Ci'][1], source['p_Ci'][1] + m_norm[1] * arrow_length],
                [source['p_Ci'][2], source['p_Ci'][2] + m_norm[2] * arrow_length],
                mutation_scale=15, lw=2, arrowstyle='->', color=colors[i]
            )
            ax1.add_artist(arrow)
            ax1.text(source['p_Ci'][0], source['p_Ci'][1], source['p_Ci'][2],
                    f'  {source["name"]}', color=colors[i])

        ax1.legend(loc='upper left', fontsize=8)

        # Plot 2: Relative displacements
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        ax2.set_title('Relative Displacement Estimation (Sensor Frame)')

        for i in range(M):
            rho_true = R_true.T @ (p_true - sources[i]['p_Ci'])
            ax2.scatter(rho_true[0], rho_true[1], rho_true[2],
                       c=colors[i], marker='o', s=100, label=f'True rho{i}')
            rho_hat = details['rho_hats'][:, i]
            ax2.scatter(rho_hat[0], rho_hat[1], rho_hat[2],
                       c=colors[i], marker='x', s=100, label=f'Est rho{i}')

        ax2.legend()

        # Plot 3: Timing comparison
        ax3 = fig.add_subplot(133)
        methods = ['MaPS', 'NLLS-LM']
        avg_times = [avg_time_maps, avg_time_nlls]
        median_times = [median_time_maps, median_time_nlls]

        x = np.arange(len(methods))
        width = 0.35

        bars1 = ax3.bar(x - width/2, avg_times, width, label='Average', color=['steelblue', 'coral'])
        bars2 = ax3.bar(x + width/2, median_times, width, label='Median', color=['lightblue', 'lightsalmon'])

        ax3.set_ylabel('Execution Time (us)')
        ax3.set_title('Computational Performance Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(methods)
        ax3.legend()
        ax3.set_yscale('log')

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax3.annotate(f'{height:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax3.annotate(f'{height:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig('comparison_results.png', dpi=150)
        print('\nPlot saved to comparison_results.png')
    except ImportError as e:
        print(f'\nmatplotlib not installed or error: {e}')
        print('Skipping plot')


if __name__ == '__main__':
    main()
