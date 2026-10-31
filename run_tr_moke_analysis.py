"""
run_tr_moke_analysis.py
=======================
Executes the non-linear TR-MOKE calculation for a ferromagnetic thin film transducer,
taking into account that both the pump and probe have Gaussian intensity profiles
with a 1/e^2 intensity radius of 10 microns.

Compares:
1. Local on-axis response vs. probe-weighted spatial average.
2. What happens as the pump pulse energy increases from the linear regime
   up to and well above the Curie temperature (T_c = 330 K).
3. The behavior of the in-phase (V_in) and out-of-phase (V_out) lock-in signals
   as a function of delay time, demonstrating how non-linear demagnetization and
   phase-lagged thermal accumulation cause V_out to develop an emergent step and decay
   at high fluence.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from tr_moke_model import TRMOKEModel


def run_analysis():
    print("=" * 80)
    print("TIME-RESOLVED MAGNETO-OPTIC KERR EFFECT (TR-MOKE) NON-LINEAR MODEL")
    print("WITH GAUSSIAN SPATIAL PROFILES (w_pump = w_probe = 10 microns)")
    print("=" * 80)

    # Instantiate model with parameters from prompt
    model = TRMOKEModel(w_pump=10e-6, w_probe=10e-6)

    # Print summary of system parameters
    print("\n--- Physical Model Parameters ---")
    for key, val in model.summary_dict().items():
        print(f"  {key:<35}: {val}")

    # Delay time array: -0.5 ns (-500 ps) to 4 ns (4000 ps)
    t_delays_ps = np.concatenate([
        np.linspace(-500.0, -20.0, 50),
        np.linspace(-20.0, -2.0, 40),
        np.linspace(-2.0, 10.0, 121),
        np.linspace(10.0, 500.0, 150),
        np.linspace(500.0, 4000.0, 200),
    ])
    t_delays = np.unique(t_delays_ps) * 1e-12  # in seconds

    # Pump incident peak fluences to compare:
    # Critical incident fluence to reach T_c on axis is F_crit = 6.0 J/m^2 (0.60 mJ/cm^2)
    fluences = np.array([1.5, 3.0, 4.5, 6.0, 8.0, 12.0, 16.0])

    print("\n--- Fluence Sweep: Local On-Axis vs. Spatially-Averaged Demagnetization ---")
    header = (
        f"{'F_peak (J/m²)':<15} {'F (mJ/cm²)':<13} {'Peak T_0 (K)':<14} "
        f"{'T_0 > T_c?':<12} {'Local Max (%)':<16} {'Spatially Avg Max (%)':<22}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for F in fluences:
        delta_T_peak = model.pulse_temperature_jump(F)
        T_peak_axis = model.T_0 + delta_T_peak
        exceeds_Tc = T_peak_axis >= model.T_c
        t_above_ps = model.duration_above_curie(F) * 1e12

        # Local on-axis profiles
        _, T_profile_local = model.single_pulse_thermal_profile(t_delays, F)
        M_profile_local, delta_M_local, frac_dM_local = model.single_pulse_moke_response(t_delays, F)

        # Spatially-averaged profiles (convolved with Gaussian probe profile)
        _, T_profile_avg = model.spatially_averaged_single_pulse_thermal(t_delays, F)
        M_profile_avg, delta_M_avg, frac_dM_avg = model.spatially_averaged_moke_response(t_delays, F)

        # Calibrated lock-in response (spatially averaged)
        vin_avg, vout_avg, phi_0 = model.spatially_averaged_calibrated_lockin(t_delays, F, calib_fluence=1.0)

        max_demag_local_pct = -np.min(delta_M_local) / model.M_base * 100.0
        max_demag_avg_pct = -np.min(delta_M_avg) / model.M_base * 100.0

        results.append({
            "fluence": F,
            "T_peak_axis": T_peak_axis,
            "delta_T_peak": delta_T_peak,
            "exceeds_Tc": exceeds_Tc,
            "t_above_ps": t_above_ps,
            "t_delays": t_delays,
            "T_profile_local": T_profile_local,
            "T_profile_avg": T_profile_avg,
            "M_profile_local": M_profile_local,
            "M_profile_avg": M_profile_avg,
            "frac_dM_local": frac_dM_local,
            "frac_dM_avg": frac_dM_avg,
            "vin_avg": vin_avg,
            "vout_avg": vout_avg,
            "phi_0": phi_0,
            "max_demag_local_pct": max_demag_local_pct,
            "max_demag_avg_pct": max_demag_avg_pct,
        })

        exceeds_str = "YES" if exceeds_Tc else "No"
        print(
            f"{F:<15.2f} {F*0.1:<13.3f} {T_peak_axis:<14.1f} {exceeds_str:<12} "
            f"{max_demag_local_pct:<16.2f} {max_demag_avg_pct:<22.2f}"
        )

    # =========================================================================
    # FIGURE 1: Comprehensive Comparison of Spatially-Averaged Response vs Delay
    # =========================================================================
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.autolayout": True,
    })

    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    cmap = plt.cm.plasma
    colors = [cmap(i) for i in np.linspace(0.05, 0.90, len(fluences))]

    # Plot (a): Probe-Averaged Film Temperature <T(t_d)>
    ax_temp = axes[0, 0]
    for r, c in zip(results, colors):
        label = f"{r['fluence']:.1f} J/m² (Peak {r['T_peak_axis']:.1f} K)"
        ax_temp.plot(r["t_delays"] * 1e9, r["T_profile_avg"], color=c, lw=2.0, label=label)
    ax_temp.axhline(model.T_c, color="tab:red", ls="--", lw=1.5, label=f"Curie T_c = {model.T_c:.0f} K")
    ax_temp.axhline(model.T_0, color="gray", ls=":", lw=1.2, label=f"Base T_0 = {model.T_0:.0f} K")
    ax_temp.axvline(0.0, color="gray", ls=":", lw=1.2, alpha=0.7)
    ax_temp.set_ylabel("Spatially-Averaged Temp ⟨T⟩ (K)")
    ax_temp.set_title("(a) Probe-Averaged Temperature Response ⟨T(t_d)⟩")
    ax_temp.set_xlim(-0.5, 4.0)
    ax_temp.grid(True, alpha=0.3)
    ax_temp.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    # Plot (b): Spatially-Averaged Magnetization <M(t_d)> / M_0
    ax_mag = axes[0, 1]
    for r, c in zip(results, colors):
        ax_mag.plot(r["t_delays"] * 1e9, r["M_profile_avg"] / model.M_0, color=c, lw=2.0, label=f"{r['fluence']:.1f} J/m²")
    ax_mag.axhline(0.0, color="tab:red", ls="--", lw=1.2, alpha=0.7)
    ax_mag.axvline(0.0, color="gray", ls=":", lw=1.2, alpha=0.7)
    ax_mag.set_ylabel("Normalized Magnetization ⟨M(t_d)⟩ / M_0")
    ax_mag.set_title("(b) Spatially-Averaged Magnetization Dynamics")
    ax_mag.set_xlim(-0.5, 4.0)
    ax_mag.set_ylim(-0.01, model.M_base * 1.1)
    ax_mag.grid(True, alpha=0.3)
    ax_mag.legend(loc="lower right", framealpha=0.9, fontsize=8.5)

    # Plot (c): Spatially-Averaged Demagnetization -<Delta M(t_d)> / M(T_0)
    ax_frac = axes[1, 0]
    for r, c in zip(results, colors):
        ax_frac.plot(r["t_delays"] * 1e9, -r["frac_dM_avg"], color=c, lw=2.0, label=f"{r['fluence']:.1f} J/m²")
    ax_frac.axhline(1.0, color="tab:red", ls="--", lw=1.5, label="Complete Local Demagnetization (100%)")
    ax_frac.axvline(0.0, color="gray", ls=":", lw=1.2, alpha=0.7)
    ax_frac.set_ylabel(r"Demagnetization $-\langle\Delta M\rangle / M(T_0)$")
    ax_frac.set_title(r"(c) Spatially-Averaged TR-MOKE Signal $-\langle\Delta M(t_d)\rangle / M(T_0)$")
    ax_frac.set_xlim(-0.5, 4.0)
    ax_frac.set_ylim(-0.05, 1.15)
    ax_frac.grid(True, alpha=0.3)
    ax_frac.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    # Plot (d): Fluence-Normalized Signal: (-<Delta M>) / F_peak
    ax_norm = axes[1, 1]
    for r, c in zip(results, colors):
        norm_sig = (-r["frac_dM_avg"]) / r["fluence"]
        ax_norm.plot(r["t_delays"] * 1e9, norm_sig, color=c, lw=2.0, label=f"{r['fluence']:.1f} J/m²")
    ax_norm.axvline(0.0, color="gray", ls=":", lw=1.2, alpha=0.7)
    ax_norm.set_ylabel(r"Fluence-Normalized Signal $[-\langle\Delta M\rangle / M(T_0)] / F_{\mathrm{peak}}$")
    ax_norm.set_title(r"(d) Linearity Test: Probe Signal Divided by Peak Fluence")
    ax_norm.set_xlim(-0.5, 4.0)
    ax_norm.grid(True, alpha=0.3)
    ax_norm.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    # Plot (e): Comparison of Local (Dashed) vs Spatially-Averaged (Solid)
    ax_compare = axes[2, 0]
    for r, c in zip(results[3:], colors[3:]):  # fluences >= 6.0 J/m^2
        ax_compare.plot(r["t_delays"] * 1e9, -r["frac_dM_avg"], color=c, lw=2.2, label=f"Avg: {r['fluence']:.1f} J/m²")
        ax_compare.plot(r["t_delays"] * 1e9, -r["frac_dM_local"], color=c, ls="--", lw=1.5, alpha=0.7)
    ax_compare.axvline(0.0, color="gray", ls=":", lw=1.2, alpha=0.7)
    ax_compare.plot([], [], color="black", lw=2.2, label="Spatially Averaged")
    ax_compare.plot([], [], color="black", ls="--", lw=1.5, label="Local On-Axis")
    ax_compare.set_xlabel("Probe Time Delay t_d (ns)")
    ax_compare.set_ylabel(r"Demagnetization $-\Delta M / M(T_0)$")
    ax_compare.set_title("(e) Local On-Axis vs Spatially Averaged Comparison")
    ax_compare.set_xlim(-0.5, 4.0)
    ax_compare.set_ylim(-0.05, 1.15)
    ax_compare.grid(True, alpha=0.3)
    ax_compare.legend(loc="upper right", framealpha=0.9, fontsize=8.0)

    # Plot (f): Early-delay zoom (-500 ps to 500 ps) showing spatial softening of saturation
    ax_zoom = axes[2, 1]
    for r, c in zip(results, colors):
        ax_zoom.plot(r["t_delays"] * 1e12, -r["frac_dM_avg"], color=c, lw=2.0, label=f"{r['fluence']:.1f} J/m²")
    ax_zoom.axhline(1.0, color="tab:red", ls="--", lw=1.5)
    ax_zoom.axvline(0.0, color="gray", ls=":", lw=1.2, alpha=0.7)
    ax_zoom.set_xlabel("Probe Time Delay t_d (ps)")
    ax_zoom.set_ylabel(r"Demagnetization $-\langle\Delta M\rangle / M(T_0)$")
    ax_zoom.set_title("(f) Ultrafast Kinetics (-500 ps to +500 ps)")
    ax_zoom.set_xlim(-500.0, 500.0)
    ax_zoom.set_ylim(-0.05, 1.10)
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.legend(loc="lower right", framealpha=0.9, fontsize=8.5)

    fig.suptitle(
        f"Non-Linear TR-MOKE with Gaussian Beam Profiles (w_pump = w_probe = 10 µm)\n"
        f"Film: d = {model.d_film*1e9:.0f} nm, G = {model.G_interface/1e6:.0f} MW/m²K, "
        f"T_c = {model.T_c:.0f} K, T_0 = {model.T_0:.0f} K (Critical Peak Fluence = {model.critical_incident_fluence:.1f} J/m²)",
        fontsize=13,
        fontweight="bold",
    )
    plt.savefig("tr_moke_nonlinear_comparison.png", dpi=300)
    print("\n[+] Saved main comparison plot to 'tr_moke_nonlinear_comparison.png'")

    # =========================================================================
    # FIGURE 2: Peak Signal Saturation: Local On-Axis vs Spatially-Averaged
    # =========================================================================
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    dense_fluences = np.linspace(0.1, 25.0, 300)
    t_peak_eval = np.array([50e-12])  # 50 ps delay (at peak demagnetization)

    peak_demag_local = []
    peak_demag_avg = []

    for f in dense_fluences:
        _, delta_M_loc, _ = model.single_pulse_moke_response(t_peak_eval, f)
        _, delta_M_spat, _ = model.spatially_averaged_moke_response(t_peak_eval, f)
        peak_demag_local.append(-delta_M_loc[0] / model.M_base)
        peak_demag_avg.append(-delta_M_spat[0] / model.M_base)

    peak_demag_local = np.array(peak_demag_local)
    peak_demag_avg = np.array(peak_demag_avg)

    # Linear response approximation (taking into account spatial averaging factor of 1/2):
    # In linear regime with equal Gaussian beams: <Delta T> = 0.5 * Delta T_peak
    dM_dT_0 = model.dM_dT(model.T_0)
    linear_demag_local = -(dM_dT_0 * np.array([model.pulse_temperature_jump(f) for f in dense_fluences])) / model.M_base
    linear_demag_avg = 0.5 * linear_demag_local

    # Plot 1: Peak Demagnetization vs Incident Fluence
    ax1.plot(dense_fluences, peak_demag_local, color="tab:blue", ls="--", lw=2.0, label="Local On-Axis Response")
    ax1.plot(dense_fluences, peak_demag_avg, color="tab:purple", lw=2.5, label="Spatially-Averaged (Gaussian w = 10 µm)")
    ax1.plot(dense_fluences, linear_demag_avg, color="tab:green", ls=":", lw=2.0, label="Linear Approx (with spatial avg)")
    ax1.axvline(model.critical_incident_fluence, color="tab:red", ls=":", lw=2.0, label=f"Critical Fluence F_crit = {model.critical_incident_fluence:.1f} J/m²")
    ax1.axhline(1.0, color="gray", ls="--", lw=1.2, label="Complete Demagnetization (100%)")

    # Scatter points for the simulated cases
    for r in results:
        ax1.scatter(r["fluence"], -np.min(r["frac_dM_avg"]), color="crimson", s=45, zorder=5)

    ax1.set_xlabel("Peak Incident Pump Fluence F_peak (J/m²)")
    ax1.set_ylabel(r"Peak Demagnetization $|\Delta M_{\mathrm{peak}}| / M(T_0)$")
    ax1.set_title("Peak Demagnetization: Local vs Spatially Averaged")
    ax1.set_xlim(0.0, 25.0)
    ax1.set_ylim(0.0, 1.25)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", framealpha=0.9, fontsize=9)

    # Plot 2: Demagnetized Area Radius r_c(F)
    # Inside radius r_c, local fluence exceeds F_crit, so T > T_c:
    # F(r_c) = F_peak * exp(-2 * r_c^2 / w^2) = F_crit => r_c = w * sqrt(0.5 * ln(F_peak / F_crit))
    r_c_um = np.zeros_like(dense_fluences)
    mask_above = dense_fluences > model.critical_incident_fluence
    r_c_um[mask_above] = (model.w_pump * 1e6) * np.sqrt(0.5 * np.log(dense_fluences[mask_above] / model.critical_incident_fluence))

    ax2.plot(dense_fluences, r_c_um, color="tab:red", lw=2.5, label="Demagnetized Spot Radius $r_c$ ($T > T_c$)")
    ax2.axhline(10.0, color="tab:blue", ls="--", lw=1.5, label="Probe Beam 1/e² Radius ($w = 10\ \mu\mathrm{m}$)")
    ax2.axvline(model.critical_incident_fluence, color="tab:red", ls=":", lw=2.0, label=f"Critical Fluence ({model.critical_incident_fluence:.1f} J/m²)")

    ax2.set_xlabel("Peak Incident Pump Fluence F_peak (J/m²)")
    ax2.set_ylabel("Demagnetized Spot Radius $r_c$ (µm)")
    ax2.set_title("Expansion of Demagnetized Spot Size Across Probe Beam")
    ax2.set_xlim(0.0, 25.0)
    ax2.set_ylim(0.0, 15.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=9)

    fig2.suptitle(
        f"Spatial Averaging Effect: Gradual Saturation vs Demagnetized Spot Expansion\n"
        f"Pump & Probe 1/e² Radii w_0 = w_1 = 10 µm",
        fontsize=13,
        fontweight="bold",
    )
    plt.savefig("tr_moke_saturation_analysis.png", dpi=300)
    print("[+] Saved saturation analysis plot to 'tr_moke_saturation_analysis.png'")

    # =========================================================================
    # FIGURE 3: Spatially-Averaged Lock-In Signals: Emergence of V_out Step at High Fluence
    # =========================================================================
    fig3, (ax_in, ax_out) = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    idx_neg = np.argmin(np.abs(t_delays - (-100e-12)))
    idx_pos = np.argmin(np.abs(t_delays - (+20e-12)))

    print("\n--- Spatially-Averaged Lock-In Analysis: Emergence of V_out Step at High Fluence ---")
    header_lockin = f"{'F_peak (J/m²)':<16} {'Step V_in':<14} {'Step V_out':<14} {'V_out/V_in (%)':<16} {'Phase Shift (deg)':<18}"
    print(header_lockin)
    print("-" * len(header_lockin))

    for r, c in zip(results, colors):
        vin = r["vin_avg"]
        vout = r["vout_avg"]
        step_vin = vin[idx_pos] - vin[idx_neg]
        step_vout = vout[idx_pos] - vout[idx_neg]
        leakage_pct = (step_vout / step_vin) * 100.0 if step_vin != 0 else 0.0
        phase_shift_deg = np.arctan2(step_vout, step_vin) * 180 / np.pi

        print(f"{r['fluence']:<16.2f} {step_vin:<14.5f} {step_vout:<14.5f} {leakage_pct:<16.2f} {phase_shift_deg:<18.2f}")

        ax_in.plot(t_delays * 1e9, -vin / model.M_base, color=c, lw=2.0, label=f"{r['fluence']:.1f} J/m²")
        ax_out.plot(t_delays * 1e9, -vout / model.M_base, color=c, lw=2.0, label=f"{r['fluence']:.1f} J/m²")

    ax_in.axvline(0.0, color="gray", ls=":", lw=1.2, alpha=0.7)
    ax_in.set_xlabel("Probe Time Delay t_d (ns)")
    ax_in.set_ylabel(r"Spatially-Averaged In-Phase Signal $-\langle V_{\mathrm{in}}\rangle / M(T_0)$")
    ax_in.set_title("(a) In-Phase Signal V_in(t_d)")
    ax_in.set_xlim(-0.5, 4.0)
    ax_in.grid(True, alpha=0.3)
    ax_in.legend(loc="upper right", fontsize=8.5)

    ax_out.axvline(0.0, color="gray", ls=":", lw=1.2, alpha=0.7)
    ax_out.set_xlabel("Probe Time Delay t_d (ns)")
    ax_out.set_ylabel(r"Spatially-Averaged Out-of-Phase Signal $-\langle V_{\mathrm{out}}\rangle / M(T_0)$")
    ax_out.set_title("(b) Out-of-Phase Signal V_out(t_d): Non-Linear Step Leakage")
    ax_out.set_xlim(-0.5, 4.0)
    ax_out.grid(True, alpha=0.3)
    ax_out.legend(loc="upper right", fontsize=8.5)

    fig3.suptitle(
        "Spatially-Averaged Lock-In Signals: Emergence of Time-Dependent Out-of-Phase Signal at High Fluence\n"
        "(Concentric Gaussian Pump & Probe w = 10 µm; Lock-in phase calibrated at low fluence to make V_out smooth)",
        fontsize=13,
        fontweight="bold",
    )
    plt.savefig("tr_moke_lockin_phase_leakage.png", dpi=300)
    print("[+] Saved lock-in phase leakage plot to 'tr_moke_lockin_phase_leakage.png'")

    print("\n[+] Calculation and visualization complete!")
    print("=" * 80)


if __name__ == "__main__":
    run_analysis()
