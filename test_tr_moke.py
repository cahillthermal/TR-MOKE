"""
test_tr_moke.py
===============
Unit tests for the TRMOKEModel physical simulation.
"""

import numpy as np
from tr_moke_model import TRMOKEModel


def test_default_parameters():
    m = TRMOKEModel()
    assert np.isclose(m.d_film, 20e-9)
    assert np.isclose(m.G_interface, 100e6)
    assert np.isclose(m.C_v, 3e6)
    assert np.isclose(m.T_c, 330.0)
    assert np.isclose(m.T_0, 300.0)
    assert np.isclose(m.mag_exponent, 1.5)
    assert m.mag_model == "bloch"
    assert np.isclose(m.f_mod, 10e6)
    assert np.isclose(m.f_rep, 80e6)
    assert np.isclose(m.absorptance, 0.30)

    # Verify critical model is supported as an option
    m_crit = TRMOKEModel(mag_model="critical", mag_exponent=0.36)
    assert m_crit.mag_model == "critical"
    assert np.isclose(m_crit.mag_exponent, 0.36)


def test_thermal_properties():
    m = TRMOKEModel()
    # C_film = 3e6 * 20e-9 = 0.06 J/(m^2*K)
    assert np.isclose(m.C_film, 0.06)
    # tau_th = 0.06 / 1e8 = 600 ps
    assert np.isclose(m.tau_thermal, 600e-12)
    # delta_T_curie = 30 K
    assert np.isclose(m.delta_T_curie, 30.0)
    # critical absorbed fluence = 0.06 * 30 = 1.8 J/m^2
    assert np.isclose(m.critical_absorbed_fluence, 1.8)
    # critical incident fluence = 1.8 / 0.30 = 6.0 J/m^2
    assert np.isclose(m.critical_incident_fluence, 6.0)


def test_magnetization_cutoff():
    m = TRMOKEModel()
    # Below T_c: M > 0
    m_base = m.magnetization(300.0)
    assert m_base > 0.0
    # Exactly at or above T_c: M = 0
    assert np.isclose(m.magnetization(330.0), 0.0)
    assert np.isclose(m.magnetization(350.0), 0.0)


def test_duration_above_curie():
    m = TRMOKEModel()
    # Below critical fluence
    assert m.duration_above_curie(4.0) == 0.0
    # Exactly critical fluence
    assert np.isclose(m.duration_above_curie(6.0), 0.0)
    # Above critical fluence: t_above = tau_th * ln(F / F_crit)
    f = 12.0
    expected_t = 600e-12 * np.log(12.0 / 6.0)
    assert np.isclose(m.duration_above_curie(f), expected_t)


def test_single_pulse_asymptotics():
    m = TRMOKEModel(pulse_width_sigma=0.0)  # ideal step
    t_delays = np.array([-1e-9, 0.0, 600e-12, 4e-9])
    fluence = 6.0
    dT, T = m.single_pulse_thermal_profile(t_delays, fluence, use_convolution=False)

    # Before pulse
    assert np.isclose(dT[0], 0.0)
    assert np.isclose(T[0], 300.0)

    # At t = 0
    assert np.isclose(dT[1], 30.0)
    assert np.isclose(T[1], 330.0)

    # At t = tau_th
    assert np.isclose(dT[2], 30.0 * np.exp(-1.0))

    # At t = 4 ns (~6.67 tau_th)
    assert np.isclose(dT[3], 30.0 * np.exp(-4.0 / 0.6), atol=0.1)


def test_lockin_periodicity():
    m = TRMOKEModel(accum_ratio=0.0)
    t_delays = np.linspace(-20e-12, 4e-9, 50)
    v_in, v_out, v_amp = m.periodic_lockin_response(t_delays, incident_fluence=3.0)

    assert len(v_in) == 50
    assert np.all(np.isfinite(v_in))
    assert np.all(np.isfinite(v_out))
    # In-phase signal at t < 0 should be near zero when accum_ratio = 0
    assert np.abs(v_in[0]) < np.abs(np.min(v_in)) * 0.05


def test_calibrated_lockin_leakage():
    m = TRMOKEModel()
    t_delays = np.linspace(-20e-12, 4e-9, 100)
    
    # At low fluence (1.0 J/m^2), out-of-phase signal has virtually zero step
    vin_low, vout_low, phi_0 = m.calibrated_lockin_response(t_delays, incident_fluence=1.0)
    step_vout_low = vout_low[10] - vout_low[0]
    assert np.isclose(step_vout_low, 0.0, atol=1e-4)

    # At high fluence (12.0 J/m^2), non-linearity causes significant step in V_out
    vin_high, vout_high, _ = m.calibrated_lockin_response(t_delays, incident_fluence=12.0)
    step_vout_high = vout_high[10] - vout_high[0]
    assert np.abs(step_vout_high) > 1e-3


def test_spatial_averaging_linear_half():
    m = TRMOKEModel(pulse_width_sigma=0.0)
    t_delays = np.array([0.0])
    f_peak = 2.0  # low fluence (linear)
    
    # Peak on-axis temperature jump:
    dT_on_axis = m.pulse_temperature_jump(f_peak)
    
    # Spatially-averaged temperature jump:
    avg_dT, _ = m.spatially_averaged_single_pulse_thermal(t_delays, f_peak, use_convolution=False)
    
    # Exact analytical result for equal Gaussian beams is exactly half:
    assert np.isclose(avg_dT[0], 0.5 * dT_on_axis, rtol=1e-3)


def test_magnetization_averaging_vs_temperature_averaging():
    """
    Confirms that our calculation calculates <M(T(r))> (the true probe-weighted average
    of local magnetization) and NOT M(<T>), which would be physically incorrect due to
    the severe non-linearity of magnetization and the Curie temperature cutoff.
    """
    m = TRMOKEModel(pulse_width_sigma=0.0)
    t_delays = np.array([0.0])
    f_peak = 8.0  # J/m^2 (peak on-axis T is 340 K > T_c, but <T> is 320 K < T_c)
    
    # 1. Average temperature <T>:
    _, avg_T = m.spatially_averaged_single_pulse_thermal(t_delays, f_peak, use_convolution=False)
    # Average temperature rise is 20 K -> <T> = 320 K < T_c = 330 K
    assert avg_T[0] < m.T_c
    M_of_avg_T = m.magnetization(avg_T[0])
    
    # 2. Correct probe-weighted average of magnetization <M(T)>:
    avg_M, _, _ = m.spatially_averaged_moke_response(t_delays, f_peak, use_convolution=False)
    
    # Due to the severe non-linearity and M(T) >= 0 cutoff at T_c,
    # <M(T)> differs significantly from M(<T>)!
    assert not np.isclose(avg_M[0], M_of_avg_T, rtol=0.05)
    
    # Demagnetization fraction:
    demag_correct = (m.M_base - avg_M[0]) / m.M_base
    demag_incorrect = (m.M_base - M_of_avg_T) / m.M_base
    
    print(f"\nVerification of non-linear spatial averaging at F_peak = {f_peak:.1f} J/m^2:")
    print(f"  Peak on-axis temperature : 340.0 K (exceeds T_c = 330 K)")
    print(f"  Average temperature <T>  : {avg_T[0]:.1f} K (below T_c)")
    print(f"  INCORRECT M(<T>) demag   : {demag_incorrect * 100:.2f}% (evaluating M at average T)")
    print(f"  CORRECT <M(T(r))> demag  : {demag_correct * 100:.2f}% (true probe average of local magnetization)")
    print(f"  Discrepancy              : {abs(demag_correct - demag_incorrect)*100:.2f}%")


if __name__ == "__main__":
    test_default_parameters()
    test_thermal_properties()
    test_magnetization_cutoff()
    test_duration_above_curie()
    test_single_pulse_asymptotics()
    test_lockin_periodicity()
    test_calibrated_lockin_leakage()
    test_spatial_averaging_linear_half()
    test_magnetization_averaging_vs_temperature_averaging()
    print("All unit tests passed successfully!")
