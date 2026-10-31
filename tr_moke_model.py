"""
tr_moke_model.py
================
Physical model of the non-linear response of a ferromagnetic thin film transducer
in a Time-Resolved Magneto-Optic Kerr Effect (TR-MOKE) experiment.

Physical Assumptions:
- Thin metal film on a substrate with extremely high thermal conductivity (infinite heat sink).
- Film cools to the substrate via interfacial thermal conductance G.
- Thermal response is governed by a simple exponential decay:
      d(Delta_T)/dt = - (G / C_film) * Delta_T = - Delta_T / tau_th
- Magnetization M(T) follows a power law below Curie temperature T_c:
      M(T) = M_0 * [1 - (T / T_c)^gamma]   (Bloch law, default gamma = 3/2)
      or M_0 * (1 - T / T_c)^gamma
  and drops to 0 for T >= T_c.
- Probe pulse is weak (negligible heating) and samples Kerr rotation theta_K proportional to M(T).
- Laser repetition rate (f_rep = 80 MHz) and pump modulation (f_mod = 10 MHz)
  with lock-in detection at f_mod.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union
import numpy as np
from scipy.special import erf


@dataclass
class TRMOKEModel:
    """
    Time-Resolved Magneto-Optic Kerr Effect (TR-MOKE) model for a ferromagnetic thin film.

    Parameters:
    -----------
    d_film : float
        Thickness of the metal film in meters. Default: 20 nm (20e-9 m).
    G_interface : float
        Thermal conductance per unit area between film and substrate in W/(m^2*K).
        Default: 100 MW/(m^2*K) (100e6 W/(m^2*K)).
    C_v : float
        Volumetric heat capacity of the metal film in J/(m^3*K).
        Default: 3 J/(cm^3*K) = 3e6 J/(m^3*K).
    T_c : float
        Curie temperature of the ferromagnetic film in Kelvin. Default: 330 K.
    T_0 : float
        Base temperature of the sample in Kelvin. Default: 300 K.
    mag_exponent : float
        Exponent describing the temperature dependence of magnetization below T_c.
        Default: 1.5 (3/2, Bloch power law).
    f_mod : float
        Modulation frequency of the pump beam in Hertz. Default: 10 MHz (10e6 Hz).
    f_rep : float
        Repetition rate of the pulsed laser in Hertz. Default: 80 MHz (80e6 Hz).
    absorptance : float
        Fraction of incident pump energy absorbed by the film. Default: 0.30 (30%).
    M_0 : float
        Saturation magnetization at T = 0 K in arbitrary units. Default: 1.0.
    mag_model : str
        Functional form for M(T) below T_c:
        - 'bloch': M(T) = M_0 * (1 - (T/T_c)^gamma) (default, gamma = 1.5)
        - 'critical': M(T) = M_0 * (1 - T/T_c)^gamma
        Default: 'bloch'.
    pulse_width_sigma : float
        Gaussian pulse width sigma (in seconds) for instrument response convolution.
        Set to 0 for ideal instantaneous step response. Default: 1.5e-12 (1.5 ps).
    """

    d_film: float = 20e-9
    G_interface: float = 100e6
    C_v: float = 3e6
    T_c: float = 330.0
    T_0: float = 300.0
    mag_exponent: float = 1.5
    f_mod: float = 10e6
    f_rep: float = 80e6
    absorptance: float = 0.30
    M_0: float = 1.0
    mag_model: Literal["bloch", "critical"] = "bloch"
    pulse_width_sigma: float = 1.5e-12
    accum_ratio: float = 0.20
    thermal_phase_lag: float = 35.0
    w_pump: float = 10e-6
    w_probe: float = 10e-6
    n_spatial_nodes: int = 50

    @property
    def C_film(self) -> float:
        """Areal heat capacity of the metal film (J / (m^2 * K))."""
        return self.C_v * self.d_film

    @property
    def tau_thermal(self) -> float:
        """
        Characteristic thermal decay time constant tau_th = C_film / G_interface (seconds).
        For 20 nm film, C_v = 3 J/(cm^3*K), G = 100 MW/(m^2*K):
            C_film = 0.06 J/(m^2*K)
            tau_th = 0.06 / 1e8 = 600 ps.
        """
        return self.C_film / self.G_interface

    @property
    def tau_rep(self) -> float:
        """Time between laser pulses (seconds): 1 / f_rep (12.5 ns for 80 MHz)."""
        return 1.0 / self.f_rep

    @property
    def T_mod(self) -> float:
        """Period of the pump modulation (seconds): 1 / f_mod (100 ns for 10 MHz)."""
        return 1.0 / self.f_mod

    @property
    def pulses_per_mod_cycle(self) -> int:
        """Number of laser pulses in one pump modulation period: f_rep / f_mod (8 for 80/10)."""
        return int(round(self.f_rep / self.f_mod))

    @property
    def delta_T_curie(self) -> float:
        """Temperature rise needed to reach Curie temperature from base temperature (K)."""
        return max(0.0, self.T_c - self.T_0)

    @property
    def critical_absorbed_fluence(self) -> float:
        """
        Minimum absorbed energy per unit area (J/m^2) to reach T_c from T_0 in a single pulse:
            F_abs,crit = C_film * (T_c - T_0)
        """
        return self.C_film * self.delta_T_curie

    @property
    def critical_incident_fluence(self) -> float:
        """
        Minimum incident pump energy per unit area (J/m^2) to reach T_c from T_0 in a single pulse:
            F_crit = F_abs,crit / absorptance
        """
        return self.critical_absorbed_fluence / self.absorptance

    @property
    def M_base(self) -> float:
        """Magnetization at base temperature T_0."""
        return float(self.magnetization(self.T_0))

    def magnetization(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the spontaneous magnetization M(T) at temperature T.

        Parameters:
        -----------
        T : float or np.ndarray
            Temperature in Kelvin.

        Returns:
        --------
        M : float or np.ndarray
            Spontaneous magnetization in units of M_0. Zero for T >= T_c.
        """
        T_arr = np.asarray(T, dtype=float)
        M = np.zeros_like(T_arr)
        mask = (T_arr >= 0.0) & (T_arr < self.T_c)

        if self.mag_model == "bloch":
            # Bloch T^alpha law: M(T) = M_0 * (1 - (T / T_c)^alpha)
            M[mask] = self.M_0 * (1.0 - (T_arr[mask] / self.T_c) ** self.mag_exponent)
        elif self.mag_model == "critical":
            # Scaling form: M(T) = M_0 * (1 - T / T_c)^alpha
            M[mask] = self.M_0 * ((1.0 - T_arr[mask] / self.T_c) ** self.mag_exponent)
        else:
            raise ValueError(f"Unknown mag_model: {self.mag_model}")

        if np.isscalar(T):
            return float(M.item())
        return M

    def dM_dT(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Derivative of magnetization with respect to temperature dM/dT.
        Used for analyzing linear response limits.
        """
        T_arr = np.asarray(T, dtype=float)
        dM = np.zeros_like(T_arr)
        mask = (T_arr >= 0.0) & (T_arr < self.T_c)

        if self.mag_model == "bloch":
            dM[mask] = -self.M_0 * (self.mag_exponent / self.T_c) * (T_arr[mask] / self.T_c) ** (self.mag_exponent - 1.0)
        elif self.mag_model == "critical":
            dM[mask] = -self.M_0 * (self.mag_exponent / self.T_c) * ((1.0 - T_arr[mask] / self.T_c) ** (self.mag_exponent - 1.0))

        if np.isscalar(T):
            return float(dM.item())
        return dM

    def pulse_temperature_jump(self, incident_fluence: float) -> float:
        """
        Calculate instantaneous temperature rise Delta_T_jump (K) from one pump pulse.
            Delta_T_jump = (absorptance * F_incident) / C_film
        """
        absorbed_fluence = self.absorptance * incident_fluence
        return absorbed_fluence / self.C_film

    def duration_above_curie(self, incident_fluence: float) -> float:
        """
        Analytical time (seconds) during which the film temperature remains above T_c
        following a single pump pulse of given incident fluence.
        If Delta_T_jump <= T_c - T_0, returns 0.0.
            t_above = tau_th * ln( Delta_T_jump / (T_c - T_0) )
        """
        delta_T_jump = self.pulse_temperature_jump(incident_fluence)
        if delta_T_jump <= self.delta_T_curie or self.delta_T_curie == 0:
            return 0.0
        return self.tau_thermal * np.log(delta_T_jump / self.delta_T_curie)

    def single_pulse_thermal_profile(
        self,
        t_delay: np.ndarray,
        incident_fluence: float,
        use_convolution: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute temperature rise Delta_T(t_d) and absolute temperature T(t_d) for a single pulse.

        Parameters:
        -----------
        t_delay : np.ndarray
            Probe delay times in seconds (e.g. from -20 ps to 4 ns).
        incident_fluence : float
            Pump pulse energy per unit area in J/m^2.
        use_convolution : bool
            If True and pulse_width_sigma > 0, convolves exponential decay with Gaussian pulse.
            If False or pulse_width_sigma == 0, uses Heaviside step function.

        Returns:
        --------
        delta_T : np.ndarray
            Temperature rise relative to base temperature (K).
        T : np.ndarray
            Absolute temperature T_0 + delta_T (K).
        """
        delta_T_jump = self.pulse_temperature_jump(incident_fluence)
        tau = self.tau_thermal
        sigma = self.pulse_width_sigma

        if use_convolution and sigma > 0:
            # Analytical convolution of exponential decay H(t)*exp(-t/tau) with Gaussian pulse
            # G(t) = 1/(sqrt(2*pi)*sigma) * exp(-t^2 / (2*sigma^2))
            # Integral: 1/2 * exp((sigma^2 / (2*tau^2)) - (t / tau)) * [1 + erf((t - sigma^2/tau) / (sqrt(2)*sigma))]
            arg_exp = (sigma**2 / (2.0 * tau**2)) - (t_delay / tau)
            arg_erf = (t_delay - (sigma**2 / tau)) / (np.sqrt(2.0) * sigma)
            
            # Prevent overflow for large negative delay
            delta_T = 0.5 * delta_T_jump * np.exp(np.clip(arg_exp, -100.0, 100.0)) * (1.0 + erf(arg_erf))
            # Zero out any numerical artifacts at deep negative delay
            delta_T[t_delay < -5.0 * sigma] = 0.0
        else:
            delta_T = np.where(t_delay >= 0.0, delta_T_jump * np.exp(-t_delay / tau), 0.0)

        T = self.T_0 + delta_T
        return delta_T, T

    def single_pulse_moke_response(
        self,
        t_delay: np.ndarray,
        incident_fluence: float,
        use_convolution: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute instantaneous MOKE signals as a function of probe delay for a single pulse.

        Returns:
        --------
        M : np.ndarray
            Magnetization M(t_d).
        delta_M : np.ndarray
            Kerr signal change Delta_M(t_d) = M(t_d) - M(T_0).
        fractional_delta_M : np.ndarray
            Normalized Kerr signal change: (M(t_d) - M(T_0)) / M(T_0).
        """
        _, T = self.single_pulse_thermal_profile(t_delay, incident_fluence, use_convolution)
        M = self.magnetization(T)
        m_base = self.M_base
        delta_M = M - m_base
        fractional_delta_M = delta_M / (m_base if m_base != 0 else 1.0)
        return M, delta_M, fractional_delta_M

    def periodic_lockin_response(
        self,
        t_delay: np.ndarray,
        incident_fluence: float,
        modulation_type: Literal["square", "sine"] = "square",
        harmonic: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate the periodic steady-state response of the film under modulated pump pulses
        and calculate the lock-in amplifier in-phase (X) and out-of-phase (Y) signals
        at modulation frequency f_mod as a function of delay time t_d.

        Parameters:
        -----------
        t_delay : np.ndarray
            Probe delay times in seconds.
        incident_fluence : float
            Peak incident pump pulse energy per unit area in J/m^2.
        modulation_type : 'square' or 'sine'
            Pump modulation waveform envelope at f_mod.
        harmonic : int
            Lock-in detection harmonic (default 1 = fundamental f_mod).

        Returns:
        --------
        V_in : np.ndarray
            In-phase lock-in signal X(t_d).
        V_out : np.ndarray
            Out-of-phase lock-in signal Y(t_d).
        V_amp : np.ndarray
            Lock-in amplitude R(t_d) = sqrt(X^2 + Y^2).
        """
        n_pulses = self.pulses_per_mod_cycle
        tau_rep = self.tau_rep
        t_mod = self.T_mod
        tau_th = self.tau_thermal
        m_base = self.M_base

        # Calculate pump modulation factor for each pulse k in [0, n_pulses - 1]
        pulse_indices = np.arange(n_pulses)
        t_pulse = pulse_indices * tau_rep

        if modulation_type == "square":
            # 50% duty cycle square wave: ON for first half, OFF for second half
            mod_weights = np.where(pulse_indices < n_pulses // 2, 1.0, 0.0)
        elif modulation_type == "sine":
            # Raised cosine envelope between 0 and 1
            mod_weights = 0.5 * (1.0 + np.cos(2.0 * np.pi * self.f_mod * t_pulse))
        else:
            raise ValueError(f"Unknown modulation_type: {modulation_type}")

        # Multi-pulse inter-pulse accumulation factor:
        # Since tau_th ~ 600 ps and tau_rep = 12.5 ns, decay per pulse is exp(-12.5/0.6) ~ 1e-9.
        # But for arbitrary parameters, we compute the exact periodic accumulation.
        decay_per_pulse = np.exp(-tau_rep / tau_th)

        # Precompute the accumulated background temperature rise before each pulse k:
        # 1. High-frequency pulse-to-pulse residual from exponential decay:
        delta_T_jump = self.pulse_temperature_jump(incident_fluence)
        decay_cycle = np.exp(-t_mod / tau_th)
        cycle_denom = max(1e-15, 1.0 - decay_cycle)

        T_accum = np.zeros(n_pulses)
        for k in range(n_pulses):
            acc = 0.0
            for j in range(1, n_pulses + 1):
                prev_idx = (k - j) % n_pulses
                acc += mod_weights[prev_idx] * delta_T_jump * (decay_per_pulse ** j)
            T_accum[k] = acc / cycle_denom

        # 2. Modulation-frequency baseline temperature oscillation:
        # In real TDTR / TR-MOKE, periodic pump modulation deposits average power
        # that produces a baseline temperature wave at f_mod with thermal phase lag psi:
        psi = np.radians(self.thermal_phase_lag)
        T_mod_baseline = self.accum_ratio * delta_T_jump * (1.0 + np.cos(2.0 * np.pi * self.f_mod * t_pulse - psi))

        # Now evaluate signal for each probe delay t_d:
        V_in = np.zeros_like(t_delay, dtype=float)
        V_out = np.zeros_like(t_delay, dtype=float)

        omega_ref = 2.0 * np.pi * self.f_mod * harmonic

        # For each delay t_d, evaluate M_k(t_d) for all k, then Fourier project
        for i, td in enumerate(t_delay):
            M_samples = np.zeros(n_pulses)
            for k in range(n_pulses):
                # Effective temperature at delay td after pulse k:
                # Prior pulse accumulation decays as exp(-td / tau_th) for td >= 0
                accum_at_td = T_accum[k] * np.exp(-max(0.0, td) / tau_th) if td >= 0 else T_accum[k]
                
                # Pulse k contribution:
                dT_k, _ = self.single_pulse_thermal_profile(
                    np.array([td]), incident_fluence * mod_weights[k], use_convolution=True
                )
                T_total = self.T_0 + T_mod_baseline[k] + accum_at_td + dT_k[0]
                M_samples[k] = self.magnetization(T_total)

            # Signal difference relative to base magnetization
            delta_M_k = M_samples - m_base

            # Lock-in demodulation at harmonic f_mod:
            # Discrete Fourier sum over the N pulses in the modulation cycle:
            ref_cos = np.cos(omega_ref * t_pulse)
            ref_sin = np.sin(omega_ref * t_pulse)

            # Normalization factor 2 / N for single-channel peak amplitude
            V_in[i] = (2.0 / n_pulses) * np.sum(delta_M_k * ref_cos)
            V_out[i] = (2.0 / n_pulses) * np.sum(delta_M_k * ref_sin)

        V_amp = np.sqrt(V_in**2 + V_out**2)
        return V_in, V_out, V_amp

    def calibrated_lockin_response(
        self,
        t_delay: np.ndarray,
        incident_fluence: float,
        calib_fluence: float = 1.0,
        modulation_type: Literal["square", "sine"] = "square",
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calculates the lock-in in-phase V_in and out-of-phase V_out signals using the experimental
        calibration protocol where the reference phase phi_0 is set at low fluence (linear regime)
        to make V_out smooth through zero delay (i.e. zero step at t_d = 0).

        Parameters:
        -----------
        t_delay : np.ndarray
            Probe delay times in seconds.
        incident_fluence : float
            Pump pulse energy per unit area in J/m^2.
        calib_fluence : float
            Low reference fluence used to determine reference phase (default 1.0 J/m^2).
        modulation_type : 'square' or 'sine'

        Returns:
        --------
        V_in : np.ndarray
            Calibrated in-phase signal.
        V_out : np.ndarray
            Calibrated out-of-phase signal.
        phi_0 : float
            Calibrated lock-in phase angle in radians.
        """
        # Determine calibration phase from linear reference
        idx_neg = np.argmin(np.abs(t_delay - (-10e-12)))
        idx_pos = np.argmin(np.abs(t_delay - (+20e-12)))

        X_ref, Y_ref, _ = self.periodic_lockin_response(t_delay, calib_fluence, modulation_type=modulation_type)
        dX = X_ref[idx_pos] - X_ref[idx_neg]
        dY = Y_ref[idx_pos] - Y_ref[idx_neg]
        phi_0 = float(np.arctan2(dY, dX))

        # Evaluate at target fluence and rotate by calibrated phase phi_0
        X, Y, _ = self.periodic_lockin_response(t_delay, incident_fluence, modulation_type=modulation_type)
        V_in = X * np.cos(phi_0) + Y * np.sin(phi_0)
        V_out = -X * np.sin(phi_0) + Y * np.cos(phi_0)

        return V_in, V_out, phi_0

    def spatial_quadrature_nodes_and_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the Gauss-Legendre quadrature nodes x_n and weights w_n for
        the Gaussian spatial average over normalized beam profile x in [0, 1].
        For concentric Gaussian beams with 1/e^2 radii w_pump and w_probe:
            r = w_probe * sqrt(-ln(x) / 2)
            F(r) = F_peak * x^( (w_probe / w_pump)^2 )
            Weight = dx
        """
        # Gauss-Legendre nodes and weights mapped to [0, 1]
        nodes, weights = np.polynomial.legendre.leggauss(self.n_spatial_nodes)
        x_nodes = 0.5 * (nodes + 1.0)
        x_weights = 0.5 * weights
        return x_nodes, x_weights

    def spatially_averaged_single_pulse_thermal(
        self,
        t_delay: np.ndarray,
        incident_peak_fluence: float,
        use_convolution: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates probe-weighted spatial average of temperature rise <Delta_T(t_d)>
        and absolute temperature <T(t_d)>:
            <T(t_d)> = int_0^infty T(r, t_d) * I_probe(r) * 2*pi*r dr
        """
        x_nodes, x_weights = self.spatial_quadrature_nodes_and_weights()
        gamma = (self.w_probe / self.w_pump) ** 2

        avg_delta_T = np.zeros_like(t_delay, dtype=float)
        for x, w in zip(x_nodes, x_weights):
            f_local = incident_peak_fluence * (x ** gamma)
            dT_local, _ = self.single_pulse_thermal_profile(t_delay, f_local, use_convolution)
            avg_delta_T += w * dT_local

        avg_T = self.T_0 + avg_delta_T
        return avg_delta_T, avg_T

    def spatially_averaged_moke_response(
        self,
        t_delay: np.ndarray,
        incident_peak_fluence: float,
        use_convolution: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates probe-weighted spatial average of the TR-MOKE response:
            <M(t_d)> = int_0^infty M(T(r, t_d)) * I_probe(r) * 2*pi*r dr
            <Delta_M(t_d)> = <M(t_d)> - M(T_0)
        """
        x_nodes, x_weights = self.spatial_quadrature_nodes_and_weights()
        gamma = (self.w_probe / self.w_pump) ** 2
        m_base = self.M_base

        avg_M = np.zeros_like(t_delay, dtype=float)
        for x, w in zip(x_nodes, x_weights):
            f_local = incident_peak_fluence * (x ** gamma)
            M_local, _, _ = self.single_pulse_moke_response(t_delay, f_local, use_convolution)
            avg_M += w * M_local

        avg_delta_M = avg_M - m_base
        avg_frac_delta_M = avg_delta_M / (m_base if m_base != 0 else 1.0)
        return avg_M, avg_delta_M, avg_frac_delta_M

    def spatially_averaged_periodic_lockin(
        self,
        t_delay: np.ndarray,
        incident_peak_fluence: float,
        modulation_type: Literal["square", "sine"] = "square",
        harmonic: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Probe-weighted spatial average of the raw periodic lock-in response <X(t_d)> and <Y(t_d)>.
        """
        x_nodes, x_weights = self.spatial_quadrature_nodes_and_weights()
        gamma = (self.w_probe / self.w_pump) ** 2

        avg_Vin = np.zeros_like(t_delay, dtype=float)
        avg_Vout = np.zeros_like(t_delay, dtype=float)

        for x, w in zip(x_nodes, x_weights):
            f_local = incident_peak_fluence * (x ** gamma)
            vin, vout, _ = self.periodic_lockin_response(
                t_delay, f_local, modulation_type=modulation_type, harmonic=harmonic
            )
            avg_Vin += w * vin
            avg_Vout += w * vout

        avg_Vamp = np.sqrt(avg_Vin**2 + avg_Vout**2)
        return avg_Vin, avg_Vout, avg_Vamp

    def spatially_averaged_calibrated_lockin(
        self,
        t_delay: np.ndarray,
        incident_peak_fluence: float,
        calib_fluence: float = 1.0,
        modulation_type: Literal["square", "sine"] = "square",
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Probe-weighted spatial average of lock-in signals V_in and V_out under experimental
        calibration, where reference phase phi_0 is calibrated at low fluence (calib_fluence)
        to make spatially-averaged V_out smooth across delay zero.
        """
        idx_neg = np.argmin(np.abs(t_delay - (-10e-12)))
        idx_pos = np.argmin(np.abs(t_delay - (+20e-12)))

        # Calibrate phase at low fluence using spatially-averaged signal
        X_ref, Y_ref, _ = self.spatially_averaged_periodic_lockin(
            t_delay, calib_fluence, modulation_type=modulation_type
        )
        dX = X_ref[idx_pos] - X_ref[idx_neg]
        dY = Y_ref[idx_pos] - Y_ref[idx_neg]
        phi_0 = float(np.arctan2(dY, dX))

        # Rotate spatially-averaged signal at target fluence
        X, Y, _ = self.spatially_averaged_periodic_lockin(
            t_delay, incident_peak_fluence, modulation_type=modulation_type
        )
        V_in = X * np.cos(phi_0) + Y * np.sin(phi_0)
        V_out = -X * np.sin(phi_0) + Y * np.cos(phi_0)

        return V_in, V_out, phi_0

    def summary_dict(self) -> dict:
        """Return a dictionary of all key physical parameters and derived quantities."""
        return {
            "Film thickness d": f"{self.d_film * 1e9:.1f} nm",
            "Interface conductance G": f"{self.G_interface / 1e6:.1f} MW/(m^2*K)",
            "Volumetric heat capacity C_v": f"{self.C_v / 1e6:.2f} J/(cm^3*K)",
            "Areal heat capacity C_film": f"{self.C_film * 1e3:.2f} mJ/(m^2*K)",
            "Thermal decay time tau_th": f"{self.tau_thermal * 1e12:.1f} ps ({self.tau_thermal * 1e9:.3f} ns)",
            "Base temperature T_0": f"{self.T_0:.1f} K",
            "Curie temperature T_c": f"{self.T_c:.1f} K",
            "Delta T to reach Curie (T_c - T_0)": f"{self.delta_T_curie:.1f} K",
            "Magnetization exponent": f"{self.mag_exponent:.2f}",
            "Magnetization model": self.mag_model,
            "Pump absorptance": f"{self.absorptance * 100:.1f} %",
            "Critical absorbed fluence": f"{self.critical_absorbed_fluence:.3f} J/m^2 ({self.critical_absorbed_fluence * 0.1:.4f} mJ/cm^2)",
            "Critical incident fluence": f"{self.critical_incident_fluence:.3f} J/m^2 ({self.critical_incident_fluence * 0.1:.4f} mJ/cm^2)",
            "Laser repetition rate f_rep": f"{self.f_rep / 1e6:.1f} MHz (pulse interval {self.tau_rep * 1e9:.1f} ns)",
            "Pump modulation frequency f_mod": f"{self.f_mod / 1e6:.1f} MHz (period {self.T_mod * 1e9:.1f} ns)",
            "Pulses per modulation cycle": self.pulses_per_mod_cycle,
            "Pump 1/e^2 radius w_pump": f"{self.w_pump * 1e6:.1f} um",
            "Probe 1/e^2 radius w_probe": f"{self.w_probe * 1e6:.1f} um",
        }
