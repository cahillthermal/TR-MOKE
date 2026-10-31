# Time-Resolved Magneto-Optic Kerr Effect (TR-MOKE) Non-Linear Model

A physical simulation package for modeling the non-linear thermal, magnetic, and lock-in detection dynamics of a ferromagnetic thin film transducer in Time-Resolved Magneto-Optic Kerr Effect (TR-MOKE) experiments.

---

## Table of Contents
- [Physical Background & Theory](#physical-background--theory)
  - [1. Thermal Decay Dynamics](#1-thermal-decay-dynamics)
  - [2. Magnetization Temperature Dependence](#2-magnetization-temperature-dependence)
  - [3. Curie Temperature Exceedance & Saturation](#3-curie-temperature-exceedance--saturation)
  - [4. Lock-In Detection & Out-of-Phase Signal Leakage](#4-lock-in-detection--out-of-phase-signal-leakage)
  - [5. Gaussian Spatial Averaging](#5-gaussian-spatial-averaging)
- [Default Physical Parameters](#default-physical-parameters)
- [Project Architecture](#project-architecture)
- [Installation & Requirements](#installation--requirements)
- [Quickstart & Usage Guide](#quickstart--usage-guide)
  - [Basic Simulation Example](#basic-simulation-example)
  - [Switching Magnetization Models](#switching-magnetization-models)
  - [Running Lock-In Demodulation](#running-lock-in-demodulation)
  - [Running the Complete Analysis Script](#running-the-complete-analysis-script)
  - [Running Unit Tests](#running-unit-tests)
- [Generated Visualizations](#generated-visualizations)

---

## Physical Background & Theory

### 1. Thermal Decay Dynamics
The sample consists of a ferromagnetic metal thin film of thickness $d$ deposited on a substrate with extremely high thermal conductivity (infinite heat sink maintained at base temperature $T_0 = 300\text{ K}$).

Cooling of the film is dominated by heat transport across the interface thermal conductance per unit area $G$. The thermal response of the metal film is governed by a simple exponential decay:
$$\frac{d(\Delta T)}{dt} = -\frac{G}{C_{\mathrm{film}}} \Delta T = -\frac{1}{\tau_{\mathrm{th}}} \Delta T$$

Where:
- $C_{\mathrm{film}} = C_v \cdot d$ is the areal heat capacity of the metal film ($60\text{ mJ}/(\text{m}^2\cdot\text{K})$).
- $\tau_{\mathrm{th}} = \frac{C_{\mathrm{film}}}{G} = \frac{C_v d}{G} = 600\text{ ps}$ is the characteristic thermal relaxation time constant.

When a pump pulse with incident fluence $F_0$ strikes the film, the instantaneous temperature jump is:
$$\Delta T_{\mathrm{jump}} = \frac{A \cdot F_0}{C_{\mathrm{film}}}$$
where $A = 0.30$ is the optical absorptance.

### 2. Magnetization Temperature Dependence
The Kerr rotation $\theta_K(t_d)$ measured by a weak probe pulse is directly proportional to the spontaneous magnetization $M(T(t_d))$. Two functional forms are provided in [`TRMOKEModel.magnetization()`](tr_moke_model.py:141):

1. **Bloch Power Law (Default, `mag_model = "bloch"`)**:
   $$M(T) = \begin{cases} M_0 \left[ 1 - \left(\dfrac{T}{T_c}\right)^\alpha \right], & 0 \le T < T_c \\[1em] 0, & T \ge T_c \end{cases}$$
   With default $\alpha = 1.5$ ($3/2$), derived from spin-wave theory.
2. **Critical Power Law (`mag_model = "critical"`)**:
   $$M(T) = \begin{cases} M_0 \left( 1 - \dfrac{T}{T_c} \right)^\beta, & 0 \le T < T_c \\[1em] 0, & T \ge T_c \end{cases}$$
   With exponent $\beta \approx 0.36$ (3D Heisenberg) or $\beta = 0.5$ (mean-field).

### 3. Curie Temperature Exceedance & Saturation
The temperature rise needed to reach the Curie temperature $T_c = 330\text{ K}$ from $T_0 = 300\text{ K}$ is $\Delta T_{\mathrm{crit}} = 30\text{ K}$.
- **Critical absorbed fluence**: $F_{\mathrm{abs, crit}} = C_{\mathrm{film}} (T_c - T_0) = 1.8\text{ J/m}^2$.
- **Critical incident fluence**: $F_{\mathrm{crit}} = \frac{1.8}{0.30} = 6.0\text{ J/m}^2 = 0.60\text{ mJ/cm}^2$.

When $F_0 > F_{\mathrm{crit}}$, the film exceeds $T_c$ and enters the paramagnetic state ($M = 0$). Local demagnetization saturates completely ($-\Delta M / M(T_0) = 1.0$) for a plateau duration:
$$t_{\mathrm{above}} = \tau_{\mathrm{th}} \ln\left(\frac{F_0}{F_{\mathrm{crit}}}\right)$$

### 4. Lock-In Detection & Out-of-Phase Signal Leakage
In TR-MOKE, the pump pulse train ($f_{\mathrm{rep}} = 80\text{ MHz}$, pulse interval $12.5\text{ ns}$) is modulated at $f_{\mathrm{mod}} = 10\text{ MHz}$ ($8$ pulses per modulation cycle).

- **Linear Regime Calibration**: At low fluence, the experimentalist sets the lock-in reference phase $\phi_0$ so that the delay-dependent step $\Delta V$ at $t_d = 0$ is aligned strictly into the in-phase channel $V_{\mathrm{in}}$. In this calibration, the out-of-phase signal $V_{\mathrm{out}}(t_d)$ has zero step at $t_d = 0$ and is completely smooth across delay zero.
- **Non-Linear Regime ($T > T_c$)**: As pump fluence increases, non-linear demagnetization mixes with the phase-lagged modulation-frequency background heating. Pulses later in the heating cycle start from a higher baseline temperature and experience longer saturation than pulses early in the cycle. This creates an RF phase shift $\Delta \phi(F)$. In the detector, this phase shift rotates the signal:
  $$V_{\mathrm{in}}(t_d) \propto \Delta V(t_d) \cos(\Delta \phi), \quad V_{\mathrm{out}}(t_d) \propto \Delta V(t_d) \sin(\Delta \phi)$$
  As a result, $V_{\mathrm{out}}(t_d)$ develops a **sharp step at $t_d = 0$ and exponential relaxation** that mirrors $V_{\mathrm{in}}(t_d)$.

### 5. Gaussian Spatial Averaging
Both the pump and probe beams have Gaussian spatial profiles with $1/e^2$ intensity radius $w_0 = w_1 = 10\ \mu\mathrm{m}$:
$$F(r) = F_{\mathrm{peak}} \exp\left(-\frac{2 r^2}{w_0^2}\right), \quad I_{\mathrm{probe}}(r) = \frac{2}{\pi w_1^2} \exp\left(-\frac{2 r^2}{w_1^2}\right)$$

The probe measures the **weighted spatial average of local magnetization**:
$$\langle M(t_d) \rangle = \int_0^\infty M(T(r, t_d)) \cdot I_{\mathrm{probe}}(r) \cdot 2\pi r \, dr$$

Because $M(T)$ is strongly non-linear, evaluating $\langle M(T(r)) \rangle$ is physically required and produces a gradual, rounded saturation curve as the demagnetized spot radius $r_c = w \sqrt{0.5 \ln(F_{\mathrm{peak}} / F_{\mathrm{crit}})}$ expands across the probe beam.

---

## Default Physical Parameters

| Parameter | Symbol | Default Value | SI Units | Reference in Code |
|---|---|---|---|---|
| Film Thickness | $d$ | 20.0 | $\text{nm}$ ($20 \times 10^{-9}\text{ m}$) | [`TRMOKEModel.d_film`](tr_moke_model.py:67) |
| Interface Conductance | $G$ | 100.0 | $\text{MW}/(\text{m}^2\cdot\text{K})$ ($10^8\text{ W}/(\text{m}^2\cdot\text{K})$) | [`TRMOKEModel.G_interface`](tr_moke_model.py:68) |
| Volumetric Heat Capacity | $C_v$ | 3.0 | $\text{J}/(\text{cm}^3\cdot\text{K})$ ($3 \times 10^6\text{ J}/(\text{m}^3\cdot\text{K})$) | [`TRMOKEModel.C_v`](tr_moke_model.py:69) |
| Areal Heat Capacity | $C_{\mathrm{film}}$ | 60.0 | $\text{mJ}/(\text{m}^2\cdot\text{K})$ ($0.06\text{ J}/(\text{m}^2\cdot\text{K})$) | [`TRMOKEModel.C_film`](tr_moke_model.py:86) |
| Thermal Decay Time | $\tau_{\mathrm{th}}$ | 600.0 | $\text{ps}$ ($0.6 \times 10^{-9}\text{ s}$) | [`TRMOKEModel.tau_thermal`](tr_moke_model.py:90) |
| Base Temperature | $T_0$ | 300.0 | $\text{K}$ | [`TRMOKEModel.T_0`](tr_moke_model.py:71) |
| Curie Temperature | $T_c$ | 330.0 | $\text{K}$ | [`TRMOKEModel.T_c`](tr_moke_model.py:70) |
| Magnetization Model | `mag_model` | `'bloch'` | Bloch power law (or `'critical'`) | [`TRMOKEModel.mag_model`](tr_moke_model.py:77) |
| Temperature Exponent | $\alpha$ | 1.5 | $3/2$ (or $\beta = 0.36$ in critical model) | [`TRMOKEModel.mag_exponent`](tr_moke_model.py:72) |
| Pump Absorptance | $A$ | 0.30 | 30% | [`TRMOKEModel.absorptance`](tr_moke_model.py:75) |
| Critical Incident Fluence | $F_{\mathrm{crit}}$ | 6.0 | $\text{J/m}^2$ ($0.60\text{ mJ/cm}^2$) | [`TRMOKEModel.critical_incident_fluence`](tr_moke_model.py:129) |
| Laser Repetition Rate | $f_{\mathrm{rep}}$ | 80.0 | $\text{MHz}$ ($\tau_{\mathrm{rep}} = 12.5\text{ ns}$) | [`TRMOKEModel.f_rep`](tr_moke_model.py:74) |
| Pump Modulation Frequency | $f_{\mathrm{mod}}$ | 10.0 | $\text{MHz}$ ($T_{\mathrm{mod}} = 100\text{ ns}$) | [`TRMOKEModel.f_mod`](tr_moke_model.py:73) |
| Pump Beam $1/e^2$ Radius | $w_0$ | 10.0 | $\mu\text{m}$ ($10 \times 10^{-6}\text{ m}$) | [`TRMOKEModel.w_pump`](tr_moke_model.py:81) |
| Probe Beam $1/e^2$ Radius | $w_1$ | 10.0 | $\mu\text{m}$ ($10 \times 10^{-6}\text{ m}$) | [`TRMOKEModel.w_probe`](tr_moke_model.py:82) |
| Probe Delay Range | $t_d$ | -0.5 to 4.0 | $\text{ns}$ ($-500\text{ ps}$ to $+4000\text{ ps}$) | [`run_analysis()`](run_tr_moke_analysis.py:38) |

---

## Project Architecture

```
TR-MOKE/
├── tr_moke_model.py                 # Core physical model class TRMOKEModel
├── run_tr_moke_analysis.py          # Execution script for fluence sweeps & plotting
├── test_tr_moke.py                  # Automated unit test verification suite
├── prompt.txt                       # Initial task specification
├── README.md                        # Documentation and user guide
├── tr_moke_nonlinear_comparison.png # 6-panel summary of dynamics & spatial averaging
├── tr_moke_saturation_analysis.png  # Local vs. spatially-averaged saturation curve
└── tr_moke_lockin_phase_leakage.png # In-phase and out-of-phase lock-in signals
```

---

## Installation & Requirements

The model requires standard Python scientific libraries:
```bash
python --version   # Python 3.9+ recommended
pip install numpy scipy matplotlib
```

---

## Quickstart & Usage Guide

### Basic Simulation Example

```python
import numpy as np
from tr_moke_model import TRMOKEModel

# 1. Instantiate the physical model with default parameters
model = TRMOKEModel()

# 2. Define probe delay times from -0.5 ns to +4.0 ns
t_delays = np.linspace(-0.5e-9, 4.0e-9, 500)

# 3. Compute spatially-averaged MOKE response for an incident pump fluence of 8.0 J/m^2
F_peak = 8.0  # J/m^2 (exceeds critical fluence of 6.0 J/m^2)
avg_M, avg_delta_M, avg_frac_delta_M = model.spatially_averaged_moke_response(t_delays, F_peak)

# Demagnetization fraction at peak (t_d = 50 ps)
print(f"Spatially-averaged peak demagnetization: {-np.min(avg_frac_delta_M) * 100:.2f}%")
```

### Switching Magnetization Models

```python
# Use the default Bloch power-law model (exponent 1.5):
bloch_model = TRMOKEModel(mag_model="bloch", mag_exponent=1.5)

# Or switch to the critical power-law scaling model (3D Heisenberg exponent 0.36):
crit_model = TRMOKEModel(mag_model="critical", mag_exponent=0.36)
```

### Running Lock-In Demodulation

```python
# Compute calibrated lock-in signals where V_out is zeroed at low fluence (1.0 J/m^2)
vin, vout, phi_0 = model.spatially_averaged_calibrated_lockin(
    t_delays=t_delays,
    incident_peak_fluence=12.0,  # J/m^2
    calib_fluence=1.0            # J/m^2
)

# vin has the in-phase step and decay
# vout shows the emergent out-of-phase step resulting from non-linear phase leakage
```

### Running the Complete Analysis Script

To execute the full simulation suite across fluences from $1.5\text{ J/m}^2$ to $16.0\text{ J/m}^2$ and generate all figures:
```bash
python run_tr_moke_analysis.py
```

### Running Unit Tests

To verify numerical accuracy, parameter scaling, and spatial averaging theorems:
```bash
python test_tr_moke.py
```

---

## Generated Visualizations

1. **`tr_moke_lockin_phase_leakage.png`**:
   - Shows calibrated in-phase $V_{\mathrm{in}}(t_d)$ and out-of-phase $V_{\mathrm{out}}(t_d)$ signals starting from $-0.5\text{ ns}$ delay time.
   - Highlights the flat, smooth $V_{\mathrm{out}}$ baseline at low fluence and the emergent step and recovery at high fluence.
2. **`tr_moke_saturation_analysis.png`**:
   - Compares the sharp on-axis local saturation kink against the smooth probe-averaged saturation curve.
   - Shows the expansion of the demagnetized disk radius $r_c(F_{\mathrm{peak}})$ across the $10\ \mu\mathrm{m}$ probe beam.
3. **`tr_moke_nonlinear_comparison.png`**:
   - 6-panel overview: probe-averaged temperature, magnetization dynamics, normalized demagnetization, linearity check, local vs. spatially-averaged kinetics, and ultrafast kinetics around delay zero.
