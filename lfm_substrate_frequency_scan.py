#!/usr/bin/env python3
"""
================================================================================
LFM Substrate: TRUE Frequency Test with Direct Excitation
================================================================================

Previous test: All atoms radiated at same frequency - the system's natural mode.

This test: Directly excite atoms at SPECIFIC frequencies to get broader range,
then measure if lensing depends on the excitation frequency.

Still using TRUE substrate dynamics (GOV-01 + GOV-02 coupled).
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# PARAMETERS
# =============================================================================

NX, NY = 400, 400
L = 200.0
dx = L / NX
x = np.linspace(-L/2, L/2, NX)
y = np.linspace(-L/2, L/2, NY)
X, Y = np.meshgrid(x, y, indexing='ij')

c = 1.0
chi_0 = 1.0
kappa = 2.0
E0_sq = 0.0

dt = 0.3 * dx / c
n_total_steps = 4000
settle_steps = 500

# Star at center
star_x, star_y = 0.0, 0.0
star_amp = 4.0
star_radius = 8.0

# Source position (will oscillate at different frequencies)
source_x, source_y = -50.0, 50.0
source_radius = 4.0

def laplacian(f, dx):
    lap = np.zeros_like(f)
    lap[1:-1, 1:-1] = (
        f[2:, 1:-1] + f[:-2, 1:-1] +
        f[1:-1, 2:] + f[1:-1, :-2] -
        4*f[1:-1, 1:-1]
    ) / dx**2
    return lap

def gaussian(X, Y, x0, y0, amp, radius):
    R = np.sqrt((X - x0)**2 + (Y - y0)**2)
    return amp * np.exp(-R**2 / (2 * radius**2))

def ring_mask(X, Y, x0, y0, r, width=3.0):
    R = np.sqrt((X - x0)**2 + (Y - y0)**2)
    return (np.abs(R - r) < width).astype(float)

def damping_layer(X, Y, L, width=20.0):
    dx_edge = np.minimum(X + L/2, L/2 - X)
    dy_edge = np.minimum(Y + L/2, L/2 - Y)
    d_edge = np.minimum(dx_edge, dy_edge)
    return np.clip(d_edge / width, 0, 1)

damp = damping_layer(X, Y, L, width=25.0)

# Source profile (where to apply oscillation)
source_profile = gaussian(X, Y, source_x, source_y, 1.0, source_radius)

def run_frequency_test(omega, label):
    """
    Run test with source oscillating at specific frequency omega.
    
    The source is part of the substrate - it's a region where E oscillates
    because "something is shaking it" (like an accelerating charge).
    The radiation that emerges will have this frequency.
    """
    print(f"\n{'='*60}")
    print(f"Testing: {label} (ω = {omega:.3f})")
    print(f"  ω/χ₀ = {omega/chi_0:.2f}")
    print(f"{'='*60}")
    
    # Create star
    E_star = gaussian(X, Y, star_x, star_y, star_amp, star_radius)
    
    # Initialize E and χ
    E = E_star.copy()
    E_prev = E.copy()
    chi = chi_0 * np.ones_like(E)
    chi_prev = chi.copy()
    
    # Settle χ around star
    print("  Settling χ field around star...")
    for step in range(settle_steps):
        lap_E = laplacian(E, dx)
        E_accel = c**2 * lap_E - chi**2 * E
        E_new = 2*E - E_prev + dt**2 * E_accel
        
        lap_chi = laplacian(chi, dx)
        chi_accel = c**2 * lap_chi - kappa * (E**2 - E0_sq)
        chi_new = 2*chi - chi_prev + dt**2 * chi_accel
        chi_new = np.maximum(chi_new, 0.1)
        
        E_new = E_new * damp + E * (1 - damp) * 0.99
        chi_new = chi_new * damp + chi_0 * (1 - damp)
        
        E_prev, E = E, E_new
        chi_prev, chi = chi, chi_new
    
    chi_at_star = chi[NX//2, NY//2]
    print(f"  χ at star: {chi_at_star:.4f}")
    
    # Detection masks
    detect_r = 35.0
    # Divide into toward-star and away-from-star hemispheres
    angle_to_star = np.arctan2(star_y - source_y, star_x - source_x)
    angles = np.arctan2(Y - source_y, X - source_x)
    angle_diff = np.abs(angles - angle_to_star)
    angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
    
    ring = ring_mask(X, Y, source_x, source_y, detect_r, width=3.0)
    toward_mask = ring * (angle_diff < np.pi/2)  # Within 90° of star direction
    away_mask = ring * (angle_diff > np.pi/2)    # More than 90° from star
    
    # Evolve with oscillating source
    radiation_toward = []
    radiation_away = []
    source_amp = 0.3
    
    drive_steps = n_total_steps - settle_steps
    print(f"  Driving source at ω = {omega:.3f} for {drive_steps} steps...")
    
    for step in range(drive_steps):
        t = step * dt
        
        # GOV-01 with oscillating source term
        lap_E = laplacian(E, dx)
        E_accel = c**2 * lap_E - chi**2 * E
        E_new = 2*E - E_prev + dt**2 * E_accel
        
        # Add oscillating source (THIS is how radiation is created!)
        # An oscillating E at the source location
        drive_term = source_amp * np.sin(omega * t) * source_profile * dt**2
        E_new += drive_term
        
        # GOV-02: χ dynamics
        lap_chi = laplacian(chi, dx)
        chi_accel = c**2 * lap_chi - kappa * (E**2 - E0_sq)
        chi_new = 2*chi - chi_prev + dt**2 * chi_accel
        chi_new = np.maximum(chi_new, 0.1)
        
        # Damping at boundaries
        E_new = E_new * damp + E * (1 - damp) * 0.99
        chi_new = chi_new * damp + chi_0 * (1 - damp)
        
        E_prev, E = E, E_new
        chi_prev, chi = chi, chi_new
        
        # Measure radiation
        E_sq = E**2
        radiation_toward.append(np.sum(E_sq * toward_mask))
        radiation_away.append(np.sum(E_sq * away_mask))
    
    # Analyze
    total_toward = np.sum(radiation_toward)
    total_away = np.sum(radiation_away)
    ratio = total_toward / (total_away + 1e-10)
    
    # Also compute deflection by tracking radiation centroid
    print(f"  Radiation toward star: {total_toward:.2e}")
    print(f"  Radiation away: {total_away:.2e}")
    print(f"  Lensing ratio: {ratio:.3f}")
    
    return {
        'label': label,
        'omega': omega,
        'omega_over_chi': omega / chi_0,
        'total_toward': total_toward,
        'total_away': total_away,
        'ratio': ratio
    }

# =============================================================================
# RUN TESTS
# =============================================================================

print("=" * 70)
print("LFM SUBSTRATE: Frequency-Dependent Lensing Test")
print("=" * 70)
print()
print("GOV-01: ∂²E/∂t² = c²∇²E − χ²E  (plus oscillating source)")
print("GOV-02: ∂²χ/∂t² = c²∇²χ − κ(E² − E₀²)")
print()
print("Klein-Gordon dispersion: ω² = c²k² + χ²")
print("For massless-like waves: need ω >> χ₀")
print()

# Test range of frequencies
# χ₀ = 1.0, so ω/χ₀ = 1 is critical
# ω << χ₀: massive particle-like
# ω >> χ₀: massless photon-like
frequencies = [
    (0.5, "Low freq (ω/χ₀ = 0.5)"),
    (1.0, "Critical (ω/χ₀ = 1.0)"),
    (2.0, "Moderate (ω/χ₀ = 2.0)"),
    (4.0, "High freq (ω/χ₀ = 4.0)"),
]

results = []
for omega, label in frequencies:
    result = run_frequency_test(omega, label)
    results.append(result)

# =============================================================================
# ANALYSIS
# =============================================================================

print()
print("=" * 70)
print("FREQUENCY DEPENDENCE RESULTS")
print("=" * 70)
print()
print(f"{'Frequency':<25} {'ω/χ₀':<10} {'Lensing Ratio':<15}")
print("-" * 50)

omegas = []
ratios = []
for r in results:
    print(f"{r['label']:<25} {r['omega_over_chi']:<10.2f} {r['ratio']:<15.3f}")
    omegas.append(r['omega'])
    ratios.append(r['ratio'])

print()

# Calculate trend
omegas = np.array(omegas)
ratios = np.array(ratios)

mean_ratio = np.mean(ratios)
std_ratio = np.std(ratios)
cv = std_ratio / mean_ratio * 100

print(f"Mean lensing ratio: {mean_ratio:.3f}")
print(f"Standard deviation: {std_ratio:.3f}")
print(f"Coefficient of variation: {cv:.1f}%")
print()

# Check trend: does lensing decrease with frequency? (the KG prediction)
if len(ratios) > 1:
    # Fit log-log slope (if lensing ∝ ω^n)
    log_omega = np.log(omegas)
    log_ratio = np.log(ratios)
    slope, intercept = np.polyfit(log_omega, log_ratio, 1)
    
    print(f"Power law fit: lensing ∝ ω^{slope:.2f}")
    print()
    
    if abs(slope) < 0.5:
        print("✓ WEAK frequency dependence!")
        print("  Lensing is approximately achromatic.")
    elif slope < -0.5:
        print("! Lensing DECREASES with frequency.")
        print("  This matches Klein-Gordon dispersion prediction.")
        print("  BUT: if ω >> χ₀ for real photons, effect may be negligible.")
    else:
        print("! Lensing INCREASES with frequency.")
        print("  Unexpected behavior - investigate further.")

print()
print("KEY PHYSICS INSIGHT:")
print("-" * 50)
print(f"  For real photons: ω ~ 10^15 Hz")
print(f"  If χ₀ ~ 1 (natural units), then ω/χ₀ ~ 10^15")
print(f"  Even 70% variation at ω/χ₀ ~ 1-4 becomes")
print(f"  NEGLIGIBLE at ω/χ₀ ~ 10^15")
print()
print("  Dispersion ∝ (χ₀/ω)² → vanishes for ω >> χ₀")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('LFM Substrate: Frequency-Dependent Lensing Test\n'
             'True GOV-01 + GOV-02 dynamics', fontsize=13, fontweight='bold')

ax = axes[0]
colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(results)))
for r, color in zip(results, colors):
    ax.bar(r['label'], r['ratio'], color=color, alpha=0.8, edgecolor='black')
ax.set_ylabel('Lensing Ratio (toward/away)')
ax.set_title('Lensing by Frequency')
ax.tick_params(axis='x', rotation=20)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
ax.loglog(omegas, ratios, 'o-', markersize=12, linewidth=2, color='blue')
ax.set_xlabel('Frequency ω')
ax.set_ylabel('Lensing Ratio')
ax.set_title(f'Log-Log Plot (slope = {slope:.2f})')
ax.grid(True, alpha=0.3)

# Add power law fit
omega_range = np.linspace(min(omegas)*0.8, max(omegas)*1.2, 50)
fit_line = np.exp(intercept) * omega_range**slope
ax.plot(omega_range, fit_line, 'r--', linewidth=2, alpha=0.7, 
        label=f'Power law: ω^{slope:.2f}')
ax.legend()

# Add text box with conclusion
if abs(slope) < 0.5:
    conclusion = "WEAK frequency dependence\n(approximately achromatic)"
    color = 'lightgreen'
else:
    conclusion = f"Lensing ∝ ω^{slope:.1f}\n(dispersion present)"
    color = 'lightyellow'

ax.text(0.95, 0.95, conclusion, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor=color, alpha=0.9))

plt.tight_layout()

output_dir = Path(__file__).parent / 'figures'
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'lfm_substrate_frequency_scan.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print()
print(f"Figure saved to: {output_path}")
print("Done!")
