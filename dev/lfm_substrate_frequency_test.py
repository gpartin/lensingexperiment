#!/usr/bin/env python3
"""
================================================================================
LFM Substrate Lensing: Frequency Dependence Test (TRUE DYNAMICS)
================================================================================

Key question: Does the lensing effect depend on radiation frequency?
The Reddit critique was about chromatic (frequency-dependent) lensing.

We test by creating atoms of different sizes, which should radiate at
different natural frequencies, and measure if lensing ratio differs.

All using TRUE substrate dynamics (GOV-01 + GOV-02 coupled).
No static χ. No artificial test waves.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# PARAMETERS
# =============================================================================

# Grid
NX, NY = 400, 400
L = 200.0
dx = L / NX
x = np.linspace(-L/2, L/2, NX)
y = np.linspace(-L/2, L/2, NY)
X, Y = np.meshgrid(x, y, indexing='ij')

# Physics
c = 1.0
chi_0 = 1.0
kappa = 2.0
E0_sq = 0.0

# Time stepping
dt = 0.3 * dx / c
n_steps = 3000
settle_steps = 400

# Star (same for all tests)
star_x, star_y = 0.0, 0.0
star_amp = 4.0
star_radius = 8.0

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

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

def run_lensing_test(atom_radius, electron_orbit_r, label):
    """
    Run lensing test with given atom size.
    Smaller atoms with tighter electron orbits → higher frequency.
    """
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"  Atom radius: {atom_radius}, Electron orbit: {electron_orbit_r}")
    print(f"{'='*60}")
    
    # Atom position (off-axis)
    atom_x, atom_y = -60.0, 40.0
    
    # Create star and atom
    E_star = gaussian(X, Y, star_x, star_y, star_amp, star_radius)
    E_nucleus = gaussian(X, Y, atom_x, atom_y, 2.0, atom_radius)
    
    electron_x = atom_x + electron_orbit_r
    electron_y = atom_y
    E_electron = gaussian(X, Y, electron_x, electron_y, 0.8, atom_radius)
    
    E = E_star + E_nucleus + E_electron
    E_prev = E.copy()
    
    chi = chi_0 * np.ones_like(E)
    chi_prev = chi.copy()
    
    # Settle
    print("  Settling χ field...")
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
    
    # Excite electron
    kick_region = gaussian(X, Y, electron_x, electron_y, 1.0, atom_radius) > 0.5
    kick_strength = 0.15
    E_prev[kick_region] = E[kick_region] - kick_strength * dt * E[kick_region]
    
    # Detection masks
    detect_r = 30.0
    toward_mask = ring_mask(X, Y, atom_x, atom_y, detect_r, width=3.0) * (X > atom_x)
    away_mask = ring_mask(X, Y, atom_x, atom_y, detect_r, width=3.0) * (X < atom_x)
    
    # Evolve and track
    radiation_toward = []
    radiation_away = []
    
    # Track oscillation frequency by monitoring E at a point near atom
    probe_x_idx = int((atom_x + 20 + L/2) / dx)
    probe_y_idx = int((atom_y + L/2) / dx)
    E_probe = []
    
    evolution_steps = n_steps - settle_steps
    for step in range(evolution_steps):
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
        
        E_sq = E**2
        radiation_toward.append(np.sum(E_sq * toward_mask))
        radiation_away.append(np.sum(E_sq * away_mask))
        E_probe.append(E[probe_x_idx, probe_y_idx])
    
    # Analyze
    total_toward = np.sum(radiation_toward)
    total_away = np.sum(radiation_away)
    ratio = total_toward / (total_away + 1e-10)
    
    # Estimate frequency from probe signal via FFT
    E_probe = np.array(E_probe)
    if len(E_probe) > 10:
        # FFT to find dominant frequency
        fft = np.fft.rfft(E_probe - np.mean(E_probe))
        freqs = np.fft.rfftfreq(len(E_probe), dt)
        peak_idx = np.argmax(np.abs(fft[1:])) + 1  # Skip DC
        freq = freqs[peak_idx] if peak_idx < len(freqs) else 0
    else:
        freq = 0
    
    print(f"  Radiation toward: {total_toward:.2e}")
    print(f"  Radiation away: {total_away:.2e}")
    print(f"  Lensing ratio: {ratio:.2f}")
    print(f"  Dominant frequency: {freq:.4f}")
    
    return {
        'label': label,
        'atom_radius': atom_radius,
        'electron_orbit': electron_orbit_r,
        'total_toward': total_toward,
        'total_away': total_away,
        'ratio': ratio,
        'frequency': freq
    }

# =============================================================================
# RUN TESTS
# =============================================================================

print("=" * 70)
print("LFM SUBSTRATE LENSING: Frequency Dependence Test (TRUE DYNAMICS)")
print("=" * 70)
print()
print("Using ONLY coupled GOV-01 + GOV-02. No static χ. No test waves.")
print()
print("Testing atoms of different sizes (different natural frequencies)")
print("All use same star, same geometry, just different atom structure")
print()

# Different atom configurations
# Smaller atoms with tighter electron orbits → higher frequency
configs = [
    (5.0, 15.0, "Large atom (low freq)"),
    (3.0, 10.0, "Medium atom (mid freq)"),
    (2.0, 6.0, "Small atom (high freq)"),
]

results = []
for radius, orbit, label in configs:
    result = run_lensing_test(radius, orbit, label)
    results.append(result)

# =============================================================================
# SUMMARY
# =============================================================================

print()
print("=" * 70)
print("FREQUENCY DEPENDENCE SUMMARY")
print("=" * 70)
print()
print(f"{'Atom Type':<25} {'Frequency':<12} {'Lensing Ratio':<15}")
print("-" * 52)

frequencies = []
ratios = []
for r in results:
    print(f"{r['label']:<25} {r['frequency']:<12.4f} {r['ratio']:<15.2f}")
    frequencies.append(r['frequency'])
    ratios.append(r['ratio'])

print()

# Calculate variation
if len(ratios) > 1:
    mean_ratio = np.mean(ratios)
    std_ratio = np.std(ratios)
    variation = std_ratio / mean_ratio * 100 if mean_ratio > 0 else 0
    
    print(f"Mean lensing ratio: {mean_ratio:.2f}")
    print(f"Standard deviation: {std_ratio:.2f}")
    print(f"Coefficient of variation: {variation:.1f}%")
    print()
    
    if variation < 30:
        print("✓ LENSING IS APPROXIMATELY ACHROMATIC!")
        print("  Different frequencies show similar lensing behavior.")
        print("  This addresses the chromatic lensing concern from Klein-Gordon.")
    elif variation < 60:
        print("! Moderate frequency dependence detected.")
        print("  Need to check if this matches observed constraints.")
    else:
        print("✗ Strong frequency dependence detected.")
        print("  This would be a problem for the theory.")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('LFM Substrate Lensing: Frequency Dependence Test\n'
             '(True GOV-01 + GOV-02 coupled dynamics)', 
             fontsize=13, fontweight='bold')

ax = axes[0]
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results)))
bars = ax.bar([r['label'] for r in results], ratios, color=colors, alpha=0.8, edgecolor='black')
ax.set_ylabel('Lensing Ratio (toward/away)')
ax.set_title('Lensing Strength by Atom Type')
ax.tick_params(axis='x', rotation=15)
ax.axhline(mean_ratio, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_ratio:.1f}')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
sc = ax.scatter(frequencies, ratios, s=200, c=colors, edgecolors='black', linewidth=2)
for r, color in zip(results, colors):
    ax.annotate(r['label'].split()[0], (r['frequency'], r['ratio']), 
                textcoords="offset points", xytext=(8, 5), fontsize=10)
ax.set_xlabel('Radiation Frequency')
ax.set_ylabel('Lensing Ratio (toward/away)')
ax.set_title('Lensing vs Frequency')
ax.grid(True, alpha=0.3)

# Add trend line
if len(frequencies) > 1 and max(frequencies) > min(frequencies):
    z = np.polyfit(frequencies, ratios, 1)
    p = np.poly1d(z)
    freq_range = np.linspace(min(frequencies)*0.8, max(frequencies)*1.2, 50)
    ax.plot(freq_range, p(freq_range), 'r--', alpha=0.6, linewidth=2, label='Linear trend')
    
    # Calculate slope significance
    slope = z[0]
    print()
    print(f"Linear fit slope: {slope:.2f} (lensing ratio per unit frequency)")
    ax.legend()

plt.tight_layout()

output_dir = Path(__file__).parent / 'figures'
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'lfm_substrate_frequency_dependence.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print()
print(f"Figure saved to: {output_path}")
print("Done!")
