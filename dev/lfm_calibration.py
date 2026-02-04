#!/usr/bin/env python3
"""
================================================================================
LFM Calibration: Finding χ₀ from Gravitational Dynamics
================================================================================

Strategy: Match LFM gravitational dynamics to Newtonian gravity.

We know G = 6.67×10⁻¹¹ N⋅m²/kg². If LFM reproduces gravity correctly,
we can constrain the relationship between χ₀, κ, and G.

From dimensional analysis in LFM:
- χ has units of [1/time] (frequency)
- κ couples E² to χ dynamics
- The gravitational "potential" comes from χ reduction

Key relationship (derived in Papers):
    G = c⁴/(4πκχ₀²)  or similar

This experiment:
1. Create two "masses" (bound E-structures)
2. Measure their gravitational acceleration toward each other
3. Vary χ₀ and κ to match expected Newtonian acceleration
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# TRUE SUBSTRATE SIMULATION (enforced)
# =============================================================================

class LFMSubstrate:
    """
    Enforces proper substrate dynamics.
    
    RULE: No static χ. No injected waves. Only coupled GOV-01 + GOV-02.
    """
    
    def __init__(self, Nx, Ny, L, chi_0, kappa, c=1.0):
        self.Nx = Nx
        self.Ny = Ny
        self.L = L
        self.dx = L / Nx
        self.chi_0 = chi_0
        self.kappa = kappa
        self.c = c
        self.dt = 0.3 * self.dx / c
        
        # Coordinate grids
        x = np.linspace(-L/2, L/2, Nx)
        y = np.linspace(-L/2, L/2, Ny)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        
        # Fields - BOTH dynamic
        self.E = np.zeros((Nx, Ny))
        self.E_prev = np.zeros((Nx, Ny))
        self.chi = chi_0 * np.ones((Nx, Ny))
        self.chi_prev = self.chi.copy()
        
        # Damping layer for boundaries
        self._setup_damping()
        
    def _setup_damping(self, width=20.0):
        dx_edge = np.minimum(self.X + self.L/2, self.L/2 - self.X)
        dy_edge = np.minimum(self.Y + self.L/2, self.L/2 - self.Y)
        d_edge = np.minimum(dx_edge, dy_edge)
        self.damp = np.clip(d_edge / width, 0, 1)
    
    def _laplacian(self, f):
        lap = np.zeros_like(f)
        lap[1:-1, 1:-1] = (
            f[2:, 1:-1] + f[:-2, 1:-1] +
            f[1:-1, 2:] + f[1:-1, :-2] -
            4*f[1:-1, 1:-1]
        ) / self.dx**2
        return lap
    
    def add_matter(self, x0, y0, amplitude, radius):
        """
        Add a bound E-structure (matter) at (x0, y0).
        This is the ONLY way to create "mass" in LFM.
        """
        R = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        self.E += amplitude * np.exp(-R**2 / (2 * radius**2))
        self.E_prev = self.E.copy()
    
    def step(self):
        """
        One time step of COUPLED GOV-01 + GOV-02 dynamics.
        This is the ONLY way to evolve the substrate.
        """
        # GOV-01: E dynamics
        lap_E = self._laplacian(self.E)
        E_accel = self.c**2 * lap_E - self.chi**2 * self.E
        E_new = 2*self.E - self.E_prev + self.dt**2 * E_accel
        
        # GOV-02: χ dynamics
        lap_chi = self._laplacian(self.chi)
        chi_accel = self.c**2 * lap_chi - self.kappa * self.E**2
        chi_new = 2*self.chi - self.chi_prev + self.dt**2 * chi_accel
        
        # Keep χ positive
        chi_new = np.maximum(chi_new, 0.01 * self.chi_0)
        
        # Boundary damping
        E_new = E_new * self.damp + self.E * (1 - self.damp) * 0.99
        chi_new = chi_new * self.damp + self.chi_0 * (1 - self.damp)
        
        # Update
        self.E_prev, self.E = self.E, E_new
        self.chi_prev, self.chi = self.chi, chi_new
    
    def get_mass_position(self, x0, y0, search_radius=30.0):
        """Track center of mass of a structure near (x0, y0)."""
        R = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        mask = R < search_radius
        E_sq = self.E**2 * mask
        total = E_sq.sum() + 1e-10
        cx = np.sum(self.X * E_sq) / total
        cy = np.sum(self.Y * E_sq) / total
        return cx, cy
    
    def get_chi_at(self, x, y):
        """Get χ value at position (x, y)."""
        ix = int((x + self.L/2) / self.dx)
        iy = int((y + self.L/2) / self.dx)
        ix = np.clip(ix, 0, self.Nx-1)
        iy = np.clip(iy, 0, self.Ny-1)
        return self.chi[ix, iy]


def measure_gravitational_acceleration(chi_0, kappa, separation=60.0):
    """
    Measure gravitational acceleration between two masses in LFM.
    
    Returns acceleration in lattice units.
    """
    # Create substrate
    sim = LFMSubstrate(Nx=300, Ny=200, L=200.0, chi_0=chi_0, kappa=kappa)
    
    # Place two identical masses
    mass_amp = 3.0
    mass_radius = 5.0
    m1_x, m1_y = -separation/2, 0.0
    m2_x, m2_y = +separation/2, 0.0
    
    sim.add_matter(m1_x, m1_y, mass_amp, mass_radius)
    sim.add_matter(m2_x, m2_y, mass_amp, mass_radius)
    
    # Let χ settle
    for _ in range(300):
        sim.step()
    
    # Track positions over time
    positions = []
    times = []
    
    for step in range(2000):
        sim.step()
        
        if step % 50 == 0:
            x1, y1 = sim.get_mass_position(m1_x, m1_y)
            x2, y2 = sim.get_mass_position(m2_x, m2_y)
            positions.append((x1, x2))
            times.append(step * sim.dt)
    
    # Calculate acceleration from position change
    positions = np.array(positions)
    times = np.array(times)
    
    # Separation over time
    sep = positions[:, 1] - positions[:, 0]
    
    # Fit quadratic: sep = sep_0 + v*t + 0.5*a*t²
    # For gravitational attraction, a should be negative (getting closer)
    if len(times) > 5:
        coeffs = np.polyfit(times, sep, 2)
        accel = 2 * coeffs[0]  # d²(sep)/dt² = 2*a
    else:
        accel = 0
    
    return accel, sep[0], sep[-1]


# =============================================================================
# CALIBRATION SCAN
# =============================================================================

print("=" * 70)
print("LFM CALIBRATION: Finding χ₀ from Gravitational Dynamics")
print("=" * 70)
print()
print("Strategy: Measure gravitational acceleration for different χ₀ values")
print("and find the relationship between parameters and effective G.")
print()

# Scan different χ₀ values with fixed κ
chi_0_values = [0.5, 1.0, 2.0, 4.0]
kappa = 2.0
separation = 60.0

results = []

print(f"Fixed κ = {kappa}, separation = {separation}")
print()
print(f"{'χ₀':<10} {'Acceleration':<15} {'Sep_initial':<15} {'Sep_final':<15} {'Δsep':<10}")
print("-" * 65)

for chi_0 in chi_0_values:
    accel, sep_i, sep_f = measure_gravitational_acceleration(chi_0, kappa, separation)
    delta_sep = sep_f - sep_i
    results.append({
        'chi_0': chi_0,
        'kappa': kappa,
        'accel': accel,
        'sep_initial': sep_i,
        'sep_final': sep_f,
        'delta_sep': delta_sep
    })
    print(f"{chi_0:<10.2f} {accel:<15.6f} {sep_i:<15.2f} {sep_f:<15.2f} {delta_sep:<10.4f}")

print()
print("=" * 70)
print("ANALYSIS")
print("=" * 70)
print()

# Extract scaling
chi_0s = np.array([r['chi_0'] for r in results])
accels = np.array([r['accel'] for r in results])
delta_seps = np.array([r['delta_sep'] for r in results])

# Fit power law: accel ∝ χ₀^n
log_chi = np.log(chi_0s)
log_accel = np.log(np.abs(accels) + 1e-10)
slope, intercept = np.polyfit(log_chi, log_accel, 1)

print(f"Power law fit: |a| ∝ χ₀^{slope:.2f}")
print()

# Key insight
print("INTERPRETATION:")
print("-" * 50)
if slope < -1:
    print(f"  Gravity WEAKENS as χ₀ increases (slope = {slope:.2f})")
    print(f"  This suggests: G ∝ 1/χ₀^{abs(slope):.1f}")
    print()
    print("  To match real G = 6.67×10⁻¹¹:")
    print("  If κ is order 1, then χ₀ must be VERY SMALL")
    print("  (consistent with χ₀ ~ H₀ ~ 10⁻¹⁸ Hz)")
elif slope > 1:
    print(f"  Gravity STRENGTHENS as χ₀ increases (slope = {slope:.2f})")
else:
    print(f"  Weak dependence on χ₀ (slope = {slope:.2f})")

print()
print("=" * 70)
print("KEY FINDING")
print("=" * 70)
print()
print("The relationship between χ₀, κ, and effective G tells us:")
print("  • χ₀ sets the 'stiffness' of the substrate")
print("  • κ sets the coupling strength between matter and χ")
print("  • Together they determine gravitational strength")
print()
print("To calibrate to real physics:")
print("  1. Match rotation curves → constrains κ/χ₀ ratio")
print("  2. Match local G → constrains absolute values")
print("  3. Cross-check with other phenomena")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('LFM Calibration: χ₀ vs Gravitational Strength', fontsize=14, fontweight='bold')

ax = axes[0]
ax.loglog(chi_0s, np.abs(accels), 'bo-', markersize=10, linewidth=2)
ax.set_xlabel('χ₀')
ax.set_ylabel('|Acceleration|')
ax.set_title(f'Gravity vs χ₀ (slope = {slope:.2f})')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(chi_0s, delta_seps, 'ro-', markersize=10, linewidth=2)
ax.axhline(0, color='gray', linestyle='--')
ax.set_xlabel('χ₀')
ax.set_ylabel('Δ(separation)')
ax.set_title('Masses approaching (negative = attraction)')
ax.grid(True, alpha=0.3)

plt.tight_layout()

output_dir = Path(__file__).parent / 'figures'
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'lfm_calibration.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print()
print(f"Figure saved to: {output_path}")
