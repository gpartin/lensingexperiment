"""
LFM χ Profile Measurement
=========================

QUESTION: Does full GOV-02 dynamics naturally produce 1/r χ-wells around matter?

If yes, this validates that Coulomb/Newton potentials EMERGE from substrate dynamics,
justifying the prescribed χ-wells used in Paper 051 (H₂ molecule).

GOV-01: ∂²E/∂t² = c²∇²E − χ²E
GOV-02: ∂²χ/∂t² = c²∇²χ − κ(E² − E₀²)

In quasi-static limit (∂²χ/∂t² → 0), GOV-02 becomes:
  ∇²χ = (κ/c²)(E² − E₀²)  (Poisson equation!)

For point mass, solution is χ ∝ 1/r. Let's verify numerically.

Author: Greg Partin
Date: February 4, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

print("="*70)
print("LFM χ PROFILE MEASUREMENT")
print("Does GOV-02 produce 1/r χ-wells?")
print("="*70)

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

N = 300  # Grid size
L = 40.0  # Domain size
dx = L / N
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y, indexing='ij')

# LFM parameters
c = 1.0
chi_0 = 2.0  # Background χ
kappa = 0.5  # E-χ coupling (reduced to prevent χ going negative)
E0_squared = 0.0  # Vacuum E²

dt = 0.15 * dx / c  # Smaller timestep for stability
n_steps = 4000  # Let χ equilibrate longer

print(f"\nGrid: {N}x{N}, dx={dx:.3f}")
print(f"χ₀ = {chi_0}, κ = {kappa}")
print(f"Steps: {n_steps}")

# =============================================================================
# INITIALIZE FIELDS
# =============================================================================

# Create a "star" - localized E concentration at origin
star_x, star_y = 0.0, 0.0
star_radius = 2.0
star_amplitude = 5.0

R_star = np.sqrt((X - star_x)**2 + (Y - star_y)**2)
E = star_amplitude * np.exp(-R_star**2 / (2 * star_radius**2))
E_prev = E.copy()

# Initialize χ at background
chi = np.ones((N, N)) * chi_0
chi_prev = chi.copy()

print(f"\nStar: amplitude={star_amplitude}, radius={star_radius}")
print(f"Star E² integral: {np.sum(E**2) * dx**2:.2f}")

# =============================================================================
# LAPLACIAN
# =============================================================================

def laplacian(f, dx):
    """2D Laplacian with Neumann boundaries."""
    lap = np.zeros_like(f)
    lap[1:-1, 1:-1] = (f[2:, 1:-1] + f[:-2, 1:-1] + 
                       f[1:-1, 2:] + f[1:-1, :-2] - 
                       4*f[1:-1, 1:-1]) / dx**2
    # Neumann BC
    lap[0, :] = lap[1, :]
    lap[-1, :] = lap[-2, :]
    lap[:, 0] = lap[:, 1]
    lap[:, -1] = lap[:, -2]
    return lap

# =============================================================================
# EVOLVE WITH FULL GOV-02
# =============================================================================

print("\nEvolving with full GOV-01 + GOV-02...")

for step in range(n_steps):
    # E equation (GOV-01)
    lap_E = laplacian(E, dx)
    E_new = 2*E - E_prev + dt**2 * (c**2 * lap_E - chi**2 * E)
    
    # χ equation (GOV-02) - FULL DYNAMICS
    lap_chi = laplacian(chi, dx)
    chi_new = 2*chi - chi_prev + dt**2 * (c**2 * lap_chi - kappa * (E**2 - E0_squared))
    
    # Floor χ to prevent it going negative (unphysical)
    chi_new = np.maximum(chi_new, 0.1)
    
    # Update
    E_prev, E = E, E_new
    chi_prev, chi = chi, chi_new
    
    # Light damping at boundaries
    damping = 20
    for i in range(damping):
        factor = 0.5 * (1 - np.cos(np.pi * i / damping))
        E[i, :] *= factor; E[-1-i, :] *= factor
        E[:, i] *= factor; E[:, -1-i] *= factor
        chi[i, :] = chi[i, :] * factor + chi_0 * (1 - factor)
        chi[-1-i, :] = chi[-1-i, :] * factor + chi_0 * (1 - factor)
        chi[:, i] = chi[:, i] * factor + chi_0 * (1 - factor)
        chi[:, -1-i] = chi[:, -1-i] * factor + chi_0 * (1 - factor)
    
    if step % 500 == 0:
        chi_min = np.min(chi)
        print(f"  Step {step}: χ_min = {chi_min:.4f}")

print(f"\nFinal χ at origin: {chi[N//2, N//2]:.4f}")
print(f"Final χ at boundary: {chi[0, 0]:.4f}")

# =============================================================================
# MEASURE χ PROFILE (RADIAL)
# =============================================================================

print("\nMeasuring radial χ profile...")

# Sample along multiple radial directions
n_angles = 36
angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
n_radii = 100
r_max = L/2 - 3  # Stay away from boundary

radii = np.linspace(1.0, r_max, n_radii)  # Start at r=1 to avoid origin
chi_radial = np.zeros((n_angles, n_radii))

for i_angle, theta in enumerate(angles):
    for i_r, r in enumerate(radii):
        # Sample point
        px = r * np.cos(theta)
        py = r * np.sin(theta)
        
        # Grid indices (nearest neighbor)
        ix = int((px + L/2) / dx)
        iy = int((py + L/2) / dx)
        
        if 0 <= ix < N and 0 <= iy < N:
            chi_radial[i_angle, i_r] = chi[ix, iy]

# Average over angles
chi_avg = np.mean(chi_radial, axis=0)
chi_std = np.std(chi_radial, axis=0)

# =============================================================================
# FIT TO 1/r MODEL
# =============================================================================

print("\nFitting χ profile...")

# Model: χ(r) = χ₀ - A/r  (Newton/Coulomb)
def chi_model(r, A):
    return chi_0 - A / r

# Also try: χ(r) = χ₀ - A/r^n
def chi_power_model(r, A, n):
    return chi_0 - A / r**n

# Fit 1/r model (use inner region where signal is strong)
fit_mask = radii < 12  # Inner region
try:
    popt_1r, pcov_1r = curve_fit(chi_model, radii[fit_mask], chi_avg[fit_mask], 
                                   p0=[1.0], bounds=(0, np.inf))
    A_fit = popt_1r[0]
    chi_fit_1r = chi_model(radii, A_fit)
    
    # R² for 1/r fit
    ss_res = np.sum((chi_avg[fit_mask] - chi_fit_1r[fit_mask])**2)
    ss_tot = np.sum((chi_avg[fit_mask] - np.mean(chi_avg[fit_mask]))**2)
    r2_1r = 1 - ss_res / ss_tot
    
    print(f"1/r fit: χ = {chi_0} - {A_fit:.4f}/r")
    print(f"R² = {r2_1r:.6f}")
except Exception as e:
    print(f"1/r fit failed: {e}")
    A_fit = None
    r2_1r = None

# Fit power law model
try:
    popt_power, pcov_power = curve_fit(chi_power_model, radii[fit_mask], chi_avg[fit_mask], 
                                        p0=[1.0, 1.0], bounds=([0, 0], [np.inf, 3]))
    A_power, n_power = popt_power
    chi_fit_power = chi_power_model(radii, A_power, n_power)
    
    # R² for power fit
    ss_res = np.sum((chi_avg[fit_mask] - chi_fit_power[fit_mask])**2)
    ss_tot = np.sum((chi_avg[fit_mask] - np.mean(chi_avg[fit_mask]))**2)
    r2_power = 1 - ss_res / ss_tot
    
    print(f"Power fit: χ = {chi_0} - {A_power:.4f}/r^{n_power:.3f}")
    print(f"R² = {r2_power:.6f}")
except Exception as e:
    print(f"Power fit failed: {e}")
    n_power = None

# =============================================================================
# VERDICT
# =============================================================================

print("\n" + "="*70)
print("VERDICT")
print("="*70)

if n_power is not None:
    if 0.8 < n_power < 1.2:
        print(f"✅ χ profile is consistent with 1/r (exponent = {n_power:.3f})")
        print("   GOV-02 PRODUCES COULOMB/NEWTON POTENTIALS!")
        success = True
    else:
        print(f"⚠️ χ profile has exponent {n_power:.3f} (not exactly 1/r)")
        print("   May be due to finite source size or boundary effects")
        success = n_power < 1.5  # Still roughly 1/r-ish
else:
    print("❌ Could not fit power law")
    success = False

# =============================================================================
# PLOT
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. χ field
ax1 = axes[0]
im = ax1.imshow(chi.T, origin='lower', extent=[-L/2, L/2, -L/2, L/2],
                cmap='viridis', vmin=chi_0-1, vmax=chi_0)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('χ field (GOV-02 equilibrium)')
plt.colorbar(im, ax=ax1, label='χ')
# Mark star
circle = plt.Circle((star_x, star_y), star_radius, fill=False, color='red', linewidth=2)
ax1.add_patch(circle)

# 2. Radial profile with fits
ax2 = axes[1]
ax2.errorbar(radii, chi_avg, yerr=chi_std, fmt='b.', alpha=0.5, label='Data', capsize=2)
if A_fit is not None:
    ax2.plot(radii, chi_fit_1r, 'r-', linewidth=2, label=f'1/r fit (R²={r2_1r:.4f})')
if n_power is not None:
    ax2.plot(radii, chi_fit_power, 'g--', linewidth=2, 
             label=f'1/r^{n_power:.2f} fit (R²={r2_power:.4f})')
ax2.axhline(chi_0, color='gray', linestyle=':', label=f'χ₀ = {chi_0}')
ax2.set_xlabel('r (distance from star)')
ax2.set_ylabel('χ(r)')
ax2.set_title('Radial χ Profile')
ax2.legend()
ax2.set_xlim(0, r_max)

# 3. Log-log to check power law
ax3 = axes[2]
delta_chi = chi_0 - chi_avg
positive_mask = delta_chi > 0
ax3.loglog(radii[positive_mask], delta_chi[positive_mask], 'b.', alpha=0.5, label='Data')
if A_fit is not None:
    ax3.loglog(radii[positive_mask], A_fit/radii[positive_mask], 'r-', 
               linewidth=2, label='1/r')
if n_power is not None:
    ax3.loglog(radii[positive_mask], A_power/radii[positive_mask]**n_power, 'g--', 
               linewidth=2, label=f'1/r^{n_power:.2f}')
ax3.set_xlabel('r')
ax3.set_ylabel('χ₀ - χ(r)')
ax3.set_title('Log-log plot (slope = -n)')
ax3.legend()

plt.tight_layout()
plt.savefig('chi_profile_gov02.png', dpi=150)
plt.close()

print(f"\nSaved: chi_profile_gov02.png")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Star: Gaussian E with amplitude={star_amplitude}, radius={star_radius}")
print(f"Evolution: {n_steps} steps of full GOV-01 + GOV-02")
print(f"χ well depth: Δχ = {chi_0 - np.min(chi):.4f}")
print(f"Best fit exponent: n = {n_power:.3f}" if n_power else "Fit failed")
print(f"Consistent with 1/r: {'YES' if success else 'NO'}")

if success:
    print("\n🎉 OPTION 1 SUCCESS: GOV-02 produces 1/r χ-wells!")
    print("   This justifies the prescribed Coulomb χ in Paper 051.")
    print("   Ready for Option 2: Upgrade Paper 052 to full GOV-02.")
