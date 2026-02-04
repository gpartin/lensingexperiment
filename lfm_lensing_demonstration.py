#!/usr/bin/env python3
"""
================================================================================
LFM Gravitational Lensing: First Principles Demonstration
================================================================================

This script demonstrates that gravitational light bending EMERGES from the
Lattice Field Medium (LFM) governing equations without any assumed Newtonian
or Einsteinian physics.

WHAT THIS PROVES:
    Waves propagating through an LFM lattice bend TOWARD concentrations of
    energy ("mass"), purely from the wave equation dynamics. No gravity is
    put in - it comes out.

THE ONLY EQUATIONS USED:
    
    GOV-01 (Wave Equation):
        ∂²E/∂t² = c²∇²E − χ²E
        
        Waves propagate and are influenced by the local χ field.
        
    GOV-03 (Chi Response):
        χ² = χ₀² − g⟨E²⟩_τ
        
        The χ field is reduced wherever wave energy (E²) is concentrated.
        This is how "mass" affects the substrate.

HOW TO RUN:
    python lfm_lensing_demonstration.py
    
    Output: Console results + figures/lfm_lensing_demo.png

REQUIREMENTS:
    numpy, matplotlib (standard scientific Python)

================================================================================
LICENSE: MIT License

Copyright (c) 2026 LFM Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

================================================================================
CONTACT:
    Repository: https://github.com/gpartin/Papers
    For questions about LFM theory, see the paper series at the repository.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SimulationConfig:
    """All simulation parameters in one place."""
    # Grid
    Nx: int = 500          # Grid points in x
    Ny: int = 300          # Grid points in y
    dx: float = 1.0        # Lattice spacing
    
    # Physics
    c: float = 1.0         # Wave speed
    chi_0: float = 1.0     # Background χ value
    g: float = 0.3         # E²-χ coupling strength
    
    # Mass (energy concentration at center)
    mass_amplitude: float = 1.0
    mass_radius: float = 10.0
    
    # Test wave
    wave_k: float = 0.5    # Wave number
    wave_width: float = 30.0  # Packet width
    
    # Simulation
    cfl_factor: float = 0.4  # CFL stability factor
    n_propagation_steps: int = 2000
    
    @property
    def dt(self) -> float:
        """Time step from CFL condition."""
        return self.cfl_factor * self.dx / self.c
    
    @property
    def omega(self) -> float:
        """Wave frequency from dispersion relation."""
        return np.sqrt(self.c**2 * self.wave_k**2 + self.chi_0**2)
    
    @property
    def group_velocity(self) -> float:
        """Group velocity of wave packet."""
        return self.c**2 * self.wave_k / self.omega


# =============================================================================
# CORE PHYSICS: The only equations we use
# =============================================================================

def compute_chi_field(X: np.ndarray, Y: np.ndarray, config: SimulationConfig) -> np.ndarray:
    """
    Compute equilibrium χ field from a central mass using GOV-03.
    
    GOV-03: χ² = χ₀² − g⟨E²⟩
    
    The mass is modeled as an oscillating E field at the center.
    For a harmonic oscillator, ⟨E²⟩ = 0.5 × amplitude².
    
    This is the ONLY place χ comes from - no Newtonian 1/r assumed!
    """
    R = np.sqrt(X**2 + Y**2)
    
    # Mass profile: Gaussian energy concentration
    mass_profile = config.mass_amplitude * np.exp(-R**2 / (2 * config.mass_radius**2))
    
    # Time-averaged E² for oscillating mass
    E_sq_avg = 0.5 * mass_profile**2
    
    # GOV-03: χ² = χ₀² − g⟨E²⟩
    chi_sq = config.chi_0**2 - config.g * E_sq_avg
    chi_sq = np.maximum(chi_sq, 0.01)  # Keep χ positive
    
    return np.sqrt(chi_sq)


def laplacian_2d(field: np.ndarray, dx: float) -> np.ndarray:
    """
    Discrete Laplacian using central differences.
    
    ∇²f ≈ (f[i+1,j] + f[i-1,j] + f[i,j+1] + f[i,j-1] - 4f[i,j]) / dx²
    """
    lap = np.zeros_like(field)
    lap[1:-1, 1:-1] = (
        field[2:, 1:-1] + field[:-2, 1:-1] +
        field[1:-1, 2:] + field[1:-1, :-2] -
        4 * field[1:-1, 1:-1]
    ) / dx**2
    return lap


def propagate_wave_GOV01(E: np.ndarray, E_prev: np.ndarray, 
                          chi: np.ndarray, config: SimulationConfig) -> np.ndarray:
    """
    One time step of wave propagation using GOV-01.
    
    GOV-01: ∂²E/∂t² = c²∇²E − χ²E
    
    Leapfrog integration (Verlet 1967):
        E_new = 2E - E_prev + dt² × (c²∇²E - χ²E)
    
    This is the ONLY wave equation we use - no modifications!
    """
    lap_E = laplacian_2d(E, config.dx)
    E_accel = config.c**2 * lap_E - chi**2 * E
    E_new = 2 * E - E_prev + config.dt**2 * E_accel
    return E_new


# =============================================================================
# EXPERIMENT
# =============================================================================

def compute_centroid(field: np.ndarray, X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
    """Compute energy-weighted centroid of a field."""
    f_sq = field**2
    total = f_sq.sum() + 1e-10
    cx = np.sum(X * f_sq) / total
    cy = np.sum(Y * f_sq) / total
    return cx, cy


def create_traveling_wave(X: np.ndarray, Y: np.ndarray, 
                          x0: float, y0: float,
                          config: SimulationConfig,
                          time_offset: float = 0) -> np.ndarray:
    """
    Create a traveling wave packet.
    
    For proper initialization of a rightward-traveling wave:
    - At t=0: E = envelope(x-x0) × cos(k×x)
    - At t=-dt: E = envelope(x-x0+v×dt) × cos(k×x + ω×dt)
    """
    v = config.group_velocity
    omega = config.omega
    
    x_shift = v * time_offset
    phase = -omega * time_offset
    
    envelope = np.exp(-((X - x0 - x_shift)**2 + (Y - y0)**2) / (2 * config.wave_width**2))
    carrier = np.cos(config.wave_k * X + phase)
    
    return envelope * carrier


def run_single_trajectory(impact_param: float, chi: np.ndarray, 
                          X: np.ndarray, Y: np.ndarray,
                          config: SimulationConfig) -> dict:
    """
    Propagate a wave packet past the mass and measure deflection.
    
    Returns trajectory and deflection data.
    """
    # Starting position (far left of grid)
    x0 = -config.Nx * config.dx / 2 + 50
    
    # Initialize wave packet
    E = create_traveling_wave(X, Y, x0, impact_param, config, time_offset=0)
    E_prev = create_traveling_wave(X, Y, x0, impact_param, config, time_offset=-config.dt)
    
    # Record initial position
    x_init, y_init = compute_centroid(E, X, Y)
    
    trajectory_x = [x_init]
    trajectory_y = [y_init]
    
    # Propagate using ONLY GOV-01
    for step in range(config.n_propagation_steps):
        E_new = propagate_wave_GOV01(E, E_prev, chi, config)
        E_prev = E.copy()
        E = E_new
        
        # Record trajectory periodically
        if step % 200 == 0:
            cx, cy = compute_centroid(E, X, Y)
            trajectory_x.append(cx)
            trajectory_y.append(cy)
    
    # Final position
    x_final, y_final = compute_centroid(E, X, Y)
    
    delta_x = x_final - x_init
    delta_y = y_final - y_init
    
    # Deflection angle (small angle approximation)
    theta = delta_y / delta_x if abs(delta_x) > 10 else 0
    
    return {
        'impact_param': impact_param,
        'x_init': x_init,
        'y_init': y_init,
        'x_final': x_final,
        'y_final': y_final,
        'delta_x': delta_x,
        'delta_y': delta_y,
        'theta_rad': theta,
        'theta_mrad': theta * 1000,
        'trajectory_x': trajectory_x,
        'trajectory_y': trajectory_y,
        'direction': 'TOWARD mass' if delta_y < 0 else 'AWAY from mass' if delta_y > 0 else 'none'
    }


def run_lensing_experiment():
    """
    Main experiment: demonstrate gravitational lensing from first principles.
    """
    print("=" * 70)
    print("LFM GRAVITATIONAL LENSING DEMONSTRATION")
    print("=" * 70)
    print()
    print("This experiment uses ONLY the LFM governing equations:")
    print("  GOV-01: ∂²E/∂t² = c²∇²E − χ²E")
    print("  GOV-03: χ² = χ₀² − g⟨E²⟩")
    print()
    print("No Newtonian physics. No General Relativity. Just wave dynamics.")
    print()
    
    # Configuration
    config = SimulationConfig()
    
    # Create coordinate grids
    Lx = config.Nx * config.dx
    Ly = config.Ny * config.dx
    x = np.linspace(-Lx/2, Lx/2, config.Nx)
    y = np.linspace(-Ly/2, Ly/2, config.Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    print(f"Grid: {config.Nx} × {config.Ny} points")
    print(f"Domain: [{-Lx/2:.0f}, {Lx/2:.0f}] × [{-Ly/2:.0f}, {Ly/2:.0f}]")
    print()
    
    # Step 1: Compute χ field from mass (using GOV-03)
    print("STEP 1: Computing χ field from central mass...")
    print("-" * 50)
    
    chi = compute_chi_field(X, Y, config)
    
    chi_center = chi[config.Nx//2, config.Ny//2]
    chi_edge = chi[0, config.Ny//2]
    chi_reduction = (1 - chi_center / config.chi_0) * 100
    
    print(f"  Mass amplitude: {config.mass_amplitude}")
    print(f"  Mass radius: {config.mass_radius}")
    print(f"  Coupling g: {config.g}")
    print()
    print(f"  χ at center (in mass): {chi_center:.4f}")
    print(f"  χ at edge (far from mass): {chi_edge:.4f}")
    print(f"  χ reduction at center: {chi_reduction:.1f}%")
    print()
    print("  → χ field emerged from E² concentration (no 1/r assumed!)")
    print()
    
    # Step 2: Propagate test waves
    print("STEP 2: Propagating test waves past the mass...")
    print("-" * 50)
    
    print(f"  Wave number k: {config.wave_k}")
    print(f"  Wave frequency ω: {config.omega:.3f}")
    print(f"  ω/χ₀ ratio: {config.omega/config.chi_0:.2f}")
    print(f"  Group velocity: {config.group_velocity:.3f}")
    print()
    
    # Test at multiple impact parameters
    impact_params = [20.0, 40.0, 60.0]
    results = []
    
    for b in impact_params:
        print(f"  Testing impact parameter b = {b}...")
        result = run_single_trajectory(b, chi, X, Y, config)
        results.append(result)
        print(f"    Path: ({result['x_init']:.0f}, {result['y_init']:.1f}) → "
              f"({result['x_final']:.0f}, {result['y_final']:.2f})")
        print(f"    Δy = {result['delta_y']:.4f}, θ = {result['theta_mrad']:.2f} mrad")
        print(f"    Direction: {result['direction']}")
    
    # Step 3: Analysis
    print()
    print("STEP 3: Results Analysis")
    print("=" * 70)
    print()
    print("| Impact b | Δy (toward center) | θ (mrad) | Direction |")
    print("|----------|-------------------|----------|-----------|")
    
    all_toward = True
    for r in results:
        direction_symbol = "✓" if r['delta_y'] < 0 else "✗"
        print(f"| {r['impact_param']:8.0f} | {r['delta_y']:17.4f} | {r['theta_mrad']:8.2f} | {r['direction']:9s} |")
        if r['delta_y'] >= 0:
            all_toward = False
    
    print()
    
    # Scaling analysis
    print("1/b SCALING TEST (gravity-like behavior):")
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            if abs(results[j]['theta_rad']) > 1e-10:
                ratio = results[i]['theta_rad'] / results[j]['theta_rad']
                expected = results[j]['impact_param'] / results[i]['impact_param']
                print(f"  θ(b={results[i]['impact_param']:.0f})/θ(b={results[j]['impact_param']:.0f}) = "
                      f"{ratio:.2f} (gravity predicts ~{expected:.2f})")
    
    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    if all_toward:
        print("""
    ✓ GRAVITATIONAL LENSING EMERGES FROM LFM!
    
    All test waves bent TOWARD the mass, using ONLY:
    
    • GOV-01: ∂²E/∂t² = c²∇²E − χ²E  (wave propagation)
    • GOV-03: χ² = χ₀² − g⟨E²⟩       (χ responds to energy)
    
    No Newtonian formulas. No Einstein field equations.
    Gravity emerged purely from wave dynamics on a lattice.
    
    This demonstrates that LFM can produce gravitational effects
    from first principles, answering the scalar gravity concern.
        """)
    else:
        print("    Results show mixed behavior - see data above.")
    
    # Create visualization
    create_figure(chi, results, config, X, Y, chi_center)
    
    # Save data
    save_results(results, config, chi_center)
    
    return results


def create_figure(chi: np.ndarray, results: List[dict], 
                  config: SimulationConfig,
                  X: np.ndarray, Y: np.ndarray,
                  chi_center: float):
    """Create publication-quality figure."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('LFM Gravitational Lensing: First Principles Demonstration', 
                 fontsize=14, fontweight='bold')
    
    Lx = config.Nx * config.dx
    Ly = config.Ny * config.dx
    extent = [-Lx/2, Lx/2, -Ly/2, Ly/2]
    
    # Panel 1: χ field
    ax1 = axes[0, 0]
    im = ax1.imshow(chi.T, origin='lower', extent=extent, 
                    cmap='viridis', aspect='equal')
    ax1.plot(0, 0, 'ro', markersize=12, label='Mass (E² concentration)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'χ Field from GOV-03\n(center: χ = {chi_center:.3f}, '
                  f'reduction: {(1-chi_center)*100:.1f}%)')
    ax1.legend(loc='upper right')
    plt.colorbar(im, ax=ax1, label='χ')
    
    # Panel 2: Wave trajectories
    ax2 = axes[0, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for r, color in zip(results, colors):
        ax2.plot(r['trajectory_x'], r['trajectory_y'], 'o-', 
                 color=color, markersize=4, linewidth=1.5,
                 label=f"b = {r['impact_param']:.0f}")
    ax2.axhline(y=0, color='red', linewidth=2, linestyle='-', alpha=0.7)
    ax2.axvline(x=0, color='red', linewidth=1, linestyle='--', alpha=0.5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Wave Packet Trajectories\n(propagated using GOV-01)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Deflection vs impact parameter
    ax3 = axes[1, 0]
    bs = [r['impact_param'] for r in results]
    thetas = [abs(r['theta_mrad']) for r in results]
    ax3.plot(bs, thetas, 'bo-', markersize=12, linewidth=2)
    ax3.set_xlabel('Impact parameter b')
    ax3.set_ylabel('|θ| (mrad)')
    ax3.set_title('Deflection Magnitude vs Distance')
    ax3.grid(True, alpha=0.3)
    
    # Add reference lines
    if thetas[0] > 0:
        b_ref = np.array(bs)
        theta_1_over_b = thetas[0] * bs[0] / b_ref
        ax3.plot(b_ref, theta_1_over_b, 'g--', alpha=0.5, label='∝ 1/b (Newtonian)')
        ax3.legend()
    
    # Panel 4: Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    toward_count = sum(1 for r in results if r['delta_y'] < 0)
    
    summary = f"""
    EXPERIMENT SUMMARY
    ==================
    
    Equations Used (and ONLY these):
    ─────────────────────────────────
    GOV-01: ∂²E/∂t² = c²∇²E − χ²E
    GOV-03: χ² = χ₀² − g⟨E²⟩
    
    Numerical Method:
    ─────────────────────────────────
    • Leapfrog time integration (Verlet 1967)
    • Central difference Laplacian
    • No assumed gravitational physics
    
    Parameters:
    ─────────────────────────────────
    χ₀ = {config.chi_0}
    g = {config.g}
    Wave ω/χ₀ = {config.omega/config.chi_0:.2f}
    
    Results:
    ─────────────────────────────────
    Tests toward mass: {toward_count}/{len(results)}
    
    CONCLUSION:
    ─────────────────────────────────
    {'✓ GRAVITY EMERGES from LFM!' if toward_count == len(results) else 'Mixed results'}
    """
    
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen' if toward_count == len(results) else 'wheat', 
                       alpha=0.8))
    
    plt.tight_layout()
    
    output_dir = Path(__file__).parent / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'lfm_lensing_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nFigure saved to: {output_path}")


def save_results(results: List[dict], config: SimulationConfig, chi_center: float):
    """Save results to JSON for reproducibility."""
    
    output = {
        'experiment': 'LFM Gravitational Lensing Demonstration',
        'equations_used': [
            'GOV-01: d²E/dt² = c²∇²E - χ²E',
            'GOV-03: χ² = χ₀² - g⟨E²⟩'
        ],
        'parameters': {
            'chi_0': config.chi_0,
            'g': config.g,
            'mass_amplitude': config.mass_amplitude,
            'mass_radius': config.mass_radius,
            'wave_k': config.wave_k,
            'omega': config.omega,
            'omega_over_chi0': config.omega / config.chi_0
        },
        'chi_field': {
            'at_center': chi_center,
            'reduction_percent': (1 - chi_center / config.chi_0) * 100
        },
        'results': [
            {
                'impact_param': r['impact_param'],
                'delta_y': r['delta_y'],
                'theta_mrad': r['theta_mrad'],
                'direction': r['direction']
            }
            for r in results
        ],
        'conclusion': 'Gravity emerges from LFM equations' if all(r['delta_y'] < 0 for r in results) else 'Mixed'
    }
    
    output_path = Path(__file__).parent / 'lfm_lensing_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    results = run_lensing_experiment()
