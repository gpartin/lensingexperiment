#!/usr/bin/env python3
"""
================================================================================
LFM RADIATION FROM EXCITED ATOM - Proper Coupled E/χ Dynamics
================================================================================

HOW LIGHT IS CREATED IN OUR UNIVERSE:
    The easiest ways:
    1. Heat something → atoms vibrate → radiate (thermal/blackbody)
    2. Excite an atom → electron oscillates → radiates (spectral emission)
    3. Accelerate a charge → radiates (antenna, bremsstrahlung)

All of these come down to: OSCILLATING MATTER → RADIATION

IN LFM SUBSTRATE:
    - "Atom" = bound oscillating E-field structure (nucleus + electron)
    - The χ field responds to E² via GOV-02 or GOV-03
    - When the electron oscillates, it creates oscillating E² → oscillating χ
    - These oscillations propagate outward as RADIATION

THIS EXPERIMENT:
    1. Create a hydrogen-like bound state (nucleus + orbiting electron)
    2. Excite it (perturb the electron)
    3. Use FULLY COUPLED E/χ dynamics (GOV-01 + GOV-02)
    4. Watch it radiate naturally
    5. The outgoing waves ARE the "light"

KEY DIFFERENCE FROM BEFORE:
    - Previous: χ was STATIC or used GOV-03 (fast-response approximation)
    - Now: χ is DYNAMIC via GOV-02: ∂²χ/∂t² = c²∇²χ − κ(E² − E₀²)
    - This is the full coupled wave system - mother nature's way

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass


@dataclass
class CoupledParams:
    """Parameters for fully coupled E-χ simulation."""
    # Grid
    N: int = 256
    L: float = 100.0
    
    # Physics (GOV-01 + GOV-02)
    c: float = 1.0
    chi_0: float = 1.0        # Background χ
    kappa: float = 0.5        # κ in GOV-02: coupling strength
    E0_sq: float = 0.0        # Background E² (vacuum)
    
    # Atom
    nucleus_amp: float = 3.0
    nucleus_width: float = 3.0
    electron_radius: float = 15.0  # Initial orbital radius
    electron_amp: float = 1.0
    electron_width: float = 4.0
    
    # Excitation (perturbation to electron)
    excitation_amp: float = 0.5  # Kick to start radiation
    
    @property
    def dx(self) -> float:
        return self.L / self.N
    
    @property
    def dt(self) -> float:
        return 0.25 * self.dx / self.c


def laplacian_2d(f: np.ndarray, dx: float) -> np.ndarray:
    """Central difference Laplacian."""
    lap = np.zeros_like(f)
    lap[1:-1, 1:-1] = (
        f[2:, 1:-1] + f[:-2, 1:-1] +
        f[1:-1, 2:] + f[1:-1, :-2] -
        4 * f[1:-1, 1:-1]
    ) / dx**2
    return lap


class CoupledRadiationSim:
    """
    Fully coupled E-χ simulation with radiating atom.
    
    Uses the FUNDAMENTAL coupled wave system:
        GOV-01: ∂²E/∂t² = c²∇²E − χ²E
        GOV-02: ∂²χ/∂t² = c²∇²χ − κ(E² − E₀²)
    
    Both E and χ are dynamical fields that evolve together.
    """
    
    def __init__(self, params: CoupledParams):
        self.p = params
        
        # Create grid
        x = np.linspace(-params.L/2, params.L/2, params.N)
        y = np.linspace(-params.L/2, params.L/2, params.N)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        self.R = np.sqrt(self.X**2 + self.Y**2)
        
        # Initialize E field (empty)
        self.E = np.zeros((params.N, params.N))
        self.E_prev = np.zeros((params.N, params.N))
        
        # Initialize χ field (background)
        self.chi = np.ones((params.N, params.N)) * params.chi_0
        self.chi_prev = self.chi.copy()
        
        self.t = 0.0
        
    def create_atom(self):
        """
        Create a hydrogen-like bound state.
        
        - Nucleus at origin (heavy, stationary E concentration)
        - Electron orbiting at radius r_e
        
        This creates a χ-well around the nucleus that traps the electron.
        """
        p = self.p
        
        # Nucleus: heavy Gaussian at origin (we'll keep it fixed)
        self.nucleus = p.nucleus_amp * np.exp(-self.R**2 / (2 * p.nucleus_width**2))
        
        # Electron: wave packet at (r_e, 0)
        R_e = np.sqrt((self.X - p.electron_radius)**2 + self.Y**2)
        self.electron = p.electron_amp * np.exp(-R_e**2 / (2 * p.electron_width**2))
        self.electron_prev = self.electron.copy()
        
        # Total E field
        self.E = self.nucleus + self.electron
        self.E_prev = self.E.copy()
        
        # Let χ equilibrate to the E² distribution
        # Quasi-static initial condition: χ² ≈ χ₀² - (κ/c²)(E² - E₀²)
        E_sq = self.E**2
        chi_sq = p.chi_0**2 - (p.kappa / p.c**2) * (E_sq - p.E0_sq)
        chi_sq = np.maximum(chi_sq, 0.1)
        self.chi = np.sqrt(chi_sq)
        self.chi_prev = self.chi.copy()
        
        print("Created hydrogen-like atom:")
        print(f"  Nucleus at origin, amp = {p.nucleus_amp}")
        print(f"  Electron at r = {p.electron_radius}")
        chi_center = self.chi[p.N//2, p.N//2]
        chi_edge = self.chi[0, p.N//2]
        print(f"  χ at center: {chi_center:.3f}")
        print(f"  χ at edge: {chi_edge:.3f}")
        print(f"  χ reduction: {(1 - chi_center/chi_edge)*100:.1f}%")
        
    def excite_electron(self):
        """
        Excite the electron to make it radiate.
        
        In our universe: shine light on atom, heat it up, collide with it
        In LFM: give the electron a velocity kick
        
        The excited electron will oscillate and radiate.
        """
        p = self.p
        
        # Give electron a velocity in y-direction (tangential to orbit)
        # This is done by making E_prev slightly shifted
        R_e = np.sqrt((self.X - p.electron_radius)**2 + self.Y**2)
        electron = p.electron_amp * np.exp(-R_e**2 / (2 * p.electron_width**2))
        
        # Velocity kick: shift the "previous" position
        # This creates initial velocity = (current - prev) / dt
        kick_y = p.excitation_amp * p.dt * 0.5  # Tangential kick
        R_e_shifted = np.sqrt((self.X - p.electron_radius)**2 + (self.Y - kick_y)**2)
        electron_prev = p.electron_amp * np.exp(-R_e_shifted**2 / (2 * p.electron_width**2))
        
        # Update state
        self.electron = electron
        self.electron_prev = electron_prev
        self.E = self.nucleus + self.electron
        self.E_prev = self.nucleus + self.electron_prev
        
        print(f"\nExcited electron with tangential kick (amp = {p.excitation_amp})")
        print("Electron will now oscillate and RADIATE...")
        
    def step(self):
        """
        Evolve the FULLY COUPLED system using GOV-01 + GOV-02.
        
        GOV-01: ∂²E/∂t² = c²∇²E − χ²E
        GOV-02: ∂²χ/∂t² = c²∇²χ − κ(E² − E₀²)
        
        This is the fundamental coupled wave system.
        Both fields evolve dynamically based on each other.
        """
        p = self.p
        
        # GOV-01: Update E field
        # Note: nucleus is kept fixed (infinite mass approximation)
        lap_E = laplacian_2d(self.electron, p.dx)
        E_accel = p.c**2 * lap_E - self.chi**2 * self.electron
        electron_new = 2 * self.electron - self.electron_prev + p.dt**2 * E_accel
        
        # GOV-02: Update χ field
        E_total = self.nucleus + self.electron
        E_sq = E_total**2
        lap_chi = laplacian_2d(self.chi, p.dx)
        chi_accel = p.c**2 * lap_chi - p.kappa * (E_sq - p.E0_sq)
        chi_new = 2 * self.chi - self.chi_prev + p.dt**2 * chi_accel
        
        # Keep χ positive
        chi_new = np.maximum(chi_new, 0.1)
        
        # Absorbing boundary for outgoing waves
        damping_width = 20
        damping = np.ones_like(self.E)
        for i in range(damping_width):
            factor = 1 - 0.03 * (damping_width - i) / damping_width
            damping[i, :] *= factor
            damping[-(i+1), :] *= factor
            damping[:, i] *= factor
            damping[:, -(i+1)] *= factor
        
        electron_new *= damping
        chi_new = p.chi_0 + (chi_new - p.chi_0) * damping
        
        # Update state
        self.electron_prev = self.electron.copy()
        self.electron = electron_new
        
        self.chi_prev = self.chi.copy()
        self.chi = chi_new
        
        self.E = self.nucleus + self.electron
        self.E_prev = self.nucleus + self.electron_prev
        
        self.t += p.dt
        
    def measure_radiation(self, r_detect: float) -> float:
        """Measure E² in a ring at radius r_detect."""
        mask = (self.R > r_detect - 2) & (self.R < r_detect + 2)
        return np.mean(self.E**2 * mask) if mask.sum() > 0 else 0
    
    def get_electron_position(self) -> tuple:
        """Get electron center of mass."""
        e_sq = self.electron**2
        mask = self.R > 5  # Exclude nucleus region
        e_sq_masked = e_sq * mask
        total = e_sq_masked.sum() + 1e-10
        cx = np.sum(self.X * e_sq_masked) / total
        cy = np.sum(self.Y * e_sq_masked) / total
        return cx, cy


def run_radiation_experiment():
    print("=" * 70)
    print("LFM RADIATION FROM EXCITED ATOM")
    print("Fully Coupled E-χ Dynamics (GOV-01 + GOV-02)")
    print("=" * 70)
    print()
    
    # Create simulation
    params = CoupledParams()
    sim = CoupledRadiationSim(params)
    
    print(f"Grid: {params.N} × {params.N}")
    print(f"Domain: {params.L} × {params.L}")
    print(f"dt = {params.dt:.5f}")
    print()
    print("PHYSICS:")
    print("  GOV-01: ∂²E/∂t² = c²∇²E − χ²E")
    print("  GOV-02: ∂²χ/∂t² = c²∇²χ − κ(E² − E₀²)")
    print("  Both E and χ evolve dynamically!")
    print()
    
    # Create atom
    sim.create_atom()
    
    # Excite it
    sim.excite_electron()
    
    # Run simulation
    n_steps = 3000
    print(f"\nEvolving for {n_steps} steps...")
    print("Watching for radiation at r = 30 and r = 40...")
    print()
    
    radiation_r30 = []
    radiation_r40 = []
    electron_traj = []
    times = []
    snapshots = []
    
    for step in range(n_steps):
        sim.step()
        
        if step % 50 == 0:
            rad30 = sim.measure_radiation(30)
            rad40 = sim.measure_radiation(40)
            radiation_r30.append(rad30)
            radiation_r40.append(rad40)
            times.append(sim.t)
            
            ex, ey = sim.get_electron_position()
            electron_traj.append((ex, ey))
        
        if step % 500 == 0:
            ex, ey = sim.get_electron_position()
            r_e = np.sqrt(ex**2 + ey**2)
            rad30 = sim.measure_radiation(30)
            rad40 = sim.measure_radiation(40)
            print(f"Step {step:4d}: r_e = {r_e:.2f}, E²(r=30) = {rad30:.2e}, E²(r=40) = {rad40:.2e}")
            
            if rad30 > 1e-6 or rad40 > 1e-6:
                print("         → RADIATION DETECTED!")
            
            snapshots.append({
                'E': sim.E.copy(),
                'chi': sim.chi.copy(),
                'electron': sim.electron.copy(),
                't': sim.t
            })
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    radiation_r30 = np.array(radiation_r30)
    radiation_r40 = np.array(radiation_r40)
    
    # Did radiation propagate outward?
    max_rad30 = np.max(radiation_r30)
    max_rad40 = np.max(radiation_r40)
    
    print(f"\nPeak E² at r=30: {max_rad30:.2e}")
    print(f"Peak E² at r=40: {max_rad40:.2e}")
    
    if max_rad30 > 1e-6 and max_rad40 > 1e-6:
        print("\n✓ RADIATION WAS EMITTED AND PROPAGATED!")
        print("  The oscillating electron created outgoing E-waves.")
        print("  These waves ARE the 'light' in LFM.")
    else:
        print("\n⚠ Radiation not clearly detected")
        print("  May need more time or different parameters")
    
    # Create visualization
    create_figure(sim, snapshots, radiation_r30, radiation_r40, times, electron_traj, params)
    
    return sim, snapshots


def create_figure(sim, snapshots, rad30, rad40, times, traj, params):
    """Create visualization of radiation emission."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    L = params.L
    extent = [-L/2, L/2, -L/2, L/2]
    
    # Panel 1: Initial E field
    ax = axes[0, 0]
    if len(snapshots) > 0:
        im = ax.imshow(snapshots[0]['E'].T, origin='lower', extent=extent, 
                      cmap='RdBu', aspect='equal')
        ax.set_title(f"E field at t = {snapshots[0]['t']:.1f}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax)
    
    # Panel 2: Mid-simulation E field
    ax = axes[0, 1]
    mid_idx = len(snapshots) // 2
    if mid_idx > 0:
        im = ax.imshow(snapshots[mid_idx]['E'].T, origin='lower', extent=extent,
                      cmap='RdBu', aspect='equal')
        ax.set_title(f"E field at t = {snapshots[mid_idx]['t']:.1f}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax)
    
    # Panel 3: Final E field
    ax = axes[0, 2]
    if len(snapshots) > 0:
        im = ax.imshow(snapshots[-1]['E'].T, origin='lower', extent=extent,
                      cmap='RdBu', aspect='equal')
        ax.set_title(f"E field at t = {snapshots[-1]['t']:.1f}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        # Draw detection rings
        circle30 = plt.Circle((0, 0), 30, fill=False, color='green', linestyle='--', linewidth=2)
        circle40 = plt.Circle((0, 0), 40, fill=False, color='orange', linestyle='--', linewidth=2)
        ax.add_patch(circle30)
        ax.add_patch(circle40)
        plt.colorbar(im, ax=ax)
    
    # Panel 4: Radiation intensity vs time
    ax = axes[1, 0]
    ax.semilogy(times, rad30 + 1e-12, 'g-', label='E² at r=30', linewidth=2)
    ax.semilogy(times, rad40 + 1e-12, 'orange', label='E² at r=40', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('E² (log scale)')
    ax.set_title('Radiation Intensity at Detection Rings')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 5: Electron trajectory
    ax = axes[1, 1]
    traj = np.array(traj)
    if len(traj) > 0:
        ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=1)
        ax.plot(0, 0, 'ro', markersize=10, label='Nucleus')
        ax.plot(traj[-1, 0], traj[-1, 1], 'bo', markersize=8, label='Electron (final)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Electron Trajectory')
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    # Panel 6: χ field evolution
    ax = axes[1, 2]
    if len(snapshots) > 0:
        im = ax.imshow(snapshots[-1]['chi'].T, origin='lower', extent=extent,
                      cmap='viridis', aspect='equal')
        ax.set_title(f"χ field at t = {snapshots[-1]['t']:.1f}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, label='χ')
    
    fig.suptitle('LFM Radiation from Excited Atom\n'
                '(Fully Coupled GOV-01 + GOV-02)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    output_dir = Path(__file__).parent / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'lfm_coupled_radiation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nFigure saved to: {output_path}")


if __name__ == '__main__':
    sim, snapshots = run_radiation_experiment()
