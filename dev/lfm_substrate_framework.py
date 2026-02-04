#!/usr/bin/env python3
"""
================================================================================
LFM Substrate Framework
================================================================================

This module provides the ONLY correct way to run LFM simulations.
It enforces the substrate paradigm and prevents common mistakes.

RULES (enforced by this framework):
    1. χ is NEVER static - it always evolves via GOV-02
    2. Light is NEVER injected - radiation emerges from oscillating matter
    3. All physics emerges from coupled GOV-01 + GOV-02 evolution
    4. Matter = bound E-structures, not external parameters

USAGE:
    from lfm_substrate_framework import LFMUniverse, Matter

    # Create universe
    universe = LFMUniverse(size=200, chi_0=1.0, kappa=2.0)
    
    # Add matter (the ONLY way to create mass)
    universe.add_matter(x=0, y=0, amplitude=3.0, radius=5.0, name="star")
    universe.add_matter(x=50, y=0, amplitude=1.0, radius=3.0, name="planet")
    
    # Evolve (ONLY GOV-01 + GOV-02, nothing else)
    for step in range(1000):
        universe.evolve()
    
    # Observe (passive measurement, doesn't affect simulation)
    universe.measure_radiation_at(x=30, y=0)

NO SHORTCUTS. NO CHEATING. JUST SUBSTRATE DYNAMICS.
================================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import warnings


@dataclass
class Matter:
    """
    A piece of matter in the LFM universe.
    
    Matter IS a bound E-structure on the substrate.
    It's not a parameter - it's actual oscillating field energy.
    """
    name: str
    x0: float
    y0: float
    amplitude: float
    radius: float
    
    # Tracking (updated during evolution)
    current_x: float = field(init=False)
    current_y: float = field(init=False)
    
    def __post_init__(self):
        self.current_x = self.x0
        self.current_y = self.y0


class LFMUniverse:
    """
    A proper LFM universe with coupled E-χ dynamics.
    
    This is the ONLY way to run LFM simulations correctly.
    """
    
    def __init__(self, size: float, chi_0: float, kappa: float, 
                 resolution: int = 400, c: float = 1.0):
        """
        Create an LFM universe.
        
        Args:
            size: Physical size of the simulation domain
            chi_0: Background χ value (substrate stiffness)
            kappa: Coupling constant in GOV-02
            resolution: Grid points per dimension
            c: Wave speed
        """
        self.size = size
        self.chi_0 = chi_0
        self.kappa = kappa
        self.c = c
        self.resolution = resolution
        
        # Grid setup
        self.Nx = resolution
        self.Ny = resolution
        self.dx = size / resolution
        self.dt = 0.3 * self.dx / c  # CFL condition
        
        # Coordinates
        x = np.linspace(-size/2, size/2, self.Nx)
        y = np.linspace(-size/2, size/2, self.Ny)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        
        # THE FIELDS - both are dynamical
        self.E = np.zeros((self.Nx, self.Ny))
        self.E_prev = np.zeros((self.Nx, self.Ny))
        self.chi = chi_0 * np.ones((self.Nx, self.Ny))
        self.chi_prev = self.chi.copy()
        
        # Boundary damping
        self._setup_boundaries()
        
        # Matter registry
        self.matter: Dict[str, Matter] = {}
        
        # Time tracking
        self.time = 0.0
        self.step_count = 0
        
        # Measurement points (passive observers)
        self._observers: List[Tuple[float, float, List[float]]] = []
    
    def _setup_boundaries(self, width: float = 25.0):
        """Absorbing boundary layer."""
        dx = np.minimum(self.X + self.size/2, self.size/2 - self.X)
        dy = np.minimum(self.Y + self.size/2, self.size/2 - self.Y)
        d = np.minimum(dx, dy)
        self.boundary_damp = np.clip(d / width, 0, 1)
    
    def _laplacian(self, f: np.ndarray) -> np.ndarray:
        """Discrete Laplacian."""
        lap = np.zeros_like(f)
        lap[1:-1, 1:-1] = (
            f[2:, 1:-1] + f[:-2, 1:-1] +
            f[1:-1, 2:] + f[1:-1, :-2] -
            4*f[1:-1, 1:-1]
        ) / self.dx**2
        return lap
    
    def add_matter(self, x: float, y: float, amplitude: float, 
                   radius: float, name: Optional[str] = None) -> Matter:
        """
        Add matter to the universe.
        
        This is the ONLY way to create mass. Matter is a bound E-structure,
        not a parameter or external field.
        
        Args:
            x, y: Position
            amplitude: E-field amplitude (determines "mass")
            radius: Size of the matter structure
            name: Optional identifier
            
        Returns:
            Matter object for tracking
        """
        if name is None:
            name = f"matter_{len(self.matter)}"
        
        if name in self.matter:
            raise ValueError(f"Matter '{name}' already exists")
        
        # Create the E-structure
        R = np.sqrt((self.X - x)**2 + (self.Y - y)**2)
        profile = amplitude * np.exp(-R**2 / (2 * radius**2))
        
        # Add to substrate
        self.E += profile
        self.E_prev += profile  # Start at rest
        
        # Register
        matter = Matter(name=name, x0=x, y0=y, amplitude=amplitude, radius=radius)
        self.matter[name] = matter
        
        return matter
    
    def excite_matter(self, name: str, velocity: Tuple[float, float]):
        """
        Give matter a velocity kick (excite it).
        
        This causes the matter to oscillate and radiate.
        This is how light is CREATED - through oscillating matter.
        """
        if name not in self.matter:
            raise ValueError(f"Unknown matter: {name}")
        
        m = self.matter[name]
        vx, vy = velocity
        
        # Find the matter's E-structure
        R = np.sqrt((self.X - m.current_x)**2 + (self.Y - m.current_y)**2)
        mask = R < m.radius * 3
        
        # Apply velocity by shifting E_prev
        # v = (E - E_prev) / dt => E_prev = E - v*dt
        self.E_prev[mask] = self.E[mask] - (vx + vy) * self.dt * self.E[mask]
    
    def evolve(self, steps: int = 1):
        """
        Evolve the universe using ONLY GOV-01 + GOV-02.
        
        This is the ONLY way to advance time. No shortcuts.
        
        GOV-01: ∂²E/∂t² = c²∇²E − χ²E
        GOV-02: ∂²χ/∂t² = c²∇²χ − κ(E² − E₀²)
        """
        for _ in range(steps):
            # GOV-01: E dynamics
            lap_E = self._laplacian(self.E)
            E_accel = self.c**2 * lap_E - self.chi**2 * self.E
            E_new = 2*self.E - self.E_prev + self.dt**2 * E_accel
            
            # GOV-02: χ dynamics (χ is NEVER static!)
            lap_chi = self._laplacian(self.chi)
            chi_accel = self.c**2 * lap_chi - self.kappa * self.E**2
            chi_new = 2*self.chi - self.chi_prev + self.dt**2 * chi_accel
            
            # Keep χ positive
            chi_new = np.maximum(chi_new, 0.01 * self.chi_0)
            
            # Boundary damping
            E_new = E_new * self.boundary_damp + self.E * (1 - self.boundary_damp) * 0.99
            chi_new = chi_new * self.boundary_damp + self.chi_0 * (1 - self.boundary_damp)
            
            # Update fields
            self.E_prev, self.E = self.E, E_new
            self.chi_prev, self.chi = self.chi, chi_new
            
            # Update time
            self.time += self.dt
            self.step_count += 1
            
            # Update matter positions (track where E² is concentrated)
            self._update_matter_positions()
            
            # Record observations (passive)
            self._record_observations()
    
    def _update_matter_positions(self):
        """Track where each piece of matter has moved."""
        for name, m in self.matter.items():
            R = np.sqrt((self.X - m.current_x)**2 + (self.Y - m.current_y)**2)
            mask = R < m.radius * 5
            E_sq = self.E**2 * mask
            total = E_sq.sum() + 1e-10
            m.current_x = np.sum(self.X * E_sq) / total
            m.current_y = np.sum(self.Y * E_sq) / total
    
    def add_observer(self, x: float, y: float) -> int:
        """
        Add a passive observer at (x, y).
        
        Observers measure E² at their location without affecting the simulation.
        Returns observer ID.
        """
        observer_id = len(self._observers)
        self._observers.append((x, y, []))
        return observer_id
    
    def _record_observations(self):
        """Record E² at all observer locations."""
        for i, (x, y, history) in enumerate(self._observers):
            ix = int((x + self.size/2) / self.dx)
            iy = int((y + self.size/2) / self.dx)
            ix = np.clip(ix, 0, self.Nx-1)
            iy = np.clip(iy, 0, self.Ny-1)
            history.append(self.E[ix, iy]**2)
    
    def get_observations(self, observer_id: int) -> np.ndarray:
        """Get observation history for an observer."""
        return np.array(self._observers[observer_id][2])
    
    def get_chi_at(self, x: float, y: float) -> float:
        """Get χ value at position."""
        ix = int((x + self.size/2) / self.dx)
        iy = int((y + self.size/2) / self.dx)
        ix = np.clip(ix, 0, self.Nx-1)
        iy = np.clip(iy, 0, self.Ny-1)
        return self.chi[ix, iy]
    
    def measure_radiation_ring(self, x: float, y: float, radius: float, 
                                width: float = 3.0) -> float:
        """Measure total E² in a ring around (x, y)."""
        R = np.sqrt((self.X - x)**2 + (self.Y - y)**2)
        mask = np.abs(R - radius) < width
        return np.sum(self.E**2 * mask)
    
    # =========================================================================
    # FORBIDDEN OPERATIONS - these will raise errors
    # =========================================================================
    
    def set_chi(self, *args, **kwargs):
        """FORBIDDEN: χ must evolve via GOV-02, never be set directly."""
        raise RuntimeError(
            "FORBIDDEN: Cannot set χ directly!\n"
            "χ must evolve dynamically via GOV-02.\n"
            "There is no 'static χ field' in proper LFM."
        )
    
    def inject_wave(self, *args, **kwargs):
        """FORBIDDEN: Waves must emerge from oscillating matter."""
        raise RuntimeError(
            "FORBIDDEN: Cannot inject waves!\n"
            "Light emerges from oscillating matter, not external injection.\n"
            "Use add_matter() + excite_matter() instead."
        )
    
    def add_test_wave(self, *args, **kwargs):
        """FORBIDDEN: No such thing as a 'test wave' in LFM."""
        raise RuntimeError(
            "FORBIDDEN: No 'test waves' in LFM!\n"
            "Light IS the substrate oscillating.\n"
            "Create matter, excite it, and radiation will emerge."
        )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LFM Substrate Framework - Example")
    print("=" * 70)
    print()
    
    # Create universe
    universe = LFMUniverse(size=200, chi_0=1.0, kappa=2.0, resolution=300)
    print(f"Created universe: {universe.size}×{universe.size}, χ₀={universe.chi_0}")
    
    # Add matter (this is the ONLY way to create mass)
    star = universe.add_matter(x=0, y=0, amplitude=4.0, radius=8.0, name="star")
    atom = universe.add_matter(x=-50, y=40, amplitude=1.5, radius=3.0, name="atom")
    print(f"Added matter: {list(universe.matter.keys())}")
    
    # Let χ settle
    print("Letting χ settle...")
    universe.evolve(steps=400)
    
    chi_at_star = universe.get_chi_at(0, 0)
    print(f"χ at star: {chi_at_star:.4f} (reduction: {(1-chi_at_star/universe.chi_0)*100:.1f}%)")
    
    # Add observer to detect radiation
    obs_id = universe.add_observer(x=-20, y=40)
    
    # Excite the atom (this creates light!)
    print("Exciting atom (creating light)...")
    universe.excite_matter("atom", velocity=(0.1, 0.1))
    
    # Evolve and watch radiation
    print("Evolving...")
    universe.evolve(steps=2000)
    
    # Check radiation
    observations = universe.get_observations(obs_id)
    peak_radiation = np.max(observations)
    print(f"Peak radiation detected: {peak_radiation:.6f}")
    
    # Check atom moved toward star (gravity!)
    print(f"Atom moved: ({atom.x0:.1f}, {atom.y0:.1f}) → ({atom.current_x:.1f}, {atom.current_y:.1f})")
    
    # Try a forbidden operation (will fail)
    print()
    print("Attempting forbidden operation...")
    try:
        universe.inject_wave(k=0.5, x0=-100)
    except RuntimeError as e:
        print(f"Caught: {e}")
    
    print()
    print("Done!")
