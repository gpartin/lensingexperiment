# LFM Gravitational Lensing: True Substrate Dynamics

**Demonstrating that gravitational lensing emerges from first principles in the Lattice Field Medium framework using proper substrate dynamics.**

## Key Insight

LFM is a **substrate theory**. Everything IS the substrate—you don't put things "on top of" it. Light in LFM is not something that propagates *through* the substrate; light IS the substrate oscillating.

This experiment demonstrates gravitational lensing using the full coupled dynamics where both E and χ evolve together, and radiation emerges naturally from oscillating matter.

## The Equations (BOTH dynamical, coupled)

```
GOV-01 (E wave):  ∂²E/∂t² = c²∇²E − χ²E
GOV-02 (χ wave):  ∂²χ/∂t² = c²∇²χ − κ(E² − E₀²)
```

- **E**: Wave amplitude (energy density) at each lattice point
- **χ (chi)**: Local substrate property—evolves dynamically, not static!
- **κ**: Coupling constant between E² and χ dynamics

## How It Works

1. **Create "matter"**: Bound E-structures (like atoms) on the lattice
2. **Create a "star"**: Massive E-structure that creates a χ-well via GOV-02
3. **Excite the atom**: Kick the electron so it oscillates → naturally radiates
4. **Radiation propagates**: The substrate oscillations spread outward
5. **Measure lensing**: Does radiation bend toward the star?

## Results

### Lensing Emerges
- Radiation from oscillating matter bends **toward** massive objects
- Lensing ratio: **15× more radiation toward star than away**
- χ-well depth: 90% reduction at star center

### Frequency Dependence (Addressing the Klein-Gordon Concern)
For frequencies above the mass gap (ω ≥ χ₀):

| ω/χ₀ | Lensing Ratio |
|------|---------------|
| 1.0 | 3438 |
| 2.0 | 3339 |
| 4.0 | 3253 |

**Coefficient of variation: 2.3%** — lensing is essentially achromatic!

Power law slope: -0.04 (flat)

For real photons where ω/χ₀ ~ 10¹⁵, any dispersion is unmeasurably small.

## Quick Start

```bash
# Clone the repo
git clone https://github.com/gpartin/lensingexperiment.git
cd lensingexperiment

# Install dependencies
pip install numpy matplotlib

# Run the main lensing experiment
python lfm_substrate_lensing.py

# Run frequency dependence test
python lfm_substrate_frequency_scan.py
```

## Files

| File | Description |
|------|-------------|
| `lfm_substrate_lensing.py` | **Main experiment**: star + radiating atom, measures lensing |
| `lfm_substrate_frequency_scan.py` | **Frequency test**: confirms achromatic lensing |
| `figures/` | Output figures from experiments |
| `REDDIT_RESPONSE.md` | Response to Klein-Gordon dispersion critique |
| `dev/` | Development/auxiliary scripts (not needed to run main experiments) |

## What This Demonstrates

✅ **Gravitational lensing emerges** from coupled GOV-01 + GOV-02 dynamics  
✅ **No static fields** — both E and χ evolve dynamically  
✅ **Light is substrate oscillations** — radiation emerges naturally from matter  
✅ **Achromatic for ω >> χ₀** — only 2.3% variation over 4× frequency range  
✅ **Fully reproducible** — run the code yourself  

## The Eureka Moment

Previous tests were flawed: we computed a static χ field and injected artificial "test waves" to see how they bent. This is wrong for a substrate theory.

**Correct approach**: Let the full coupled system evolve. Matter (bound E-structures) oscillates and naturally radiates. The radiation IS the substrate oscillating. No separation between "light" and "medium."

When done correctly, lensing emerges and chromatic dispersion becomes negligible.

## Context

This experiment addresses concerns about Klein-Gordon dispersion:

> "The dispersion relation ω² = c²k² + χ² implies frequency-dependent behavior..."

The key insight: this concern applies when testing a frozen background with artificial waves. In true substrate dynamics with coupled E-χ evolution, the physics is different—and achromatic lensing emerges naturally for high-frequency radiation.

See [REDDIT_RESPONSE.md](REDDIT_RESPONSE.md) for the full discussion.

## License

MIT License

## Contact

- Repository: https://github.com/gpartin/lensingexperiment
- LFM Paper Series: https://github.com/gpartin/Papers
