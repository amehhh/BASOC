
## Balanced Active Subspaces for Optimal Control (BASOC)

This repository contains the implementation of **Balanced Active Subspaces for Optimal Control (BASOC)**, a data-driven model reduction framework for optimal feedback control of high-dimensional nonlinear dynamical systems.

BASOC leverages trajectory optimization and the Pontryagin minimum principle to construct state and gradient covariance matrices from locally optimal trajectories. Balancing these matrices yields reduced coordinates for approximating the value function, constructing reduced-order models, and developing surrogate feedback control laws.

### Paper

**Emmanuel S. Ameh and Samuel E. Otto**, *Balanced Active Subspaces for Optimal Control via Trajectory Optimization*.

**arXiv preprint:** [arXiv:2610.36342](https://arxiv.org/abs/2610.36342)

### Citation

If you use this code or find our work useful, please cite:

```bibtex
@misc{ameh2026balanced,
  title={Balanced Active Subspaces for Optimal Control via Trajectory Optimization},
  author={Ameh, Emmanuel S. and Otto, Samuel E.},
  year={2026},
  eprint={2610.36342},
  archivePrefix={arXiv},
  primaryClass={math.OC},
  doi={10.48550/arXiv.2610.36342}
}
```

