# Molecular_Dynamics

## Simulated annealing re-implementation

This repository keeps a re-implementation of the simulated annealing Monte Carlo algorithm for protein folding in the HP model
in Python (3.x), using NumPy and Numba libraries.

Hydrophobic-polar protein folding (HP) model is used in the study of the general principles of protein folding.
The idea of the HP model is based on the observation that a key role in the process of folding
has the hydrophobic effect - tendency of hydrophobic amino acids to aggregate and *'hide'* from the water molecules.
Amino acids are over the alphabet **{H,P}**, where H is hydrophobic and P polar amino acid
and they are located on the square lattice.

Metropolis–Hastings algorithm is a **Markov chain Monte Carlo (MCMC)** method that allows 
sampling the set of possible configurations of protein, according to any probability distribution (here the Boltzmann distribution). 

The algorithm generates a Markov chain in which each state x^{t+1} depends only on the previous state x^t. The algorithm uses a proposal density Q(x'; x^t ), which depends on the current state x^t, to generate a new proposed sample x'. This proposal is "accepted" as the next value (x^{t+1}=x') if \alpha drawn from U(0,1) satisfies.

For more details see [1].


## 1. Simulated annealing

Usage:
```bash
$ python simulated_annealing.py
```
Output is the trajectory in the pdb format.

## 2. Visualization

You can visualizate trajectory in the pdb format using PyMOL:
```bash
$ pymol trajectory.pdb
```
and then in PyMOL console:

```bash
PyMOL> run show
```

---

[1] Dill, KA, Bromberg, S, Yue, K, Fiebig, KM, Yee, DP, Thomas, PD, Chan, HS (1995). Principles of protein folding--a perspective from simple exact models. Protein Sci., 4, 4:561-602.

