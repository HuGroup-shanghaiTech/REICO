# Element-based machine learning potentials via REICO sampling


## About REICO and EMLP

REICO (Random Exploration via Imaginary Chemical Optimization) is a data sampling method to generate an unbiased and highly diverse collection of atomic configurations, each capturing unique local environments across multiple chemical elements. By relying on fully random structure generation and subsequent short optimizations, REICO ensures the training set spans a broad space of elemental interactions, promoting genuine generality in the learned machine learning potential.

EMLP is the resulting MLP from a REICO generated dataset. By focusing on element-element interactions rather than system-specific configurations, EMLP achieves both generality and reactivity without requiring targeted sampling.

This repository contains a step by step tutorial of implementation of REICO and EMLP framework developed by Changxi Yang, Chengyu Wu, Wenbo Xie, and the group of Peijun Hu at Shanghaitech University.

In principle, RECIO can be easily integrated with any state-of-the-art atomistic ML models. In our paper, we used the [Nequip code](https://github.com/mir-group/nequip) as the ML model.

This repository contains the Ag-Pd-C-H-O gerneal and reactive EMLP model, which can be used as the computational engine for calculations in heterogeneous catalysis that involves the Ag, Pd, C, H, O.

### Reference

1. Yang C, Wu C, Xie W, Xie D, Hu P. Developing General Reactive Element-Based Machine Learning Potentials as the Main Computational Engine for Heterogeneous Catalysis. ChemRxiv. 2024; doi:10.26434/chemrxiv-2024-r8l6j

---

## Environment configuration

The following programs & packages are needed for REVICO workflow:

&emsp;- python >= 3.10

&emsp;- nequip >= 0.5.6

&emsp;- numpy >= 1.25.2, numba >= 0.58.1, ase >= 3.22.1, scikit-learn >= 1.3.2, dscribe >= 2.1.0


## Elements selection & imaginary compounds generation

No prior knowledge except elements of chemical systems is needed.


#### Recommended count of imaginary compounds for several systems:

&emsp;- Ag,Pd,C,H,O: 5000

<!-- &emsp;- Ag,Pd,C,H,O: 5000 -->

<!-- &emsp;- Pd,C,H: 3000 -->

<!-- &emsp;- Pt,O: 1000 -->


#### Generate random structures

&emsp;- randomGenerator
```
    $ python randomGS.py input.json
```
For detailed parameters, see [input.json](https://github.com/HuGroup-shanghaiTech/EMLP/blob/main/REICO/input.json)

#### command & usage:

&emsp;- Count structures in extxyz file

        $ grep -c Latt STRUCTURE_FILES[.xyz]
