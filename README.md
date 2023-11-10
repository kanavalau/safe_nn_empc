# Safe Explicit MPC by Training Neural Networks Through Constrained Optimization

Andrei Kanavalau, Sanjay Lall

Stanford University

## Installation

To run the code, install the required packages using Conda:

```bash
conda env create -f environment.yml
conda activate safe_nn_empc
```

The three example folders are populated with training data, trained empc policies as well as the results.

## Code overview
* run_example*.py contains the code needed to generate data, train, and test nominal and sst empc
* dyn_systems.py defines the dynamical system classes
* explicit_mpc_learning.py has the training routines for nominal and sst empc
* nn_training_utils.py contains functions and classes needed for the training
* explicit_mpc_testing.py contains functions needed for testing and comparing the policies
* compute_cis_polytope.py implements heuristics for iteratively computing control invariant polytopes
* utils.py has other useful function definitions