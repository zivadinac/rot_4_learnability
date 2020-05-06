# rot_4_learnability
Repository for IST rotation 4 project.

Investigating "learnability" (motivated by [Clustering of Neural Activity: A Design Principle for Population Codes](https://pub.ist.ac.at/~gtkacik/Frontiers_Clusters.pdf)) of cluster code in retinal ganglion cells.
Using method from [Error-Robust Modes of the Retinal Population Code](https://pub.ist.ac.at/~gtkacik/PLOSCompBio_RobustCode.pdf) included as a TreeHMM-local submodule.


## Installation

Clone the repository (including submodule):
`git clone --recurse-submodules git@github.com:zivadinac/rot_4_learnability.git`

`cd rot_4_learnability`

`pip install -r pip_requirements.txt`

#### Building TreeHMM
Follow instructions in [TreeHMM-local/README.md](https://github.com/zivadinac/TreeHMM-local/tree/4b903f3b34c7d9dd5e714c1d80f54d3f0a772e71).

After building TreeHMM install it as a python package:
`pip install -e TreeHMM-local/`

## Usage
Script [src/simulation.py](https://github.com/zivadinac/rot_4_learnability/blob/master/src/simulation.py) simulates population of neurons for given visual stimulus (`python src/simulation.py -h` for detailed help).
Multiple simulations can be run at the same time with [run_simulations.sh](https://github.com/zivadinac/rot_4_learnability/blob/master/run_simulations.sh).

Script [src/fit_hmm.py](https://github.com/zivadinac/rot_4_learnability/blob/master/src/fit_hmm.py) fits HMM model to spike data (`python src/fit_hmm.py -h` for detailed help).
Multiple HMM trainings can be run at the same time with [run_hmms.sh](https://github.com/zivadinac/rot_4_learnability/blob/master/run_hmms.sh).
