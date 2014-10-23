# LAMMPS Ensembles 

## Authors 
Adrian W. Lange

Luke Westby

Mladen Rasic

## Description
LAMMPS Ensembles (LE) is a multidimensional replica exchange interface to LAMMPS.

LE creates multiple instances of LAMMPS classes on a set of MPI subcommunicators.
Each subcom starts up its own LAMMPS instance and runs molecular dynamics.
After so many molecular dynamics time steps, exchanges are attempted between replicas according to
Boltzmann statistics probability (Metropolis criteria).

## Capabilities
LE can run:
- Parallel tempering replica exchange.
- Replica exchange umbrella sampling for MS-EVB and some applications
  with the colvars LAMMPS library.
- Hamiltonian exchange with off-diagonal scalar for MS-EVB.
- Multidimensional REUS with any combination of the above.
- Dimensions which are linear or circular, thereby supporting many
  possible topologies (e.g. 3-D torus).
