## Discontinuous Galerkin as communication avoiding numerical scheme
Source code, presentation slides and the final article for the seminar "Next generation HPC" in TUM.

Supervisors: Dr. rer. nat. Vasco Varduhn and M.Sc. Carsten Uphoff.

### Abstract
In this paper, we consider Discontinuous Galerkin Finite Element Method for solving partial differential equations with focus on the amount of required communication between processes. The inter-process communication is one of the bottlenecks in parallel computing, therefore, to avoid the communication a domain must be divided on as independent sub-domains as possible. We discuss the finite difference, the finite volume, the finite element, and the discontinious Galerkin finite elements methods, as well as their advantages and disadvantages with regard to accuracy, ability to handle complex geometries and parallelization efficiency. We show that Discontinuous Galerkin finite elements method is spatially compact and does not require setting up a global system of equations, which makes it very attractive for large-scale parallelizations. We present the derivation of the method for one-dimensional hyperbolic conservation law. We experimentally show, that for continuous case the method’s order corresponds to the theoretically predicted. The efficiency of the method’s parallelization is investigated using weak and strong scaling, with up to 20 nodes of the second phase on SuperMUC cluster. For 20 nodes, we obtained parallel efficiency of more than 80% for strong scaling and more than 95% for weak scaling.

### Make
```
module load mkl
mpiCC main.cpp -o binary -O3 -std=c++11  -I${MKLROOT}/include  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl
```

### Run
```
mpiexec -n $p binary <number of timesteps> <timestep> <number of elements> <polynomial basis degree (<=3)>
```
