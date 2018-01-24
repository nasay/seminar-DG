## Discontinuous Galerkin as communication avoiding numerical scheme
Source code, presentation slides and the final article for the seminar "Next generation HPC" in TUM.

Supervisors: Dr. rer. nat. Vasco Varduhn and M.Sc. Carsten Uphoff.

### Make
```
module load mkl
mpiCC main.cpp -o binary -O3 -std=c++11  -I${MKLROOT}/include  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl
```

### Run
```
mpiexec -n $p binary <number of timesteps> <timestep> <number of elements> <polynomial basis degree (<=3)>
```
