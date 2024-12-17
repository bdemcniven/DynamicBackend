from pkg_req import * 
from construction import *
from solver import *

#DynamicsBackend will (1) compute the state evolution for the system over time; (2) simulate measurement processes, compute observables, or handle measurements; (3) manage the individual components or subsystems of the full system (which can have different dimensionalities) for a multi qubit system

def dynam_backend(solver, solver_options, dim):
    backend = DynamicsBackend(
        solver=solver,
        subsystem_dims=[dim, dim], 
        solver_options=solver_options,  
    )
    return backend
