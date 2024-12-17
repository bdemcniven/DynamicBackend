from pkg_req import * 
from construction import *
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
#Solver is a qiskit class for simulating either Hamiltonian or Lindblad dynamics
#See here: https://qiskit-community.github.io/qiskit-dynamics/stubs/qiskit_dynamics.solvers.Solver.html#qiskit_dynamics.solvers.Solver
# rotating_frame defines the frame operator, which is a set of vectors in a Hilbert space providing a way to decompose vectors. Unlike a basis, the vectors in a frame may not be linearly independent and they can be overcomplete. The frame operator provides a way to measure the extent of this overcompleteness and how vectors in the space can be reconstructed from the frame elements
# rotating_frame therefore transforms the Hamiltonian to make the problem more tractable (i.e., by removing fast oscillations, time dependence, etc.)

def dynam_solver(static_ham_full, drive_op0, drive_op1, v0, v1, dt):
    solver = Solver(
        static_hamiltonian=static_ham_full,  # Full Hamiltonian
        hamiltonian_operators=[drive_op0, drive_op1, drive_op0, drive_op1],  # Interaction operators, repeated twice
        rotating_frame=static_ham_full,  # Defining reference frame for H
        hamiltonian_channels=["d0", "d1", "u0", "u1"],  # Channels for the operators
        channel_carrier_freqs={"d0": v0, "d1": v1, "u0": v1, "u1": v0},  # Frequencies applied to operators
        dt=dt,  # Time step
        array_library="jax",  # Numerical backend (e.g., JAX)
    )
    #options
    solver_options = {
        "method": "jax_odeint",  # Solver method
        "atol": 1e-6,            # Absolute tolerance
        "rtol": 1e-8,            # Relative tolerance
        "hmax": dt               # Maximum step size
    }

    return solver, solver_options