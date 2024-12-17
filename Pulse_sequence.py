from qiskit import transpile, pulse
from qiskit_dynamics import DynamicsBackend, Solver
from qiskit import QuantumCircuit, QuantumRegister, transpile, ClassicalRegister, result
from qiskit_aer import QasmSimulator
import qiskit_aer
import qiskit_aer.primitives
from qiskit_aer import Aer
from qiskit_aer.primitives import SamplerV2
from qiskit_aer.primitives import EstimatorV2
from qiskit.quantum_info import Statevector
import numpy as np
from qiskit_ibm_runtime.fake_provider import FakeValenciaV2
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_state_qsphere, plot_bloch_multivector
from qiskit.visualization import plot_bloch_vector
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient, ParamShiftSamplerGradient
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit_machine_learning.utils.loss_functions import CrossEntropyLoss
from qiskit.quantum_info import state_fidelity
import matplotlib.pyplot as plt
from qiskit_algorithms.optimizers import ADAM, SPSA, NFT
import random
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# params for model hamiltonian

dim = 3

v0 = 4.86e9
anharm0 = -0.32e9
r0 = 0.22e9

v1 = 4.97e9
anharm1 = -0.32e9
r1 = 0.26e9

J = 0.002e9

a = np.diag(np.sqrt(np.arange(1, dim)), 1)
adag = np.diag(np.sqrt(np.arange(1, dim)), -1)
N = np.diag(np.arange(dim))

ident = np.eye(dim, dtype=complex)
full_ident = np.eye(dim**2, dtype=complex)

N0 = np.kron(ident, N)
N1 = np.kron(N, ident)

a0 = np.kron(ident, a)
a1 = np.kron(a, ident)

a0dag = np.kron(ident, adag)
a1dag = np.kron(adag, ident)


static_ham0 = 2 * np.pi * v0 * N0 + np.pi * anharm0 * N0 * (N0 - full_ident)
static_ham1 = 2 * np.pi * v1 * N1 + np.pi * anharm1 * N1 * (N1 - full_ident)

static_ham_full = (
    static_ham0 + static_ham1 + 2 * np.pi * J * ((a0 + a0dag) @ (a1 + a1dag))
)

#look at driving operator

drive_op0 = 2 * np.pi * r0 * (a0 + a0dag)
drive_op1 = 2 * np.pi * r1 * (a1 + a1dag)


# build solver
dt = 1 / 4.5e9

#Look at this in qiskit doc

solver = Solver(
    static_hamiltonian=static_ham_full,
    hamiltonian_operators=[drive_op0, drive_op1, drive_op0, drive_op1],
    rotating_frame=static_ham_full,
    hamiltonian_channels=["d0", "d1", "u0", "u1"],
    channel_carrier_freqs={"d0": v0, "d1": v1, "u0": v1, "u1": v0},
    dt=dt,
    array_library="jax",
)

solver_options = {"method": "jax_odeint", "atol": 1e-6, "rtol": 1e-8, "hmax": dt}

backend = DynamicsBackend(
    solver=solver,
    subsystem_dims=[dim, dim],  # for computing measurement data
    solver_options=solver_options,  # to be used every time run is called
)

sigma = 128
num_samples = 20  #Look at what this is for

schedules = []
dat=[]
for i in range(0,1,1):
    for amp in np.linspace(0.5, 1, 2):
        print(amp)
        #Play w/ amplitude
        gauss = pulse.library.Gaussian(num_samples, amp, sigma, angle=np.pi, name="Parametric Gauss")
    
        with pulse.build() as schedule:
            with pulse.align_sequential():
                pulse.play(gauss, pulse.DriveChannel(0))
                #pulse.shift_phase(0.3, pulse.DriveChannel(1))
                #pulse.shift_frequency(0.1, pulse.DriveChannel(0))
                pulse.play(gauss, pulse.DriveChannel(0))
                pulse.acquire(duration=1, qubit_or_channel=0, register=pulse.MemorySlot(0))
                #pulse.measure_channel(0)
    
        schedules.append(schedule)
    
    job = backend.run(schedules, shots=100)
    result = job.result()
    dat.append((i,result.get_counts(0)['0'],result.get_counts(0)['1']))
    #print(result.get_counts(0))

schedules[0].show('mpl')