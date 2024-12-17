#See here for further details: https://qiskit-community.github.io/qiskit-dynamics/tutorials/dynamics_backend


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




## Requirements
#qiskit==1.2.4
#qiskit-aer==0.15.1
#qiskit-algorithms==0.3.0
#qiskit-dynamics==0.5.1
#qiskit-experiments==0.6.1
#qiskit-ibm-experiment==0.4.8
#qiskit-ibm-runtime==0.29.0
#qiskit-machine-learning==0.7.2
#qiskit-nature==0.7.2