from pkg_req import *
import numpy as np

def calculate_fidelity(counts, ideal_density_matrix):

#Normalize for probabilities
    total_shots = sum(counts.values())
    probabilities = {state: count / total_shots for state, count in counts.items()}

#Construct measured density matrix: Note needs to remain hardcoded for 2 qubit system for current backend config
    num_states = 4 
    measured_density_matrix = np.zeros((num_states, num_states), dtype=complex)
    states = ['00', '01', '10', '11']  
    for i, state in enumerate(states):
        prob = probabilities.get(state, 0)  # Prob for each state
        # |state><state|
        outer_product = np.outer(np.array([1 if j == i else 0 for j in range(num_states)]), 
                                 np.array([1 if j == i else 0 for j in range(num_states)]))
        measured_density_matrix += prob * outer_product

    #Convert to DensityMatrix object
    rho_measured = DensityMatrix(measured_density_matrix)


    fidelity = state_fidelity(rho_measured, ideal_density_matrix)
    return fidelity
