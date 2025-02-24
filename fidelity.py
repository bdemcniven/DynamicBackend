from pkg_req import *
import numpy as np

def fidelity_x0(x0, counts, ideal_density_matrix):
    x0 = [100,1,1,1]
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

def calculate_fidelity( counts, ideal_density_matrix):
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

def fidelity_circ_v1(pulse_params):
    test=pulse_circ(*pulse_params)
    required_keys = ['00', '01', '10', '11']
    for key in required_keys:
        test.setdefault(key, 0)
    counts = {'00': int(test['00']), '01': int(test['01']), '10': int(test['10']), '11': int(test['11'])} 
    total_shots = sum(counts.values())
    probabilities = {state: count / total_shots for state, count in counts.items()}
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
    rho_measured = DensityMatrix(measured_density_matrix)#measured_density_matrix#DensityMatrix(measured_density_matrix)

    fidelity = state_fidelity(rho_measured, des_state,validate=False)
    infid=1-np.abs(fidelity) 
    infid_dat.append(infid)
    print('infidelity: '+str(infid) )
    return infid 

def calculate_fidelity_circ_2s(pulse_params):
    test=pulse_circ(*pulse_params)
    required_keys = ['0', '1']
    for key in required_keys:
        test.setdefault(key, 0)
    counts = {'0': int(test['0']), '1': int(test['1'])} 
    total_shots = sum(counts.values())
    probabilities = {state: count / total_shots for state, count in counts.items()}
    num_states = 2 
    measured_density_matrix = np.zeros((num_states, num_states), dtype=complex)
    states = ['0','1']  
    for i, state in enumerate(states):
        prob = probabilities.get(state, 0)  # Prob for each state
        # |state><state|
        outer_product = np.outer(np.array([1 if j == i else 0 for j in range(num_states)]), 
                                 np.array([1 if j == i else 0 for j in range(num_states)]))
        measured_density_matrix += prob * outer_product
    #Convert to DensityMatrix object
    rho_measured = DensityMatrix(measured_density_matrix)#measured_density_matrix#DensityMatrix(measured_density_matrix)
    fidelity = state_fidelity(rho_measured, des_state,validate=False)
    infid=1-np.abs(fidelity) 
    #infid_dat.append(infid)
    print('infidelity: '+str(infid))
    return infid 

def calculate_fidelity_circ_4s(pulse_params):
    test=pulse_circ_2q(*pulse_params)
    required_keys = ['00', '01', '10', '11']
    for key in required_keys:
        test.setdefault(key, 0)
    counts = {'00': int(test['00']), '01': int(test['01']), '10': int(test['10']), '11': int(test['11'])} 
    total_shots = sum(counts.values())
    probabilities = {state: count / total_shots for state, count in counts.items()}
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
    rho_measured = DensityMatrix(measured_density_matrix)#measured_density_matrix#DensityMatrix(measured_density_matrix)
    fidelity = state_fidelity(rho_measured, des_state,validate=False)
    infid=1-np.abs(fidelity) 
    #infid_dat.append(infid)
    print('infidelity: '+str(infid) )
    return infid