import pkg_req
import numpy as np

#Hamiltonian parameters:
#dim= specifying the number of energy levels
#v_i= qubit frequencies
#anham_i=i^th Anharmonicity. This is the coefficient that introduces an anharmonic correction to the Hamiltonian, which typically arises in systems where the energy levels are not equally spaced (such as a transmon or other qubits with anharmonic potentials)
#r_i= Rabi strengths (frequency where the probability amplitudes of two different energy levels fluctuate in an oscillating EM field)
#J=coupling strength
#a/adag=lowering/raising operators
#N_i=Number operator for i^th qubit

def hamiltonian(dim, v0, v1, anharm0, anharm1, J):
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
    return static_ham_full

def driveop(dim, r):
    a = np.diag(np.sqrt(np.arange(1, dim)), 1)
    adag = np.diag(np.sqrt(np.arange(1, dim)), -1)
    ident = np.eye(dim, dtype=complex)
    a0 = np.kron(a, ident)
    a0dag = np.kron(adag, ident)

    drive_op = 2 * np.pi * r * (a0 + a0dag)
    return drive_op

#def driveop1(dim, r1):
 #   a = np.diag(np.sqrt(np.arange(1, dim)), 1)
  #  adag = np.diag(np.sqrt(np.arange(1, dim)), -1)
   # ident = np.eye(dim, dtype=complex)
    #a1 = np.kron(a, ident)
    #a1dag = np.kron(adag, ident)
#
    #drive_op1 = 2 * np.pi * r1 * (a1 + a1dag)
    #return drive_op1

