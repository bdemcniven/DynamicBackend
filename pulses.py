from pkg_req import * 

#num_samples: The number of samples in the Gaussian pulse.
#amp: The amplitude of the Gaussian pulse.
#sigma: The standard deviation (width) of the Gaussian pulse.

def gauss(num_samples_g, amp_g, sigma_g, angle_g):
    gauss = pulse.library.Gaussian(duration=num_samples_g, amp=amp_g, sigma=sigma_g, angle=angle_g)
    return gauss


#https://docs.quantum.ibm.com/api/qiskit/qiskit.pulse.library.GaussianSquare
def gauss_sq(num_samples_sq, amp_sq, sigma_sq, angle_sq, width_sq):
    gauss_square = qiskit.pulse.library.GaussianSquare(duration=num_samples_sq, amp=amp_sq, sigma=sigma_sq, angle=angle_sq, width=width_sq, limit_amplitude=None)
    return gauss_square

#https://docs.quantum.ibm.com/api/qiskit/qiskit.pulse.library.gaussian_square_echo
def gauss_CR(dur_CR, amp_CR, sigma_CR, angle_CR, width_CR):
    # Correct the order: keyword arguments should follow positional ones
    gauss_sq_echo = qiskit.pulse.library.symbolic_pulses.gaussian_square_echo(
        duration=dur_CR, 
        amp=amp_CR, 
        sigma=sigma_CR, 
        width=width_CR, 
        angle=angle_CR, 
        active_amp=0.0, 
        active_angle=0.0, 
        risefall_sigma_ratio=None, 
        limit_amplitude=None
    )
    return gauss_sq_echo
