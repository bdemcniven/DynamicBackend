from pkg_req import * 

#num_samples: The number of samples in the Gaussian pulse.
#amp: The amplitude of the Gaussian pulse.
#sigma: The standard deviation (width) of the Gaussian pulse.

def Ry10():
        pulse.play(pulse.library.Gaussian(4386, 0.7, 1000), pulse.DriveChannel(0))
        pulse.acquire(duration=1000, qubit_or_channel=0, register=pulse.MemorySlot(0))
def Ry01():
        pulse.play(pulse.library.Gaussian(3000, 0.22,1000), pulse.DriveChannel(1))
        pulse.acquire(duration=1000, qubit_or_channel=1, register=pulse.MemorySlot(0))
def Ry11():
        pulse.play(pulse.library.Gaussian(4386, 0.7, 1000), pulse.DriveChannel(0))
        #pulse.acquire(duration=1000, qubit_or_channel=0, register=pulse.MemorySlot(0))
        pulse.play(pulse.library.Gaussian(3000, 0.22,1000), pulse.DriveChannel(1))
        #pulse.acquire(duration=1000, qubit_or_channel=1, register=pulse.MemorySlot(0))
        #pulse.measure_all()

def gauss(num_samples_g, amp_g, sigma_g, angle_g):
    gauss = pulse.library.Gaussian(duration=num_samples_g, amp=amp_g, sigma=sigma_g, angle=angle_g,limit_amplitude=False)
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
        limit_amplitude=False
    )
    return gauss_sq_echo

def x180(dur_x, amp_x, sigma_x, angle_x, beta_x):
    # Correct the order: keyword arguments should follow positional ones
    x180=pulse.library.Drag(duration=dur_x, amp=amp_x, sigma=sigma_x, angle=angle_x,beta=beta_x,limit_amplitude=False) 
    return x180