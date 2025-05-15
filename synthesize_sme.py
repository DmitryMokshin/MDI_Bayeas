import numpy as np
from model_spectrum import generate_one_spec_pysme

if __name__ == '__main__':
    range_wavelength = [[5900, 6100]]

    n_mu = 15
    n_v = 160

    T_array = np.concatenate([np.arange(3000, 4000, 100.0), np.arange(4000, 6000, 250.0), np.arange(6000, 8000, 500.0),
                              np.arange(8000, 14000, 1000.0)])

    n_T = len(T_array)

    mu_array = np.linspace(0.02, 1.0, n_mu)
    vel_array = np.linspace(-300.0, 300.0, n_v)
