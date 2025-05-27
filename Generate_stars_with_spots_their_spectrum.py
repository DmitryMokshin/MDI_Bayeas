import time

import numpy as np
import h5py
from tqdm import tqdm
import Doppler_Imaging_Star as DIS
import healpy as hp
import pickle

from multiprocessing import Pool
from joblib import Parallel, delayed
from mpi4py import MPI


def compute_observe_spectrum_different_phase(star, temperature_grid, result_wavelength_grid=None):
    num_of_angles = np.random.randint(low=6, high=80, size=1)[0]

    rotation_velocity = np.random.uniform(low=10, high=150)

    min_inclination = 10.0
    max_inclination = 85.0
    cos_inclination = np.random.uniform(low=np.cos(max_inclination * np.pi / 180.0),
                                        high=np.cos(min_inclination * np.pi / 180.0))
    sin_inclination = np.sqrt(1.0 - cos_inclination ** 2)
    phases = np.random.uniform(low=0, high=2.0 * np.pi, size=num_of_angles)
    phases = np.sort(phases)

    los = np.zeros((num_of_angles, 2))
    los[:, 0] = np.arcsin(sin_inclination)
    los[:, 1] = phases

    stokes_i = star.compute_stellar_spectrum(temperature_grid, los, omega=rotation_velocity,
                                             reinterpolate_lambda=result_wavelength_grid)

    result_data = {'Temperature_grid': temperature_grid, 'Stokes_I': stokes_i, 'Phases': phases,
                   'Rot_vel': rotation_velocity, 'sin_i': sin_inclination}

    # return temperature_grid, stokes_i, phases, rotation_velocity, sin_inclination
    return result_data


def unpack_archive(archive_data):
    num_of_star = len(archive_data)

    list_temperature_grid = list(np.zeros(num_of_star))
    list_stokes_i = list(np.zeros(num_of_star))
    list_phases = list(np.zeros(num_of_star))
    list_sin = list(np.zeros(num_of_star))
    list_rotation_velocity = list(np.zeros(num_of_star))

    for i in range(num_of_star):
        list_temperature_grid[i] = archive_data[i]['Temperature_grid']
        list_stokes_i[i] = archive_data[i]['Stokes_I']
        list_phases[i] = archive_data[i]['Phases']
        list_sin[i] = archive_data[i]['sin_i']
        list_rotation_velocity[i] = archive_data[i]['Rot_vel']

    return list_temperature_grid, list_stokes_i, list_phases, list_sin, list_rotation_velocity


def master_work(file_name_result_data, file_name_surfaces_archive, num_of_side=16, num_iter_for_one_task=100):
    """
    This function manages task distribution across worker processes and handles input data loading.

    :param file_name_result_data: str
        name and path to file with result this program
    :param file_name_surfaces_archive: str
        name and path to file with temperature surface distribution computed early
    :param num_of_side: integer

    :return: None

    """

    data_surface_temperature_distribution = h5py.File(file_name_surfaces_archive, 'r')['T']

    num_of_star, num_of_pixel = data_surface_temperature_distribution.shape

    print(num_of_star, num_of_pixel)

    list_temperature_grid = list(np.zeros(num_of_star))
    list_stokes_i = list(np.zeros(num_of_star))
    list_phases = list(np.zeros(num_of_star))
    list_sin = list(np.zeros(num_of_star))
    list_rotation_velocity = list(np.zeros(num_of_star))

    star = DIS.DopplerImaging(num_of_side)

    for j in range(num_of_star // num_iter_for_one_task):
        data_surface_temperature_distribution_for_task = data_surface_temperature_distribution[
                                                         j * num_iter_for_one_task: (j + 1) * num_iter_for_one_task, :]

        result = Parallel(n_jobs=16)(
            delayed(compute_observe_spectrum_different_phase)(star,
                                                              data_surface_temperature_distribution_for_task[i, :]) for i in
            range(num_iter_for_one_task))

        (list_temperature_grid[j * num_iter_for_one_task: (j + 1) * num_iter_for_one_task],
         list_stokes_i[j * num_iter_for_one_task: (j + 1) * num_iter_for_one_task],
         list_phases[j * num_iter_for_one_task: (j + 1) * num_iter_for_one_task],
         list_sin[j * num_iter_for_one_task: (j + 1) * num_iter_for_one_task],
         list_rotation_velocity[j * num_iter_for_one_task: (j + 1) * num_iter_for_one_task]) = unpack_archive(result)

    list_wavelength = list(star.wavelength_model)

    print(list_sin)

    with open(file_name_result_data + '_temperature.pkl', 'wb') as file_result:
        pickle.dump(list_temperature_grid, file_result)

    with open(file_name_result_data + '_stokes_i.pkl', 'wb') as file_result:
        pickle.dump(list_stokes_i, file_result)

    with open(file_name_result_data + '_phases.pkl', 'wb') as file_result:
        pickle.dump(list_phases, file_result)

    with open(file_name_result_data + '_sini.pkl', 'wb') as file_result:
        pickle.dump(list_sin, file_result)

    with open(file_name_result_data + '_rot_vel.pkl', 'wb') as file_result:
        pickle.dump(list_rotation_velocity, file_result)

    with open(file_name_result_data + '_wavelength.pkl', 'wb') as file_result:
        pickle.dump(list_wavelength, file_result)


def slave_work(data_for_analyse):
    """
    This function computes spectra based on the specified surface properties and stellar rotation phases.

    :param num_of_side: integer
        num of side for healpix modeling
    :param out_lambda: float numpy array
        Spectrum wavelengths for computing
    :return:
        data for one star
    """

    return None


if __name__ == '__main__':
    num_of_side = 16

    start_time = time.time()

    master_work('test_file', './Star_maps/num_of_side_16_num_of_star_10/stars_T_spots.h5', num_of_side=num_of_side)

    print('Time compute:', time.time() - start_time)
