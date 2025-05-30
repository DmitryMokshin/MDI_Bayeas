import time
from tqdm import tqdm

import numpy as np

import Doppler_Imaging_Star as DIS

import h5py
import pickle

from joblib import Parallel, delayed


def compute_observe_spectrum_different_phase(star, temperature_grid, result_wavelength_grid=None):
    num_of_angles = np.random.randint(low=6, high=30, size=1)[0]

    rotation_velocity = np.random.uniform(low=10, high=300)

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


def master_work(file_name_result_data, file_name_surfaces_archive, num_of_side=16, num_iter_for_one_task=100,
                result_lambda_grid=None):
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

    for j in tqdm(range(num_of_star // num_iter_for_one_task)):
        data_surface_temperature_distribution_for_task = data_surface_temperature_distribution[
                                                         j * num_iter_for_one_task: (j + 1) * num_iter_for_one_task, :]

        result = Parallel(n_jobs=16)(
            delayed(compute_observe_spectrum_different_phase)(star,
                                                              data_surface_temperature_distribution_for_task[i, :],
                                                              result_wavelength_grid=result_lambda_grid) for
            i in
            range(num_iter_for_one_task))

        (list_temperature_grid[j * num_iter_for_one_task: (j + 1) * num_iter_for_one_task],
         list_stokes_i[j * num_iter_for_one_task: (j + 1) * num_iter_for_one_task],
         list_phases[j * num_iter_for_one_task: (j + 1) * num_iter_for_one_task],
         list_sin[j * num_iter_for_one_task: (j + 1) * num_iter_for_one_task],
         list_rotation_velocity[j * num_iter_for_one_task: (j + 1) * num_iter_for_one_task]) = unpack_archive(result)

    if result_lambda_grid is not None:
        list_wavelength = list(result_lambda_grid)
    else:
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


if __name__ == '__main__':
    num_of_side = 16

    start_time = time.time()

    out_lambda = np.array([5961.8, 5961.92, 5962.04, 5962.16, 5962.28, 5962.4, 5962.52, 5962.64, 5962.76,
                           5962.88, 5963., 5963.12, 5963.24, 5963.36, 5963.48, 5963.6, 5963.72, 5963.84,
                           5963.96, 5964.08, 5964.2, 5964.32, 5964.44, 5964.56, 5964.68, 5964.8, 5964.92,
                           5965.04, 5965.16, 5965.28, 5965.4, 5965.52, 5965.64, 5965.76, 5965.88, 5966.,
                           5966.12, 5966.24, 5966.36, 5966.48, 5966.6, 5966.72, 5966.84, 5966.96, 5967.08,
                           5967.2, 5967.32, 5967.44, 5967.56, 5967.68, 5967.8, 5967.92, 5968.04, 5968.16,
                           5968.28, 5968.4, 5968.52, 5968.64, 5968.76, 5968.88, 5969., 5969.12, 5969.24,
                           5969.36, 5969.48, 5969.6, 5969.72, 6052.04, 6052.16, 6052.28, 6052.4, 6052.52, 6052.64,
                           6052.76, 6052.88, 6053., 6053.12, 6053.24, 6053.36, 6053.48, 6053.6, 6053.72, 6053.84,
                           6053.96, 6054.08, 6054.2, 6054.32, 6054.44, 6054.56, 6054.68, 6054.8, 6054.92, 6055.04,
                           6055.16, 6055.28, 6055.4, 6055.52, 6055.64, 6055.76, 6055.88, 6056., 6056.12, 6056.24
                              , 6056.36, 6056.48, 6056.6, 6056.72, 6056.84, 6056.96, 6057.08, 6057.2, 6057.32
                              , 6057.44, 6057.56, 6057.68, 6057.8, 6057.92, 6058.04, 6058.16, 6058.28, 6058.4
                              , 6058.52, 6058.64, 6058.76, 6058.88, 6059., 6059.12, 6059.24, 6059.36, 6059.48
                              , 6059.6, 6059.72, 6059.84])

    master_work('test_file', './Star_maps/num_of_side_16_num_of_star_1e5/stars_T_spots.h5', num_of_side=num_of_side, result_lambda_grid=out_lambda)

    print('Time compute:', time.time() - start_time)
