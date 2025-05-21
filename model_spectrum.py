import numpy as np
from pysme.sme import SME_Structure as SME_Struct
from pysme.abund import Abund
from pysme.linelist.vald import ValdFile
import time
from pysme.synthesize import synthesize_spectrum

import zarr

import astropy.constants as const
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from joblib import Parallel, delayed


def make_name_of_vald_list(temperature, range_wavelength, microturbulence_velocity=4.0, log_grav=4.4):
    if str(temperature).split('.')[1] == '0' or str(temperature).split('.')[1] == '':
        str_temperature = str(temperature).split('.')[0]
    else:
        str_temperature = str(temperature).split('.')[0] + '_' + str(temperature).split('.')[1]

    if str(microturbulence_velocity).split('.')[1] == '0' or str(microturbulence_velocity).split('.')[1] == '':
        str_velocity_microturbulence = str(microturbulence_velocity).split('.')[0]
    else:
        str_velocity_microturbulence = str(microturbulence_velocity).split('.')[0] + '_' + \
                                       str(microturbulence_velocity).split('.')[1]

    if str(log_grav).split('.')[1] == '0' or str(log_grav).split('.')[1] == '':
        str_log_grav = str(log_grav).split('.')[0]
    else:
        str_log_grav = str(log_grav).split('.')[0] + '_' + str(log_grav).split('.')[1]

    str_left_wavelength = str(range_wavelength[0]).split('.')[0]
    str_right_wavelength = str(range_wavelength[1]).split('.')[0]

    return 'T' + str_temperature + 'g' + str_log_grav + 'vmicro' + str_velocity_microturbulence + 'l' + str_left_wavelength + '_' + str_right_wavelength + '.lin'


def degrade_resolution(lambda_orig, flux_orig, R, new_delta_lambda=None):
    """
    Понижает разрешение спектра с учётом доплеровского смещения от скорости.

    Параметры:
    ----------
    lambda_orig : array
        Длины волн исходного спектра (Å или нм).
    flux_orig : array
        Поток исходного спектра.
    R : float
        Целевая разрешающая способность (R = λ/Δλ).
    new_delta_lambda : float, optional
        Новый шаг по длине волны (Å/пиксель). Если None, выбирается автоматически.

    Возвращает:
    -----------
    lambda_new : array
        Новая сетка длин волн (с учётом скорости).
    flux_new : array
        Спектр с пониженным разрешением.
    """

    # 2. Размытие спектра
    delta_lambda = np.mean(np.diff(lambda_orig))
    delta_lambda_local = lambda_orig / R
    sigma_lambda = delta_lambda_local / (2 * np.sqrt(2 * np.log(2)))
    sigma_pixels = sigma_lambda / delta_lambda
    flux_smoothed = gaussian_filter1d(flux_orig, np.mean(sigma_pixels))

    # 3. Передискретизация
    if new_delta_lambda is None:
        new_delta_lambda = np.mean(lambda_orig) / R

    lambda_new = np.arange(lambda_orig[0], lambda_orig[-1], new_delta_lambda)
    interp_func = interp1d(lambda_orig, flux_smoothed, kind='cubic', bounds_error=False, fill_value='extrapolate')
    flux_new = interp_func(lambda_new)

    mean_wavelength = np.mean(lambda_new)

    grid_vel = (lambda_new - mean_wavelength) / mean_wavelength * const.c.to('km/s').value

    grid_vel_eq = np.linspace(min(grid_vel), max(grid_vel), len(lambda_new))

    return grid_vel_eq, lambda_new, flux_new


def generate_spec_pysme(temperature, mu_array, regions):
    sme = SME_Struct()

    sme.teff, sme.logg, sme.monh = temperature, 4.4, 0.0
    sme.abund = Abund.solar()

    name_linelist = make_name_of_vald_list(temperature, [4400, 4900], 4.0, 4.4)

    vald = ValdFile('Line_list/' + name_linelist)
    sme.linelist = vald

    if temperature > 8000:
        sme.atmo.source = "atlas9_vmic2.0.sav"
    else:
        sme.atmo.source = "marcs2012p_t1.0.sav"

    sme.atmo.method = "grid"
    sme.atmo.geom = "PP"

    sme.wran = regions

    sme.vrad_flag = "whole"

    synt_intensity = []

    for mu in mu_array:
        sme.mu = mu

        sme = synthesize_spectrum(sme)

        synt_intensity.append(np.array(sme.synth))

    grid_vel = (np.array(sme.wave) - np.mean(np.array(sme.wave))) / np.mean(np.array(sme.wave)) * const.c.to(
        'km/s').value

    num_of_wl = len(np.array(sme.wave))
    num_of_mu = len(mu_array)

    grid_wavelength = np.array(sme.wave)
    grid_intensity = np.array(synt_intensity).T

    mean_wavelength = np.mean(grid_wavelength)

    grid_vel_eq = np.linspace(min(grid_vel), max(grid_vel), num_of_wl)
    grid_wavelength_eq = mean_wavelength + mean_wavelength * grid_vel_eq / const.c.to('km/s').value

    for i in range(num_of_mu):
        interpol = interp1d(grid_vel, grid_intensity[:, i], kind='linear')
        grid_intensity[:, i] = interpol(grid_vel_eq)

    return grid_vel_eq, grid_wavelength_eq, grid_intensity


def doppler_effect(lambda_in, velocity_grid, velocity_shift, flux_orig):
    velocity_per_pxl = np.mean(np.diff(velocity_grid))

    freq_grid = np.fft.fftfreq(len(lambda_in))

    kernel = np.exp(-2 * 1j * np.pi * freq_grid[None, :] * velocity_shift[:, None] / velocity_per_pxl)

    fi = np.fft.fft(flux_orig)

    res_flux = np.fft.ifft(fi[None, :] * kernel).real

    return res_flux

def compute_spectrum(i, num_mu, num_wl, T_array, mu_array, range_wavelength):
    vel, wave, intensity = generate_spec_pysme(T_array[i], mu_array, range_wavelength)

    intensity_tmp = list(np.empty(num_mu))

    for j in range(num_mu):
        vel_general, wave_general, intensity_tmp[j] = degrade_resolution(wave, intensity[:, j], resol)

    intensity_general = np.array(intensity_tmp).transpose()

    result_spectrum = np.zeros((num_mu, num_wl, 3))

    for j in range(num_mu):
        result_spectrum[j, :, 0] = wave_general

        result_spectrum[j, :, 1] = vel_general

        result_spectrum[j, :, 2] = intensity_general[:, j]

    return result_spectrum


if __name__ == '__main__':
    print('Begin testing')

    num_mu = 16
    num_radial_vel = 160

    resol = 100000

    # range_wavelength = [[5985.1, 5989.0], [6000.1, 6005.0], [6022.0, 6026.0]]

    range_wavelength = [4400, 4900.0]

    T_array = np.concatenate([np.arange(3000, 4000, 100.0), np.arange(4000, 6000, 250.0), np.arange(6000, 8000, 500.0),
                              np.arange(8000, 14000, 1000.0)])

    mu_array = np.linspace(0.02, 1.0, num_mu)
    rad_vel_array = np.linspace(-300.0, 300.0, num_radial_vel)

    num_T = len(T_array)

    start_time = time.time()

    print('Start zero point')

    vel, wave, intensity = generate_spec_pysme(T_array[0], mu_array, range_wavelength)

    intensity_tmp = list(np.empty(num_mu))

    for j in range(num_mu):
        vel_general, wave_general, intensity_tmp[j] = degrade_resolution(wave, intensity[:, j], resol)

    intensity_general = np.array(intensity_tmp)

    num_wl = len(wave_general)

    result_spectrum = np.zeros((num_T, num_mu, num_wl, 3))

    for j in range(num_mu):
        result_spectrum[0, j, :, 0] = wave_general

        result_spectrum[0, j, :, 1] = vel_general

        result_spectrum[0, j, :, 2] = intensity_general[j, :]

    print('start parallel compute')

    result_spectrum[1:, :, :, :] = Parallel(n_jobs=-1)(delayed(compute_spectrum)(i, num_mu, num_wl, T_array, mu_array, range_wavelength) for i in range(1, num_T))

    end_time = time.time()

    print('Compute is complete')

    print('time compute:', end_time - start_time)

    file_out = zarr.open('model_spec.zarr', 'w')

    T_ds = file_out.create_dataset("T", shape=(num_T,), dtype=np.float32)
    v_ds = file_out.create_dataset("v", shape=(num_radial_vel,), dtype=np.float32)
    mu_ds = file_out.create_dataset("mu", shape=(num_mu,), dtype=np.float32)
    vel_axis_ds = file_out.create_dataset("velaxis", shape=(num_wl,), dtype=np.float32)
    wl_ds = file_out.create_dataset("wavelength", shape=(num_wl,), dtype=np.float32)
    spec_ds = file_out.create_dataset("spec", shape=(num_radial_vel, num_T, num_mu, num_wl), dtype=np.float32)

    T_ds[:] = T_array[:num_T]
    v_ds[:] = rad_vel_array
    mu_ds[:] = mu_array

    for i in range(num_T):

        for j in range(num_mu):
            spec_ds[:, i, j, :] = doppler_effect(result_spectrum[i, j, :, 0], result_spectrum[i, j, :, 1], rad_vel_array, result_spectrum[i, j, :, 2])

        vel_axis_ds[:] = result_spectrum[i, 0, :, 1]
        wl_ds[:] = result_spectrum[i, 0, :, 0]

    print('Compute is done and data save')
