import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import astropy.constants as const
import zarr

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


def doppler_effect(lambda_in, velocity_grid, velocity_shift, flux_orig):

    velocity_per_pxl = np.mean(np.diff(velocity_grid))

    freq_grid = np.fft.fftfreq(len(lambda_in))

    kernel = np.exp(-2 * 1j * np.pi * freq_grid[None, :] * velocity_shift[:, None] / velocity_per_pxl)

    fi = np.fft.fft(flux_orig)

    res_flux = np.fft.ifft(fi[None, :] * kernel).real

    return res_flux


if __name__ == "__main__":
    # Если архив содержит группу (как папка)
    zarr_group = zarr.open_group('model_spec.zarr', mode='r')

    print("Ключи в группе:", list(zarr_group.keys()))

    wavelength = zarr_group['wavelength'][:]

    temp = zarr_group['T'][:]
    mu = zarr_group['mu'][:]

    print(temp)

    spec = zarr_group['spec']

    num_rad_vel, num_t, num_mu, num_inten = spec.shape

    num_t_for_test = np.where((temp > 8000) & (temp <= 13000))

    print(num_t_for_test)

    for i in num_t_for_test[0]:
        plt.plot([min(wavelength), max(wavelength)], [1.0, 1.0], label='cont')
        plt.plot(wavelength, spec[int(num_rad_vel / 2), i, 0, :], label=r'$T$ = '+ f'{temp[i]}')
        plt.legend()

    plt.show()
