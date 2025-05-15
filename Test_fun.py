import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import astropy.constants as const


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


# Пример использования с ненулевой скоростью
if __name__ == "__main__":

# for j in range(n_mus):
#     fi = np.fft.fft(spec[:, j])
#     tmp = fi[None, :] * kernel
#     spec_ds[:, i, j, :] = np.fft.ifft(tmp).real

    # Исходный спектр (например, линия Hα на 6563 Å)
    lambda_orig = np.linspace(6550, 6580, 5000)
    flux_orig = np.exp(-(lambda_orig - 6563) ** 2 / (1 ** 2))  # Узкая линия

    velocity = np.linspace(-100, 100, 20)

    # Понижаем разрешение и вводим скорость +200 км/с (красное смещение)
    R = 15000
    grid_v, lambda_new, flux_new = degrade_resolution(lambda_orig, flux_orig, R)

    print(1 / (np.mean(np.diff(lambda_new)) / np.mean(lambda_new)))

    velocity_per_pxl = np.mean(np.diff(grid_v))

    freq_grid = np.fft.fftfreq(len(lambda_new))

    kernel = np.exp(-2 * 1j * np.pi * freq_grid[None, :] * velocity[:, None] / velocity_per_pxl)

    fi = np.fft.fft(flux_new)

    res_flux = np.fft.ifft(fi[None, :] * kernel).real

    for i in range(len(velocity)):
        plt.plot(lambda_new, res_flux[i])

    plt.show()

    # # Визуализация
    # plt.figure(figsize=(12, 5))
    # plt.plot(lambda_orig, flux_orig, label="Исходный спектр (R~∞, v=0 км/с)", alpha=0.5)
    # plt.plot(lambda_new_no_vel, flux_new_no_vel, label=f"R={R}, v=0 км/с", lw=2, linestyle='--')
    # plt.plot(lambda_new, flux_new, label=f"R={R}, v={velocity} км/с", lw=2, color='red')
    # plt.xlabel("Длина волны (Å)")
    # plt.ylabel("Поток")
    # plt.legend()
    # plt.title("Спектр после понижения разрешения и доплеровского смещения")
    # plt.grid()
    # plt.show()
