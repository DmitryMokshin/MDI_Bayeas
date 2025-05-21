import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import zarr
import scipy.interpolate as interp
import time


def poly_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


class DopplerImaging(object):
    def __init__(self, num_of_side, regions=None, root_models=None, model_spec='model_spec.zarr'):
        """
        This class does Doppler Imaging using several techniques

        Parameters
        ----------
        num_of_side : int
            number of sides in the Healpix pixellization.
        regions : array [[bounds]]
            array wavelength for line estimate
        model_spec : str
            name of archive with stellar atmosphere
        root_models : str
            way for directory with models
        """

        self.num_of_side = int(num_of_side)
        self.hp_npix = hp.nside2npix(num_of_side)

        # Generate the indices of all healpix pixels
        self.indices = np.arange(hp.nside2npix(num_of_side), dtype="int")
        self.n_healpix_pxl = len(self.indices)

        self.polar_angle, self.azimuthal_angle = hp.pixelfunc.pix2ang(self.num_of_side, np.arange(self.n_healpix_pxl),
                                                                      nest=True)
        self.polar_angle = 2 * self.polar_angle / np.pi - 1.0

        self.pixel_vectors = np.array(hp.pixelfunc.pix2vec(self.num_of_side, self.indices, nest=True))

        self.rotation_velocity = np.cross(np.array([0.0, 0.0, 1.0])[:, None], self.pixel_vectors, axisa=0, axisb=0,
                                          axisc=0)

        self.vec_boundaries = np.zeros((3, 4, self.n_healpix_pxl))
        for i in range(self.n_healpix_pxl):
            self.vec_boundaries[:, :, i] = hp.boundaries(self.num_of_side, i, nest=True)

        print(" Reading model spectra")
        if root_models is None:
            file_archive = zarr.open(model_spec, 'r')
        else:
            file_archive = zarr.open(f'{root_models}/' + model_spec, 'r')

        self.temperature_model = file_archive['T'][:]
        self.mu_model = file_archive['mu'][:]
        self.radial_velocity = file_archive['v'][:]
        self.velocity_axis_model = file_archive['velaxis'][:]
        self.wavelength_model = file_archive['wavelength'][:]
        self.spectrum_model = file_archive['spec'][:]

        ind = np.argsort(self.wavelength_model)
        self.wavelength_model = self.wavelength_model[ind]
        self.spectrum_model = self.spectrum_model[:, :, :, ind]

        # Разбитие спектры на регионы

        if regions is not None:
            n_regions = len(regions)
            for i in range(n_regions):
                print(f'Extracting region {regions[i]}')
                region = regions[i]
                left = np.argmin(np.abs(self.wavelength_model - region[0]))
                right = np.argmin(np.abs(self.wavelength_model - region[1]))
                if i == 0:
                    wl = self.wavelength_model[left:right]
                    spectrum = self.spectrum_model[:, :, :, left:right]
                else:
                    wl = np.append(wl, self.wavelength_model[left:right])
                    spectrum = np.append(spectrum, self.spectrum_model[:, :, :, left:right], axis=-1)

            self.wavelength_model = wl
            self.spectrum_model = spectrum

        self.num_vel, self.num_T, self.num_mu, self.num_wave = self.spectrum_model.shape

        self.T = np.zeros(self.n_healpix_pxl)

    def trilinear_interpolate(self, v, T, mu):
        """
        :param v: float
            value from grid velocity for spectrum
        :param T: value
            value temperature of spectrum
        :param mu:
            value cos angle mu
        :return: float
            reinterpolate intensity
        """
        ind_v0 = np.searchsorted(self.radial_velocity, v) - 1
        ind_T0 = np.searchsorted(self.temperature_model, T) - 1
        ind_m0 = np.searchsorted(self.mu_model, mu) - 1

        vd = (v - self.radial_velocity[ind_v0]) / (self.radial_velocity[ind_v0 + 1] - self.radial_velocity[ind_v0])
        Td = (T - self.temperature_model[ind_T0]) / (
                    self.temperature_model[ind_T0 + 1] - self.temperature_model[ind_T0])
        md = (mu - self.mu_model[ind_m0]) / (self.mu_model[ind_m0 + 1] - self.mu_model[ind_m0])

        c000 = self.spectrum_model[ind_v0, ind_T0, ind_m0, :]
        c001 = self.spectrum_model[ind_v0, ind_T0, ind_m0 + 1, :]
        c010 = self.spectrum_model[ind_v0, ind_T0 + 1, ind_m0, :]
        c011 = self.spectrum_model[ind_v0, ind_T0 + 1, ind_m0 + 1, :]
        c100 = self.spectrum_model[ind_v0 + 1, ind_T0, ind_m0, :]
        c101 = self.spectrum_model[ind_v0 + 1, ind_T0, ind_m0 + 1, :]
        c110 = self.spectrum_model[ind_v0 + 1, ind_T0 + 1, ind_m0, :]
        c111 = self.spectrum_model[ind_v0 + 1, ind_T0 + 1, ind_m0 + 1, :]

        f1 = (1.0 - vd[:, None])
        f2 = vd[:, None]

        c00 = c000 * f1 + c100 * f2
        c01 = c001 * f1 + c101 * f2
        c10 = c010 * f1 + c110 * f2
        c11 = c011 * f1 + c111 * f2

        f1 = (1.0 - Td[:, None])
        f2 = Td[:, None]

        c0 = c00 * f1 + c10 * f2
        c1 = c01 * f1 + c11 * f2

        c = c0 * (1.0 - md[:, None]) + c1 * md[:, None]

        return c

    def compute_stellar_spectrum(self, temperature, los, omega=0.0, clv=False, reinterpolate_lambda=None):
        """
        Compute the averaged spectrum on the star for a given temperature map and for a given rotation

        Parameters
        ----------
        los : array [n_phases, 3]
            Angles defining the LOS for each phase
        omega : float
            Rotation velocity in km/s, by default 0.0
        clv : bool, optional
            [description], by default False

        Returns
        -------
        float
            Loss function
        array of floats [n_phases, n_lambda]
            Synthetic Stokes parameters
        """

        self.T = temperature
        self.clv = clv
        self.omega = omega

        self.los = np.zeros_like(los)
        self.los[:, 0] = np.rad2deg(los[:, 1])
        self.los[:, 1] = 90 - np.rad2deg(los[:, 0])

        self.los_vec = hp.ang2vec(los[:, 0], los[:, 1])

        self.n_phases, _ = los.shape

        self.visible_pixels = []
        self.area = []
        self.mu = []
        self.total_area = np.zeros(self.n_phases)
        self.vel = []
        self.kernel = []

        self.stokesi = np.zeros((self.n_phases, self.num_wave))

        self.epsilon = 0.01

        print('Start compute star spectrum')

        for loop in range(self.n_phases):
            visible_pixels = hp.query_disc(self.num_of_side, self.los_vec[loop, :], np.pi / 2.0 - self.epsilon,
                                           nest=True)

            area_projected = np.sum(self.pixel_vectors * self.los_vec[loop, :][:, None], axis=0)

            mu = area_projected[visible_pixels]
            temperature = self.T[visible_pixels]

            self.total_area[loop] = np.sum(mu)

            vel = np.sum(self.rotation_velocity * self.los_vec[loop, :][:, None], axis=0)
            vel = self.omega * vel[visible_pixels]

            interp_spec = self.trilinear_interpolate(vel, temperature, mu)

            self.stokesi[loop, :] = np.sum(interp_spec * mu[:, None], axis=0) / self.total_area[loop] #+ np.random.normal(0, 1 / 100, self.num_wave)

        if reinterpolate_lambda is not None:
            for loop in range(self.n_phases):
                f = interp.interp1d(self.wavelength_model, self.stokesi[loop, :])
                tmp = f(reinterpolate_lambda)
                if loop == 0:
                    interp_spec = np.zeros((self.n_phases, tmp.shape[0]))
                interp_spec[loop, :] = tmp
            return interp_spec
        else:
            return self.stokesi


if __name__ == "__main__":
    num_side = 2 ** 6

    n_phases = 20
    inclination = 30.0
    los = np.zeros((n_phases, 2))
    for i in range(n_phases):
        los[i, :] = np.array([inclination * np.pi / 180.0, 2.0 * np.pi / n_phases * i])

    star = DopplerImaging(num_side)

    T_surface = 12000.0 * np.ones(hp.nside2npix(num_side))

    start = time.time()
    stokesi = star.compute_stellar_spectrum(T_surface, los, omega=50.0)
    print('Compute time', time.time() - start)

    for i in range(n_phases):
        plt.plot(star.wavelength_model / 1.0E+8, stokesi[i, :] / np.max(stokesi[i, :]), label=f'Angles {180.0 / np.pi * los[i, :]}')
        plt.legend()
        plt.grid()

    plt.show()
