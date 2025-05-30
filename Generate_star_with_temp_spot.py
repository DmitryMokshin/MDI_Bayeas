import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import h5py
from tqdm import tqdm
import time

class Surface(object):
    def __init__(self, num_of_side, num_of_star, minimum_temperature_spot=3200, maximum_temperature_spot=5800,
                 minimum_temperature_star=4000, maximum_temperature_star=5500, file_name=None):
        self.num_of_side = num_of_side
        self.num_of_pixels = hp.nside2npix(self.num_of_side)
        self.num_of_star = num_of_star

        self.min_temperature = minimum_temperature_spot
        self.max_temperature = maximum_temperature_spot

        self.maps_of_temperature = np.zeros((self.num_of_star, self.num_of_pixels))

        num_of_spots = np.random.randint(low=1, high=11, size=self.num_of_star)
        temperature_of_star = np.random.uniform(low=minimum_temperature_star, high=maximum_temperature_star, size=self.num_of_star)

        smooth = np.random.uniform(low=0.1, high=0.2, size=self.num_of_star)

        for i in tqdm(range(self.num_of_star)):
            self.maps_of_temperature[i] = self.random_star(num_of_spots[i], temperature_of_star[i], smooth[i])

        if file_name is not None:
            file_surface_out = h5py.File(file_name, 'w')
            ds = file_surface_out.create_dataset('T', (self.num_of_star, self.num_of_pixels))
            ds[:] = self.maps_of_temperature
            file_surface_out.close()

    def random_star(self, num_of_spots, temperature_of_star, smooth):
        surface_map_temperature = temperature_of_star * np.ones(self.num_of_pixels)

        for i in range(num_of_spots):
            vec = np.random.randn(3)
            vec /= np.linalg.norm(vec)
            radius = np.random.triangular(left=0.1, mode=0.1, right=1.0)

            spot_temperature_min = self.min_temperature
            spot_temperature_max = np.min([1.2 * temperature_of_star, self.max_temperature])
            T_spot = np.random.uniform(low=spot_temperature_min, high=spot_temperature_max)

            px = hp.query_disc(self.num_of_side, vec, radius, nest=False)

            surface_map_temperature[px] = T_spot

        surface_map_temperature = hp.sphtfunc.smoothing(surface_map_temperature, sigma=smooth, verbose=False)

        surface_map_temperature = hp.pixelfunc.reorder(surface_map_temperature, inp='RING', out='NESTED')

        return surface_map_temperature


if __name__ == '__main__':

    start_time = time.time()

    tmp = Surface(num_of_side=32, num_of_star=100000,
                  file_name='./Star_maps/num_of_side_32_num_of_star_1e5/stars_T_spots.h5')

    print('Time compute:', time.time() - start_time)