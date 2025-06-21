import os
import h5py
import numpy as np

class AstrophysicalModelConstants:

    def __init__(self):

        self.minimum_temperature_spot = 3200
        self.maximum_temperature_spot = 5800

        self.minimum_temperature_star = 4000
        self.maximum_temperature_star = 5500

        self.minimum_num_of_phases = 6
        self.maximum_num_of_phases = 30

        self.minimum_rotation_velocity = 10
        self.maximum_rotation_velocity = 300

        self.minimum_inclination_angle = 10
        self.maximum_inclination_angle = 85

        self.standard_model_spectrum_file_name = 'model_spec.zarr'

        file_path = './Star_maps/num_of_side_16_num_of_star_1e5/stars_T_spots.h5'

        if os.path.exists(file_path):
            print("Map compute complete, normalize parameters is computing")
            self.standard_map_stars_file_name = file_path

            map_data = h5py.File(file_path, 'r')

            num_of_star = map_data['T'].shape[0]

            mean_temperature_star = [None] * num_of_star

            for i in range(num_of_star):
                mean_temperature_star[i] = np.mean(np.array(map_data['T'][i]))

            map_data.close()

            self.average_temperature = np.mean(mean_temperature_star)
            self.std_temperature = 2.0 * np.sqrt(np.var(mean_temperature_star))

            mean_temperature_star = None

        else:
            print("Map data do not exist.")

if __name__ == '__main__':
    params = AstrophysicalModelConstants()

