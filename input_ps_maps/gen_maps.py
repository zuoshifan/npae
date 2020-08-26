import numpy as np
import h5py
import healpy as hp
import rotate


with h5py.File('../750MHz_maps/sim_pointsource_512_750MHz.hdf5', 'r') as f:
    in_map = f['map'][:]

# print in_map.shape


np.random.seed(0)

N = 1000
for i in range(N):
    if i == 0:
        # use original map for i = 0
        rot = None
    else:
        lon = 360 * np.random.rand() # degree
        lat = 180 * np.random.rand() # degree
        psi = 360 * np.random.rand() # degree
        rot = (lon, lat, psi)
        # print rot

    out_map = rotate.rotate_map(in_map[0, 0], rot=rot)
    # print out_map.shape

    with h5py.File('./rotate_pointsource_512_750MHz_%04d.hdf5' % i, 'w') as f:
        f.create_dataset('map', data=out_map.reshape(in_map.shape))