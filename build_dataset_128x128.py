import os
import numpy as np
import h5py
import healpy as hp

import matplotlib
matplotlib.use('Agg')


N = 1000
n = 128
dataset = np.zeros((N, n, n, 1))
for i in range(N):
    # read in reconstructed map
    with h5py.File('./output_sim_750MHz/map/ts/ts_%04d/map_full.hdf5' % i, 'r') as f:
        rec_map = f['map'][0, 0]

    # project to 128 x 128 image
    rec_img = hp.orthview(rec_map, rot=(0, 90, 0), xsize=200, half_sky=True, return_projected_map=True)[36:164, 36:164] # rot to make NCP at the center
    dataset[i, :, :, 0] = rec_img.data # only data of the masked array

# print dataset.shape

train = dataset[:600] # train dataset
val = dataset[600:900] # validation dataset
test = dataset[900:] # test dataset

if not os.path.isdir('./training_dataset_128x128'):
    os.mkdir('./training_dataset_128x128')

# save train, val, test dataset
np.save('./training_dataset_128x128/train.npy', train)
np.save('./training_dataset_128x128/val.npy', val)
np.save('./training_dataset_128x128/test.npy', test)
