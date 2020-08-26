import os
import numpy as np
import h5py
import healpy as hp

import matplotlib
matplotlib.use('Agg')


N = 1000
npix = 512
dataset = np.zeros((N, npix, npix, 1))
for i in range(N):
    # read in reconstructed map
    with h5py.File('./output_sim_750MHz/map/ts/ts_%04d/map_full.hdf5' % i, 'r') as f:
        rec_map = f['map'][0, 0]

    # project to 512 x 512 image
    rec_img = hp.orthview(rec_map, rot=(0, 90, 0), half_sky=True, return_projected_map=True)[144:656, 144:656] # rot to make NCP at the center
    dataset[i, :, :, 0] = rec_img.data # only data of the masked array

# print dataset.shape

train = dataset[:600] # train dataset
val = dataset[600:900] # validation dataset
test = dataset[900:] # test dataset

if not os.path.isdir('./training_dataset_512x512'):
    os.mkdir('./training_dataset_512x512')

# save train, val, test dataset
np.save('./training_dataset_512x512/train.npy', train)
np.save('./training_dataset_512x512/val.npy', val)
np.save('./training_dataset_512x512/test.npy', test)
