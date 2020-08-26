import os
import numpy as np
import h5py
import healpy as hp
import rotate
from caput import mpiutil

import matplotlib
matplotlib.use('Agg')


if mpiutil.size != 20:
    raise RuntimeError('Need 20 processes')

np.random.seed(0)
rands = np.random.rand(1000, 20)


N = 1000
m = 20 # number of augmentation by rotation
n = 256
dataset = np.zeros((N/20, m, n, n, 1))
# for i in range(N):
ln, s, e = mpiutil.split_local(N)
for li, i in enumerate(range(s, e)):
    if mpiutil.rank0:
        print '%d of %d ...' % (li, ln)
    # read in reconstructed map
    with h5py.File('./output_sim_750MHz/map/ts/ts_%04d/map_full.hdf5' % i, 'r') as f:
        rec_map = f['map'][0, 0]

    for j in range(m):
        if j == 0:
            rot_map = rec_map.copy()
        else:
            # angle = 360 * np.random.rand()
            angle = 360 * rands[i, j]
            rot_map = rotate.np_rotate(rec_map.copy(), angle)

        # project to 256 x 256 image
        rec_img = hp.orthview(rot_map, rot=(0, 90, 0), xsize=400, half_sky=True, return_projected_map=True)[72:328, 72:328] # rot to make NCP at the center
        dataset[li, j, :, :, 0] = rec_img.data # only data of the masked array


# save dataset of rank0
if mpiutil.rank0:
    if not os.path.isdir('./training_dataset_256x256_aug'):
        os.mkdir('./training_dataset_256x256_aug')
    np.save('./training_dataset_256x256_aug/dataset_rank0.npy', dataset)

# gather dataset to rank 0
dataset = mpiutil.gather_array(dataset, axis=0, root=0)

if mpiutil.rank0:
    dataset = dataset.reshape((N*m, n, n, 1))
    inds = np.arange(N*m)
    np.random.shuffle(inds[1:]) # leave 0 (the true NP) unchanged
    dataset = dataset[inds] # randomly shuffle the datasets

    print dataset.shape

    train = dataset[:12000] # train dataset
    val = dataset[12000:18000] # validation dataset
    test = dataset[18000:] # test dataset

    if not os.path.isdir('./training_dataset_256x256_aug'):
        os.mkdir('./training_dataset_256x256_aug')

    # save train, val, test dataset
    np.save('./training_dataset_256x256_aug/train.npy', train)
    np.save('./training_dataset_256x256_aug/val.npy', val)
    np.save('./training_dataset_256x256_aug/test.npy', test)
