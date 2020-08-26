# import os
import numpy as np
import h5py
import healpy as hp
import keras
# from keras.models import load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.optimizers import Adam
# import pickle

import matplotlib
matplotlib.use('Agg')
try:
    # for new version matplotlib
    import matplotlib.style as mstyle
    mstyle.use('classic')
except ImportError:
    pass
import matplotlib.pyplot as plt



# read in dataset
train = np.load('./training_dataset_256x256_aug/train.npy')
val = np.load('./training_dataset_256x256_aug/val.npy')
test = np.load('./training_dataset_256x256_aug/test.npy')

# autoencoder
nsample, npix, _, _ = train.shape

input_img = Input(shape=(npix, npix, 1))  #  1 for one pol, adapt this if using `channels_first` image data format

x = Conv2D(32, (3, 3), padding='same')(input_img)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), padding='same')(x)
x = Activation('relu')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(64, (3, 3), padding='same')(encoded)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(1, (3, 3), padding='same')(x)
decoded = Activation('linear')(x)


model = Model(input_img, decoded)
# model.compile(optimizer='adam', loss='binary_crossentropy')
# model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.compile(optimizer='adam', loss='mse')
# model.compile(optimizer='adam', loss='mae') # use mae to promote sparsity for point sources


batch_size = 32
# epochs = 100
epochs = 1

for ii in range(50):

    # es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
    # chkpt = saveDir + 'AutoEncoder_Cifar10_denoise_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
    # cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    history = model.fit(train, train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(val, val),
                        # callbacks=[es_cb,],
                        # callbacks=[es_cb, cp_cb],
                        shuffle=True)

    # save weights
    model.save_weights('./training_dataset_256x256_aug/model_weights_%04d.h5' % ii)

    predict = model.predict(train[0].reshape(1, npix, npix, 1))

    # plot predict
    plt.figure()
    plt.imshow(predict[0, :, :, 0], origin='lower', aspect='equal', vmin=0, vmax=1)
    plt.colorbar()
    plt.savefig('./training_dataset_256x256_aug/predict_32+64_%04d.png' % ii)
    plt.close()
