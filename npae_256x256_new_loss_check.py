# import os
import numpy as np
import h5py
import healpy as hp
import keras
from keras import backend as K
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



# def sigmoid_mse_loss():
#     def sigmoid_mse(y_true, y_pred):
#         return K.mean(K.sigmoid(y_true) * K.square(y_pred - y_true))
#     return sigmoid_mse

def new_mse_loss(lmb=0):
    def new_mse(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true)) + lmb * K.mean(K.square(y_pred))
    return new_mse


# read in dataset
train = np.load('./training_dataset_256x256/train.npy')
val = np.load('./training_dataset_256x256/val.npy')
test = np.load('./training_dataset_256x256/test.npy')


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
# model.compile(optimizer='adam', loss='mse')
# model.compile(optimizer='adam', loss=sigmoid_mse_loss())
lmb = 0.1
model.compile(optimizer='adam', loss=new_mse_loss(lmb))
# model.compile(optimizer='adam', loss='mae') # use mae to promote sparsity for point sources


batch_size = 32
# epochs = 100
epochs = 1

loss = []
val_loss = []
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
    model.save_weights('./training_dataset_256x256/model_weights_%04d.h5' % ii)

    predict = model.predict(train[0].reshape(1, npix, npix, 1))

    # compare loss
    loss1 = np.mean((predict - train[0].reshape(1, npix, npix, 1))**2)
    loss2 = np.mean((predict)**2)
    print 'loss: ', loss1, loss2, loss1 / loss2

    # plot predict
    plt.figure()
    plt.imshow(predict[0, :, :, 0], origin='lower', aspect='equal', vmin=0, vmax=1)
    plt.colorbar()
    plt.savefig('./training_dataset_256x256/predict_new_mse_32+64_%04d.png' % ii)
    plt.close()

    # save loss
    loss.append(history.history['loss'][-1])
    val_loss.append(history.history['val_loss'][-1])

    # plot history for loss
    plt.figure()
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.plot(loss)
    plt.plot(val_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig('./training_dataset_256x256/loss_%04d.png' % ii)
    plt.close()


# save loss for plot
with h5py.File('./training_dataset_256x256/history_loss.hdf5', 'w') as f:
    f.create_dataset('loss', data=history.history['loss'])
    f.create_dataset('val_loss', data=history.history['val_loss'])
