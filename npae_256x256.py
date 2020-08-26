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
import matplotlib.pyplot as plt



# read in dataset
train = np.load('./training_dataset_256x256/train.npy')
val = np.load('./training_dataset_256x256/val.npy')
test = np.load('./training_dataset_256x256/test.npy')


# autoencoder
nsample, npix, _, _ = train.shape

input_img = Input(shape=(npix, npix, 1))  #  1 for one pol, adapt this if using `channels_first` image data format

x = Conv2D(32, (3, 3), padding='same')(input_img)
# x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), padding='same')(x)
# x = BatchNormalization()(x)
x = Activation('relu')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (3, 3), padding='same')(encoded)
# x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), padding='same')(x)
# x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(1, (3, 3), padding='same')(x)
# x = BatchNormalization()(x)
decoded = Activation('linear')(x)


model = Model(input_img, decoded)
# model.compile(optimizer='adam', loss='binary_crossentropy')
# model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.compile(optimizer='adam', loss='mse')
# model.compile(optimizer='adam', loss='mae') # use mae to promote sparsity for point sources


batch_size = 32
# epochs = 100
epochs = 20

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

# evaluate with test dataset
score = model.evaluate(test, test, verbose=1)
print score

# predict for test dataset
predict = model.predict(test)
print predict.shape

# save predict
np.save('./training_dataset_256x256/predict.npy', predict)


# save model
json_string = model.to_json()
with open('./training_dataset_256x256/model.json', 'w') as f:
        f.write(json_string)
# save weights
model.save_weights('./training_dataset_256x256/model_weights.h5')



# save loss for plot
with h5py.File('./training_dataset_256x256/history_loss.hdf5', 'w') as f:
    f.create_dataset('loss', data=history.history['loss'])
    f.create_dataset('val_loss', data=history.history['val_loss'])


# visualization
# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig('./training_dataset_256x256/loss.png')
plt.close()

# # summarize history for accuracy
# plt.figure()
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.savefig('./training_dataset_256x256/accuracy.png')
# plt.close()
