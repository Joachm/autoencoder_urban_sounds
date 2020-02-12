import numpy as np
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Conv1D, Dropout,BatchNormalization, Input, Reshape, Flatten, MaxPooling1D, GlobalMaxPooling1D, UpSampling1D


def trainAutoencoder(numSamples=44100):
    x_train = np.load('x_train.npy')
    x_val = np.load('x_val.npy')
    

    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=10, input_shape=(numSamples,1), padding='causal', activation='relu', strides=2))

    model.add(MaxPooling1D())

    model.add(Conv1D(8,5,padding='causal', activation='relu'))
    model.add(BatchNormalization())

    model.add(UpSampling1D(2))

    model.add(Conv1D(16,5,padding='causal', activation='relu'))
    model.add(UpSampling1D(2))

    model.add(Conv1D(1,5,padding='causal', activation='linear'))

    model.compile(loss="mse",
            optimizer='adam')

    model.summary()

    model.fit(x_train, x_train,
            epochs = 30,
            batch_size = 128,
            shuffle=True,
            validation_data=(x_val, x_val),
            verbose=1)

    model.save('convAuto.hdf5')

