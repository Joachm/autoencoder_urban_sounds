import numpy as np


def split(train=(0,6500), val=(6500,7000), test=(7000,-1)):
    data = np.load('sounds.npy')
    labels = np.load('labels.npy')

    ## shuffle data set
    indx = np.arange(data.shape[0])
    np.random.shuffle(indx)

    data  = data[indx]
    labels = data[indx]


    ## split and save the data into train, validation and test.
    x_train = data[train[0]:train[1]]
    x_val = data[val[0]:val[1]]
    x_test = data[test[0]:test[1]]

    y_train = labels[train[0]:train[1]]
    y_val = labels[val[0]:val[1]]
    y_test = labels[test[0]:test[1]]


    np.save('x_train.npy', x_train)
    np.save('x_val.npy', x_val)
    np.save('x_test.npy', x_test)

    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    np.save('y_test.npy', y_test)

