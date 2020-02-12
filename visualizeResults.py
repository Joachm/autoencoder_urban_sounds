import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt


def seeResults(num=10):
    val = np.load('x_test.npy')

    model = load_model('convAuto.hdf5')

    preds = model.predict(val[:num])

    for i in range(num):
        fig, axs = plt.subplots(2)
        fig.suptitle('Sound From Test Set', fontsize=16)
        axs[0].plot(val[i])
        axs[0].set_title('original')
        axs[1].plot(preds[i])
        axs[1].set_title('decoded')
        plt.show()

