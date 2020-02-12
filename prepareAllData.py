import librosa as lr
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt



def prepare(metaPath='UrbanSound8K/metadata/UrbanSound8K.csv', dataPath='UrbanSound8K/all/',minLen=2.5,sampleLen=2., numSamples=44100):
    ## use only samples that are at least 2.5 seconds long
    data = pd.read_csv(metaPath)
    valid_data = data[['slice_file_name', 'fold' ,'classID', 'class']][ data['end']-data['start'] >= minLen ]

    #print(len(valid_data)) : 7579

    X = np.zeros((len(valid_data), numSamples, 1))
    Y = np.zeros((len(valid_data), 10))

    ## get two seconds of the samples with sample rate 22050.
    ## when rate is smaller, pad with zeros
    i = 0
    for name in valid_data['slice_file_name']:
        sample = lr.load(dataPath+name, duration=sampleLen)[0]
        X[i, :len(sample)] = sample.reshape(len(sample),1)
        Y[i,int(data.classID[data.slice_file_name==name])] = 1
        i+=1
        if i % 500==0:
            print(i)

    np.save('sounds.npy', X)
    np.save('labels.npy', Y)


