from prepareAllData import *
from splitData import *
from oneDConvAutoencoder import *
from visualizeResults import *


prepare(metaPath='UrbanSound8K/metadata/UrbanSound8K.csv', dataPath='UrbanSound8K/all/', minLen=2.5,sampleLen=2., numSamples=44100)

split(train=(0,6500), val=(6500,7000), test=(7000,-1))

trainAutoencoder(numSamples=44100)

seeResults(num=10)
