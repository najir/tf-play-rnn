import keras
from keras.preprocessing import process
import tensorflow as tf
import os
import numpy as np

SEQLEN = 100
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
vocab = sorted(set(text))
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
textValues = Text2Int(text)

examplesPerEpoch = len(text)//(SEQLEN+1)

charDataset = tf.data.Datasets.from_tensor_slices(textValues)


def Text2Int(text):
    return np.array([char2idx[c] for c in text])

def Int2Text(values):
    if type(values) != np.array():
        values = values.numpy()
    return ''.join(idx2char[values])