import keras
from keras.preprocessing import process
import tensorflow as tf
import os
import numpy as np

SEQLEN        = 100
BATCH_SIZE    = 64
EMBEDDING_DIM = 256
RNN_UNITS     = 1024
BUFFER        = 10000

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
vocab = sorted(set(text))
vocabLen = len(vocab)

char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
textValues = Text2Int(text)

examplesPerEpoch = len(text)//(SEQLEN+1)

charDataset = tf.data.Datasets.from_tensor_slices(textValues)
sequences = charDataset.batch(SEQLEN+1, drop_remainder=True)
dataset = sequences.map(SplitInputChunk)

data = dataset.shuffle(BUFFER).batch(BATCH_SIZE, drop_remainder=True)
rnnModel = BuildModel(vocabLen, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
rnnModel.compile(optimizer='adam', loss=Loss)

checkpointDir = './training_checkpoints'
checkpointPrefix = os.path.join(checkpointDir, "ckpt_{epoch}")
checkpointCallback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpointDir,
    save_weights_only = True
)

modelHistory = rnnModel.fit(data, epochs=50, callbacks=[checkpointCallback])

rnnModelSingle = BuildModel(vocabLen, EMBEDDING_DIM, RNN_UNITS, 1)
rnnModelSingle.load_weights(tf.train.latest_checkpoint(checkpointDir))
rnnModelSingle.build(tf.TensorShape([1, None]))

def GenerateText(funcModel, funcStartString, funcGenerationVal = 800):
    tempature = 1.0

    inputEval = [char2idx[s] for s in funcStartString]
    inputEval = tf.expand_dims(inputEval, 0)

    generatedText = []
    funcModel.reset_states()

    for i in range(funcGenerationVal):
        predictions = funcModel(inputEval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions/tempature
        predictedChar = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        inputEval = tf.expand_dims([predictedChar, 0])
        generatedText.append(idx2char[predictedChar])
        
    return (funcStartString + ''.join(generatedText))

def BuildModel(funcVocabLen, funcEmbeddingDim, funcRnnUnits, funcBatchSize):
    model = tf.keras.sequential([
        tf.keras.layers.Embedding(
            funcVocabLen, funcEmbeddingDim, 
            batch_input_shape = [funcBatchSize, None]
        ),
        tf.keras.layers.LSTM(
            funcRnnUnits,
            return_sequences = True,
            stateful = True,
            recurrent_initializer = 'glorot_uniform'
        ),
        tf.keras.layers.Dense(funcVocabLen)
    ])
    return model

def Loss(funcLabels, funcLogits):
    return tf.keras.losses.sparse_categorical_crossentropy(funcLabels, funcLogits, from_logits=True)

def SplitInputChunk(funcChunk):
    inputText = chunk[:-1]
    outputText = chunk[1:]
    return inputText, outputText

def Text2Int(text):
    return np.array([char2idx[c] for c in text])

def Int2Text(values):
    if type(values) != np.array():
        values = values.numpy()
    return ''.join(idx2char[values])