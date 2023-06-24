# tf-play-rnn
    Isaac Perks
    06-23-2023

# Description

Using a RNN to generate a play built with TensorFlow Shakespear dataset based on TensorFlow tutorial

- Import dataset and process data for model use
    - Open dataset from google api, read, and decode as utf-8
    - Sort our set of text given from dataset
    - Save the length of our vocab set for later
    - Create conversion for int to char and char to int for later
    - Convert text values to integer representations
    - Build our dataset using converted text
    - Create sequences of batch sizes from dataset
    - Split these batches into sets of 0 to n-1 and 0 to n
- Create our model
    - Shuffle our dataset of BUFFER size and create batches of BATCH_SIZE
    - Build our model using BuildModel function with constant values
    - Compile our model with adam optimizer and own loss function
- Train main model
    - Fit data while saving checkpoints and providing history variable data
    - Save checkpoints for future weights and data use
- Build second model
    - Load checkpoint weights
    - Rebuild from BuildModel function with batch size of 1 instead
    - Set TF shape as 1, None for individual characters
- Use GenerateText to take a text input and provide models prediction of a play

- Text2Int
    - Converts given text to its integer representation using our initial conversions
- Int2Text
    - Inverse of above
- SplitInputChunk
    - Takes chunk and returns a tuple of text data [:-1] and [1:]
- Loss
    - Creates a loss function that works for our models nested input/output batches
    - Uses labels & logits from a nodes array of data
- BuildModel(funcVocabLen, funcEmbeddingDim, funcRnnUnits, funcBatchSize)
    - Creates rnn model using given values for Embedding, LSTM, and Dense layers
        - batch size is used for embedding layers batch shape
        - vocabLen and embeddingDum directly used in Embedding layer inputs of model
        - RnnUnits used for LSTM layer input
        - Dense layer set to vocabLen for size
    - returns model created with these 3 layers and given inputs as parameters
- GenerateText(funcModel, funcStartString, funcGenerationVal = 800)
    - Uses a given model, initial string, and length to generate text
        - funcModel needs to be a model built to take single string for input
        - StartString provides initial data for predictions
        - GenerationVal gives the length of our generated text
            - Provided default of 800 as a recommended initial generation size
    - returns the start string with appended predictions for the rest of the play
